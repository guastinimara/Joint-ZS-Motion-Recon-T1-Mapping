#%%
"""Joint zero-shot motion estimation, image reconstruction and T1 mapping on simulated XCAT data.

The simulated k-space (kdata_simul.h5) is downloaded from Zenodo. The script alternates
motion field estimation, low-rank (SVD) regularization, motion-corrected CG reconstruction
and motion-corrected T1 fitting. Runs on GPU if available.
"""
import os
import sys
import tempfile
from pathlib import Path

import mrpro
import torch
import zenodo_get

# __file__ is not defined when running interactively, where the working directory is examples/scripts.
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'src')]

from cardiac_motion.MotionEstimationWithInverse import MotionEstimationWithInverse
from operators.construct_recon_operators import compute_csm
from operators.MotionOperator import MotionOperator
from parameters_estimation.T1ParameterEstimator import T1ParameterEstimator
from regularization.svd import compute_svd, update_svd
from utils.MaskImgSize import MaskImgSize
from utils.rr_threshold_functions import find_valid_rr_threshold, update_rr_expansion_and_compute_rr_mask
from utils.save import save_patient_results


# data
ZENODO_RECORD_ID = '23372515'
N_LINES_PER_DYNAMIC = 16  # k-space lines per dynamic image
N_LINES_OVERLAP = 8  # lines shared by consecutive dynamics (sliding window)
CSM_SMOOTHING_WIDTH = 15

# acquisition parameters of this dataset
TAU = torch.tensor(0.028, dtype=torch.float32)  # delay after inversion pulse (s)

# reconstruction
N_ITERATIONS = 4
CG_ITER_INITIAL = 8
CG_ITER_RECON = 5
CG_TOL = 1e-5
LAMBDA_1 = 0.1  # weight of the T1 model term
LAMBDA_2 = 0.05  # weight of the SVD (low-rank) term
LAMBDA_3 = 0.05 * 0.5  # L1 weight of the SVD coefficients

# output
SAVE_RESULTS = True
OUTPUT_DIR = Path.cwd()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the motion network is trained from a random initialization, so fix the seed to make the run reproducible.
torch.manual_seed(0)

#%%
# KData.from_file needs a path, so the download goes to a temporary folder that is deleted after loading.
with tempfile.TemporaryDirectory() as tmp_dir:
    zenodo_get.download(
        record=ZENODO_RECORD_ID,
        output_dir=tmp_dir,
        file_glob=('kdata_simul.h5', 'rr_idx.pt'),
        timeout=120,  # the default of 15 s is too short behind slow proxies
        retry_attempts=3,
    )
    kdata_simul = mrpro.data.KData.from_file(
        os.path.join(tmp_dir, 'kdata_simul.h5'),
        mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
    ).to(device)
    # cardiac phase (RR-interval) index of each dynamic.
    rr_idx = torch.load(os.path.join(tmp_dir, 'rr_idx.pt'))


# split k-space data into dynamics
sorted_idx = kdata_simul.header.acq_info.acquisition_time_stamp.squeeze().argsort()
split_idx = mrpro.utils.split_idx(sorted_idx, N_LINES_PER_DYNAMIC, N_LINES_OVERLAP)
kdata_simul_dynamic = kdata_simul[..., split_idx, :]

# compute acquisition operator for reconstruction
csm_op, img_size = compute_csm(kdata_simul, smoothing_width=CSM_SMOOTHING_WIDTH, device=device)
dcf = mrpro.data.DcfData.from_traj_voronoi(kdata_simul_dynamic.traj).to(device)
dcf_sqrt_op = mrpro.operators.DensityCompensationOp(torch.sqrt(dcf.data))
dcf_op = dcf.as_operator()
fourier_op = mrpro.operators.FourierOp.from_kdata(kdata_simul_dynamic).to(device)

acquisition_op = fourier_op @ csm_op
AHWA_op = (dcf_sqrt_op @ acquisition_op).gram

(rhs_no_motion,) = (acquisition_op.H @ dcf_op)(kdata_simul_dynamic.data)

# initial (not motion-corrected) reconstruction of the dynamics
with torch.no_grad():
    (x_moving,) = mrpro.algorithms.optimizers.cg(
        AHWA_op,
        rhs_no_motion,
        initial_value=rhs_no_motion,
        max_iterations=CG_ITER_INITIAL,
        tolerance=CG_TOL,
    )
torch.cuda.empty_cache()

num_dynamics = x_moving.shape[0]

#%%
t1_parameter_estimator = T1ParameterEstimator(kdata=kdata_simul_dynamic, 
                        opt_type='lbfgs',
                        tau=TAU,
                        device=device)
#%%
# initial T1 parameter fitting with ECG-based mask for motion-compensated references.
p_0_ref, x_0_ref, rr_expansion_offsets = find_valid_rr_threshold(
    rr_idx,
    x_moving,
    t1_parameter_estimator,
)
if p_0_ref is None or x_0_ref is None:
    raise RuntimeError('Could not find valid RR threshold.')

# store rr_idx expansion pattern for iterative updates.
base_rr_idx = rr_idx[0].item()
current_rr_expansion = rr_expansion_offsets.copy()

mask_roi = MaskImgSize(h_start=70, w_start=68, h_width=100, w_width=100)

motion_op = MotionOperator(
    times=num_dynamics,
    img_size=img_size,
    device=device,
)
motion_estimation = MotionEstimationWithInverse(
    kdata_dynamic=kdata_simul_dynamic,
    acquisition_op=acquisition_op,
    dcf_sqrt_op=dcf_sqrt_op,
    motion_op=motion_op,
    img_size=img_size,
    times=num_dynamics,
    mask_roi=mask_roi,
    device=device,
)

# initial cardiac motion fields estimation.
motion_estimation.train_net(
    x_moving.abs(),
    x_0_ref,
    episodes=40,
    steps=20,
    full_loss_phi=35,
)
torch.cuda.empty_cache()

# compute current registered dynamics and update current x with it
(x_registered,) = motion_op.M_op_inv(x_moving)
x_registered = x_registered.detach()
x_current = x_registered

# compute mask selection from sinusoidal fit to exclude 15% systolic frames
optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)
mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
    optimal_phi.detach().cpu(),
)

# initial motion-corrected T1 parameter fitting
p, qp = t1_parameter_estimator.fit(x_current, mask=mask_85_percent)

# initial SVD-based computation
x_reg_svd, U_op, _ = compute_svd(x_current, mask_roi)
x_reg_svd = x_reg_svd.detach().unsqueeze(1).unsqueeze(1)
torch.cuda.empty_cache()

print('Initialization completed.')


print('Starting iterative loop.')
# alternate motion estimation, SVD regularization, motion-corrected CG reconstruction
# and motion-corrected T1 estimation.
for iteration in range(1, N_ITERATIONS + 1):
    print(f'Iteration {iteration}/{N_ITERATIONS}')

    # expand the rr_idx selection and compute the mask of the reference dynamics
    mask_ref, current_rr_expansion = update_rr_expansion_and_compute_rr_mask(
        rr_idx,
        base_rr_idx,
        current_rr_expansion,
    )

    if not mask_ref.any():
        p_ref_updated = p_0_ref
        x_ref_updated = x_0_ref
    else:
        # try fitting with current mask
        p_ref_updated, x_ref_updated = t1_parameter_estimator.fit(
            x_current,
            mask=mask_ref,
        )
        # if NaN, fall back to previous values and don't expand further
        if torch.isnan(x_ref_updated).any():
            p_ref_updated = p_0_ref
            x_ref_updated = x_0_ref

    # update cardiac motion fields estimation
    motion_estimation.train_net(
        x_moving.abs(),
        x_ref_updated.detach(),
        episodes=30,
        steps=20,
        full_loss_phi=25,
    )
    torch.cuda.empty_cache()
    print(f'phi^({iteration}) updated.')

    # update SVD regularizer with soft-thresholding
    x_reg_svd, _ = update_svd(
        x_current,
        U_op,
        weight_l1=LAMBDA_3,
        weight_l2=LAMBDA_2,
    )
    print(f'c_j^({iteration}) updated.')

    with torch.no_grad():
        # motion-corrected image reconstruction with updated motion fields and SVD regularization.
        # solve the quadratic subproblem with CG.
        H = lambda y: (
            motion_op.adjoint(AHWA_op(motion_op.forward(y)[0])[0])[0]
            + LAMBDA_1 * y
            + LAMBDA_2 * y,
        )

        (b_0,) = motion_op.adjoint(rhs_no_motion)
        b = (
            b_0
            + LAMBDA_1 * qp
            + LAMBDA_2 * x_reg_svd
        )

        (x_next,) = mrpro.algorithms.optimizers.cg(
            operator=H,
            right_hand_side=b,
            initial_value=x_current,
            max_iterations=CG_ITER_RECON,
            tolerance=CG_TOL,
        )
    x_current = x_next.detach()
    torch.cuda.empty_cache()
    print(f'x^({iteration}) updated.')

    # initialize the parametric maps for further fitting
    p_init = p_ref_updated.detach()

    # automatic extraction of systolic frames
    optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)
    mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
        optimal_phi.detach().cpu(),
    )

    # motion-corrected T1 parameter fitting on updated image, using chosen initialization.
    p, qp = t1_parameter_estimator.fit(
        x_current,
        mask=mask_85_percent,
        initial_p=p_init,
    )
    torch.cuda.empty_cache()
    print(f'p^({iteration}) updated.')


if SAVE_RESULTS:
    torch.save(p, OUTPUT_DIR / 'p_moco.pt')
    print(f'Results saved to {OUTPUT_DIR}')

print('Finished processing.')
# %%
