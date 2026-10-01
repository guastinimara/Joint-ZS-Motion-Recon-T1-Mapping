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

REPO_ROOT = Path(__file__).resolve().parents[2]
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
ZENODO_RECORD_ID = '23058873'
N_LINES_PER_DYNAMIC = 16  # k-space lines per dynamic image
N_LINES_OVERLAP = 8  # lines shared by consecutive dynamics (sliding window)
CSM_SMOOTHING_WIDTH = 15

# acquisition parameters of this dataset
# cardiac phase (RR-interval) index of each of the 54 dynamics.
RR_IDX = torch.tensor([
    7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 0, 0, 1, 1, 2, 2, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9,
], dtype=torch.int64)
# sampling time of each dynamic after the inversion pulse, relative to the first one (s).
SAMPLING_TIME = torch.tensor([
    0.0, 0.0390625, 0.078125, 0.12109375, 0.16015625, 0.19921875,
    0.23828125, 0.27734375, 0.31640625, 0.35546875, 0.3984375, 0.4375,
    0.47265625, 0.515625, 0.5546875, 0.59375, 0.6328125, 0.671875,
    0.7109375, 0.75, 0.79296875, 0.83203125, 0.87109375, 0.91015625,
    0.94921875, 0.98828125, 1.02734375, 1.06640625, 1.10546875, 1.1484375,
    1.1875, 1.2265625, 1.265625, 1.3046875, 1.34375, 1.3828125,
    1.42578125, 1.46484375, 1.5, 1.54296875, 1.58203125, 1.62109375,
    1.66015625, 1.69921875, 1.73828125, 1.78125, 1.8203125, 1.859375,
    1.8984375, 1.9375, 1.9765625, 2.015625, 2.0546875, 2.09375,
], dtype=torch.float32)
REPETITION_TIME = torch.tensor(0.0049395603, dtype=torch.float32)  # s
TAU = torch.tensor(0.028, dtype=torch.float32)  # delay after inversion pulse (s)
NOMINAL_FA = 0.0872664600610733  # nominal flip angle (rad), i.e. 5 degrees

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

# KData.from_file needs a path, so the download goes to a temporary folder that is deleted after loading.
with tempfile.TemporaryDirectory() as tmp_dir:
    zenodo_get.download(
        record=ZENODO_RECORD_ID,
        output_dir=tmp_dir,
        file_glob=('kdata_simul.h5',),
    )
    kdata_simul = mrpro.data.KData.from_file(
        os.path.join(tmp_dir, 'kdata_simul.h5'),
        mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
    ).to(device)


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


t1_parameter_estimator = T1ParameterEstimator(
    sampling_time=SAMPLING_TIME,
    repetition_time=REPETITION_TIME,
    nominal_flip_angle=NOMINAL_FA,
    device=device,
    tau=TAU,
    opt_type='lbfgs',
)

# initial T1 parameter fitting with ECG-based mask for motion-compensated references.
p_0_ref, x_0_ref, rr_expansion_offsets = find_valid_rr_threshold(
    RR_IDX,
    x_moving,
    t1_parameter_estimator,
)
if p_0_ref is None or x_0_ref is None:
    raise RuntimeError('Could not find valid RR threshold.')

# store rr_idx expansion pattern for iterative updates.
base_rr_idx = RR_IDX[0].item()
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
        RR_IDX,
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
