# %%
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

import os
import tempfile

import torch
import mrpro
import zenodo_get

from cardiac_motion.MotionEstimationWithInverse import MotionEstimationWithInverse

from operators.MotionOperator import MotionOperator
from operators.construct_recon_operators import compute_csm

from utils.MaskImgSize import MaskImgSize
from utils.rr_threshold_functions import (
    find_valid_rr_threshold,
    update_rr_expansion_and_compute_rr_mask,
)
from utils.save import save_patient_results

from regularization.svd import compute_svd, update_svd

from parameters_estimation.T1ParameterEstimator import T1ParameterEstimator

from configs.runtime_parameters import (
    RUN_SETTINGS,
    RECONSTRUCTION_PARAMS,
)


from data.loading_xcat import load_xcat_data
from operators.construct_recon_operators import construct_reconstruction_operators


device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the simulated XCAT k-space (kdata_simul.h5) from the public Zenodo record.
ZENODO_RECORD_ID = "22894393"

# KData.from_file needs a path, so the download goes to a temporary folder that is deleted after loading.
with tempfile.TemporaryDirectory() as tmp_dir:
    zenodo_get.download(
        record=ZENODO_RECORD_ID, output_dir=tmp_dir, file_glob=("kdata_simul.h5",)
    )
    kdata_simul = mrpro.data.KData.from_file(
        os.path.join(tmp_dir, "kdata_simul.h5"),
        mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
    ).to(device)
print("kdata_simul shape:", kdata_simul.data.shape)


# %%
# split kspace data into dynamics
sorted_idx = kdata_simul.header.acq_info.acquisition_time_stamp.squeeze().argsort()
split_idx = mrpro.utils.split_idx(sorted_idx, 16, 8)
kdata_simul_dynamic = kdata_simul[..., split_idx, :]

# compute acquisition operator for reconstruction
csm_op, img_size = compute_csm(kdata_simul, smoothing_width=15, device=device)

dcf = mrpro.data.DcfData.from_traj_voronoi(kdata_simul_dynamic.traj).to(device)
dcf_sqrt_op = mrpro.operators.DensityCompensationOp(torch.sqrt(dcf.data))
dcf_op = dcf.as_operator()
fourier_op = mrpro.operators.FourierOp.from_kdata(kdata_simul_dynamic).to(device)

acquisition_op = fourier_op @ csm_op
AHWA_op = (dcf_sqrt_op @ acquisition_op).gram

# (kdata_vol,
#         kdata_simul_dynamic, 
#         rr_idx, 
#         p_true,  
#         sampling_time, repetition_time, tau, nominal_fa) = load_xcat_data(
#             RECONSTRUCTION_PARAMS,
#             RUN_SETTINGS,
#             device)


# dcf_op, dcf_sqrt_op, acquisition_op, AHWA_op, rhs, img_size = construct_reconstruction_operators(
#     kdata_dynamic= kdata_simul_dynamic,
#     kdata=kdata_vol,
#     device=device,
# )


(rhs,) = (acquisition_op.H @ dcf_op)(kdata_simul_dynamic.data)

with torch.no_grad():
    (x_moving,) = mrpro.algorithms.optimizers.cg(
        AHWA_op,
        rhs,
        initial_value=rhs,
        max_iterations=RECONSTRUCTION_PARAMS["cg_iter_initial"],
        tolerance=RECONSTRUCTION_PARAMS["cg_tol"],
    )


#%%
# XCAT acquisition parameters.
# Cardiac phase (RR-interval) index of each of the 54 dynamics.

rr_idx = torch.tensor([
    7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 0, 0, 1, 1, 2, 2, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9,
], dtype=torch.int64)

# Sampling time of each dynamic after the inversion pulse, relative to the first one (s).
sampling_time = torch.tensor([0.0000, 0.0391, 0.0781, 0.1211, 0.1602, 0.1992, 0.2383, 0.2773, 0.3164,
        0.3555, 0.3984, 0.4375, 0.4727, 0.5156, 0.5547, 0.5938, 0.6328, 0.6719,
        0.7109, 0.7500, 0.7930, 0.8320, 0.8711, 0.9102, 0.9492, 0.9883, 1.0273,
        1.0664, 1.1055, 1.1484, 1.1875, 1.2266, 1.2656, 1.3047, 1.3438, 1.3828,
        1.4258, 1.4648, 1.5000, 1.5430, 1.5820, 1.6211, 1.6602, 1.6992, 1.7383,
        1.7812, 1.8203, 1.8594, 1.8984, 1.9375, 1.9766, 2.0156, 2.0547, 2.0938])


repetition_time = torch.tensor(0.0049)  # s

nominal_fa = 0.0872664600610733  # nominal flip angle (rad), i.e. 5 degrees

num_dynamics = x_moving.shape[0]
torch.cuda.empty_cache()

#%%

t1_parameter_estimator = T1ParameterEstimator(
    sampling_time=sampling_time,
    repetition_time=repetition_time,
    nominal_flip_angle=nominal_fa,
    tau=0.0280,
    device=device,
)

# Initial T1 parameter fitting with ECG-based mask for motion-compensated references.
p_0_ref, x_0_ref, rr_expansion_offsets = find_valid_rr_threshold(
    rr_idx,
    x_moving,
    t1_parameter_estimator,
)

if p_0_ref is None or x_0_ref is None:
    raise RuntimeError("Could not find valid RR threshold.")

# Store rr_idx expansion pattern for iterative updates.
base_rr_idx = rr_idx[0].item()
current_rr_expansion = rr_expansion_offsets.copy()

# Image region around the heart used by the motion loss and the SVD regularizer.
mask = MaskImgSize(h_start=70, w_start=68, h_width=100, w_width=100)

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
    mask_roi=mask,
    device=device,
)

# Initial cardiac motion fields estimation.
motion_estimation.train_net(
    x_moving.abs(), x_0_ref, episodes=40, steps=20, full_loss_phi=35
)
torch.cuda.empty_cache()

# Compute current registered dynamics and update current x with it
(x_registered,) = motion_op.M_op_inv(x_moving)
x_registered = x_registered.detach()
x_current = x_registered

# Compute mask selection from sinusoidal fit to exclude 15% systolic frame
optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)
mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
    optimal_phi.detach().cpu()
)

# Initial motion-corrected T1 parameter fitting
p, qp = t1_parameter_estimator.fit(
    x_current, mask=mask_85_percent, recon_params=RECONSTRUCTION_PARAMS
)

# Initial SVD-based computation
x_reg_svd, U_op, _ = compute_svd(x_current, mask)
x_reg_svd = x_reg_svd.detach().unsqueeze(1).unsqueeze(1)
print("Initialization completed.\n")


print("Starting iterative loop.\n")
# Alternate motion update, SVD regularization, and CG reconstruction, MoCo T1 estimation.
for iteration in range(1, 5):
    print(f"Iteration {iteration}/4\n")

    # Update rr_idx expansion for this iteration
    # Compute mask with current expansion
    mask_ref, current_rr_expansion = update_rr_expansion_and_compute_rr_mask(
        rr_idx, base_rr_idx, current_rr_expansion
    )

    if not mask_ref.any():
        p_ref_updated = p_0_ref
        x_ref_updated = x_0_ref
    else:
        # Try fitting with current mask
        p_ref_updated, x_ref_updated = t1_parameter_estimator.fit(
            x_current, mask=mask_ref, recon_params=RECONSTRUCTION_PARAMS
        )

        if torch.isnan(x_ref_updated).any():
            p_ref_updated = p_0_ref
            x_ref_updated = x_0_ref

    # Update cardiac motion fields estimation

    motion_estimation.train_net(
        x_moving.abs(), x_ref_updated.detach(), episodes=30, steps=20, full_loss_phi=25
    )
    print(f"phi^({iteration}) updated.\n")
    optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)

    # Update SVD regularizer with soft-thresholding

    x_reg_svd, _ = update_svd(
        x_current,
        U_op,
        weight_l1=RECONSTRUCTION_PARAMS["lambda3"],
        weight_l2=RECONSTRUCTION_PARAMS["lambda2"],
    )
    print(f"c_j^({iteration}) updated.\n")
    with torch.no_grad():
        # Motion-corrected image reconstruction with updated motion fields and SVD regularization.
        # Solve the quadratic subproblem with CG.
        H = lambda y: (
            motion_op.adjoint(AHWA_op(motion_op.forward(y)[0])[0])[0]
            + RECONSTRUCTION_PARAMS["lambda1"] * y
            + RECONSTRUCTION_PARAMS["lambda2"] * y,
        )

        b_0 = motion_op.adjoint(
            acquisition_op.adjoint(dcf_op(kdata_simul_dynamic.data)[0])[0]
        )[0]
        b = (
            b_0
            + RECONSTRUCTION_PARAMS["lambda1"] * qp
            + RECONSTRUCTION_PARAMS["lambda2"] * x_reg_svd
        )

        (x_next,) = mrpro.algorithms.optimizers.cg(
            operator=H,
            right_hand_side=b,
            initial_value=x_current,
            max_iterations=RECONSTRUCTION_PARAMS["cg_iter_opt_pb"],
            tolerance=RECONSTRUCTION_PARAMS["cg_tol"],
        )
    print(f"x^({iteration}) updated.\n")
    torch.cuda.empty_cache()
    x_current = x_next.detach()

    # Initialize the parametric maps for further fitting
    p_init = p_ref_updated.detach()

    # automatic extraction of systolic frames
    mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
        optimal_phi.detach().cpu()
    )

    # Motion-corrected T1 parameter fitting on updated image, using chosen initialization.
    p, qp = t1_parameter_estimator.fit(
        x_current,
        mask=mask_85_percent,
        initial_p=p_init,
        recon_params=RECONSTRUCTION_PARAMS,
    )
    print(f"p^({iteration}) updated.\n")
    torch.cuda.empty_cache()


if RUN_SETTINGS["save_results"]:
    output_dir = os.path.join(RUN_SETTINGS["output_dir"], "simulated_data")
    os.makedirs(output_dir, exist_ok=True)
    save_patient_results(
        output_dir=output_dir,
        motion_op=motion_op,
        p=p,
        x_current=x_current,
    )

print("Finished processing.")

#%%

from dynamic_mr_visualizer import ParametricMaps

maps = ParametricMaps(
    p,
    ['M0', 'alpha', 'T1'], 
    t1_unit='s'
)

maps.plot_t1()
# %%
