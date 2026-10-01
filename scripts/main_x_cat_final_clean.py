#%%
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'src')]


import torch
import numpy as np
import mrpro 
import os

from cardiac_motion.MotionEstimationWithInverse import MotionEstimationWithInverse

from operators.MotionOperator import MotionOperator
from operators.construct_recon_operators import compute_csm, construct_reconstruction_operators


from utils.MaskImgSize import MaskImgSize
from utils.rr_threshold_functions import find_valid_rr_threshold, update_rr_expansion_and_compute_rr_mask
from utils.save import save_patient_results

from regularization.svd import compute_svd, update_svd

from parameters_estimation.T1ParameterEstimator import T1ParameterEstimator


from configs.runtime_parameters import (
    RUN_SETTINGS,
    MOTION_NET_PARAMS,
    RECONSTRUCTION_PARAMS,
)


import tempfile
import zenodo_get


device = 'cuda' if torch.cuda.is_available() else 'cpu'

ZENODO_RECORD_ID = "23058873"

# KData.from_file needs a path, so the download goes to a temporary folder that is deleted after loading.
with tempfile.TemporaryDirectory() as tmp_dir:
    zenodo_get.download(
        record=ZENODO_RECORD_ID, output_dir=tmp_dir, file_glob=("kdata_simul.h5",)
    )
    kdata_simul = mrpro.data.KData.from_file(
        os.path.join(tmp_dir, "kdata_simul.h5"),
        mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
    ).to(device)


# Cardiac phase (RR-interval) index of each of the 54 dynamics.
rr_idx = torch.tensor([
    7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 0, 0, 1, 1, 2, 2, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9,
], dtype=torch.int64)

# Sampling time of each dynamic after the inversion pulse, relative to the first one (s).
sampling_time = torch.tensor([
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

repetition_time = torch.tensor(0.0049395603, dtype=torch.float32)  # s

tau = torch.tensor(0.028, dtype=torch.float32)  # delay after inversion pulse (s)
nominal_fa = 0.0872664600610733  # nominal flip angle (rad), i.e. 5 degrees


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
    

(rhs,) = (acquisition_op.H @ dcf_op)(kdata_simul_dynamic.data)

with torch.no_grad():
    x_moving, = mrpro.algorithms.optimizers.cg(
        AHWA_op,
        rhs,
        initial_value=rhs,
        max_iterations=RECONSTRUCTION_PARAMS['cg_iterations_initial'],
        tolerance=RECONSTRUCTION_PARAMS['cg_tolerance']
    )
    
num_dynamics = x_moving.shape[0]

torch.cuda.empty_cache()

t1_parameter_estimator = T1ParameterEstimator(
    sampling_time=sampling_time,
    repetition_time=repetition_time,
    nominal_flip_angle=nominal_fa,
    device=device,
    tau=tau,
    opt_type='lbfgs')
    

# Initial T1 parameter fitting with ECG-based mask for motion-compensated references.
p_0_ref, x_0_ref, rr_expansion_offsets = find_valid_rr_threshold(
    rr_idx, x_moving, t1_parameter_estimator, 
    initial_expansion=RECONSTRUCTION_PARAMS['initial_expansion'],
)

if p_0_ref is None or x_0_ref is None:
    raise RuntimeError("Could not find valid RR threshold.")

# Store rr_idx expansion pattern for iterative updates.
base_rr_idx = rr_idx[0].item()
current_rr_expansion = rr_expansion_offsets.copy()

mask_roi = MaskImgSize(h_start=70, w_start=68, h_width=100, w_width=100)

motion_op = MotionOperator(
    times=num_dynamics,
    img_size=img_size,
    device=device,
)


motion_estimation = MotionEstimationWithInverse(
        kdata_dynamic=kdata_simul_dynamic.to(device),
        acquisition_op=acquisition_op.to(device),
        dcf_sqrt_op=dcf_sqrt_op.to(device),
        motion_op=motion_op,
        img_size=img_size,
        times=num_dynamics,
        lr=MOTION_NET_PARAMS['lr_phi'],
        weight_decay=MOTION_NET_PARAMS['weight_phi'],
        optimizer=MOTION_NET_PARAMS['optimizer'],
        mask_roi=mask_roi,
        ch_out_unet=MOTION_NET_PARAMS['ch_out_unet_phi'],
        n_enc_stages=MOTION_NET_PARAMS['n_enc_stages_phi'],
        n_filters=MOTION_NET_PARAMS['n_filters_phi'],
        weight_dc=MOTION_NET_PARAMS['weight_dc'],
        weight_mi=MOTION_NET_PARAMS['weight_mi'],
        lambd=MOTION_NET_PARAMS['lambd'],
        mu=MOTION_NET_PARAMS['mu'],
        device=device,
    )

# Initial cardiac motion fields estimation.
motion_estimation.train_net(
    x_moving.abs(),
    x_0_ref,
    episodes=MOTION_NET_PARAMS['phi_episodes_start'],
    steps=MOTION_NET_PARAMS['phi_steps_start'],
    full_loss_phi=MOTION_NET_PARAMS['full_loss_phi_start']
)


torch.cuda.empty_cache()


# Compute current registered dynamics and update current x with it
x_registered, = motion_op.M_op_inv(x_moving)
x_registered = x_registered.detach()
x_current = x_registered


# Compute mask selection from sinusoidal fit to exclude 15% systolic frame
optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)
mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
    optimal_phi.detach().cpu(),
    percentage=RECONSTRUCTION_PARAMS['dyns_selection_percentage'],
    plot=False,
)
    
# Initial motion-corrected T1 parameter fitting
p, qp = t1_parameter_estimator.fit(x_current, mask=mask_85_percent, 
                            recon_params=RECONSTRUCTION_PARAMS)
   
# Initial SVD-based computation
x_reg_svd, U_op, _ = compute_svd(x_current, mask_roi, divide_by_n=RECONSTRUCTION_PARAMS['divide_by_n'])
x_reg_svd = x_reg_svd.detach().unsqueeze(1).unsqueeze(1)


if RUN_SETTINGS['save_intermediate_results']:
    output_dir = os.path.join(RUN_SETTINGS['output_dir'], RUN_SETTINGS['note_output_dir'], 'it_0')
    os.makedirs(output_dir, exist_ok=True)
    
    save_patient_results(
        output_dir=output_dir,
        motion_op=motion_op,
        p=p,
        x_current=x_current,
    )

# Alternate motion update, SVD regularization, and CG reconstruction, MoCo T1 estimation.
for iteration in range(1, int(RECONSTRUCTION_PARAMS['opt_iterations'])  ):
 
    # Update rr_idx expansion for this iteration
    # Compute mask with current expansion
    mask_ref, current_rr_expansion = update_rr_expansion_and_compute_rr_mask(rr_idx, base_rr_idx, current_rr_expansion)

    if not mask_ref.any():
        p_ref_updated = p_0_ref
        x_ref_updated = x_0_ref
    else:
        # Try fitting with current mask
        p_ref_updated, x_ref_updated = t1_parameter_estimator.fit(x_current, mask=mask_ref, recon_params=RECONSTRUCTION_PARAMS)
        
        # If NaN, fall back to previous values and don't expand further
        if torch.isnan(x_ref_updated).any():
            p_ref_updated = p_0_ref
            x_ref_updated = x_0_ref
                            
    # Update cardiac motion fields estimation
    
    motion_estimation.train_net(
            x_moving.abs(), 
            x_ref_updated.detach(), 
            episodes= MOTION_NET_PARAMS['phi_episodes_iter'], 
            steps=MOTION_NET_PARAMS['phi_steps_iter'], 
            full_loss_phi=MOTION_NET_PARAMS['full_loss_phi_iter'])
    
    optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)
    
    
    

    # Update SVD regularizer with soft-thresholding
   
    x_reg_svd, _ = update_svd(x_current, U_op,  
                            weight_l1=RECONSTRUCTION_PARAMS['lambda3'],
                            weight_l2=RECONSTRUCTION_PARAMS['lambda2'],
                            divide_by_n=RECONSTRUCTION_PARAMS['divide_by_n'])
   
    with torch.no_grad():
        # Motion-corrected image reconstruction with updated motion fields and SVD regularization.
        # Solve the quadratic subproblem with CG.
        H = lambda y: (motion_op.adjoint(AHWA_op(motion_op.forward(y)[0])[0])[0] 
                        + 
                        RECONSTRUCTION_PARAMS["lambda1"] * y 
                        + 
                        RECONSTRUCTION_PARAMS["lambda2"] * y,)
        
        b_0 = motion_op.adjoint(acquisition_op.adjoint(dcf_op(kdata_simul_dynamic.data)[0])[0])[0]
        b = b_0 + RECONSTRUCTION_PARAMS["lambda1"] * qp + RECONSTRUCTION_PARAMS["lambda2"] * x_reg_svd
        # qp or x ref updated?
        
        x_next, = mrpro.algorithms.optimizers.cg(
                        operator = H, 
                        right_hand_side=b, 
                        initial_value=x_current,
                        max_iterations=RECONSTRUCTION_PARAMS["cg_iterations_opt_pb"], 
                        tolerance=RECONSTRUCTION_PARAMS["cg_tolerance"]
        ) 

       
    torch.cuda.empty_cache()
    x_current = x_next.detach()
    
    # Initialize the parametric maps for further fitting
    p_init = p_ref_updated.detach()
    
    # automatic extraction of systolic frames
    mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
        optimal_phi.detach().cpu(),
        percentage=RECONSTRUCTION_PARAMS['dyns_selection_percentage'],
        plot=False,
    )
 
    # Motion-corrected T1 parameter fitting on updated image, using chosen initialization.
    
    p, qp = t1_parameter_estimator.fit(x_current, mask=mask_85_percent, 
                                initial_p = p_init,
                                recon_params=RECONSTRUCTION_PARAMS)
    
        
    torch.cuda.empty_cache()
        
    
    if RUN_SETTINGS['save_intermediate_results']:
        output_dir = os.path.join(RUN_SETTINGS['output_dir'],RUN_SETTINGS['note_output_dir'], f'it_{iteration}')
        os.makedirs(output_dir, exist_ok=True)
        save_patient_results(
            output_dir=output_dir,
            iteration=iteration,
            motion_op=motion_op,
            p=p,
            x_current=x_current,
        )

if RUN_SETTINGS['save_results']:
    output_dir = os.path.join(RUN_SETTINGS['output_dir'],RUN_SETTINGS['note_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    save_patient_results(
        output_dir=output_dir,
        motion_op=motion_op,
        p=p,
        x_current=x_current,
    )


# %%
