#%%
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'src')]

import torch
import mrpro 
import os

from cardiac_motion.MotionEstimationWithInverse import MotionEstimationWithInverse

from operators.MotionOperator import MotionOperator
from operators.construct_recon_operators import construct_reconstruction_operators

from data.loading_xcat import load_xcat_data

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

patient_prefix = 'xcat_100bpm'
output_dir = os.path.join(RUN_SETTINGS['output_dir'], RUN_SETTINGS['note_output_dir'], patient_prefix)
os.makedirs(output_dir, exist_ok=True)

print(f'Working on {patient_prefix} | device={device}')

# Load patient k-space and RR metadata.
(kdata_vol,
    kdata_dynamic_simul, 
    rr_idx, 
    p_true,  
    sampling_time, repetition_time, tau, nominal_fa) = load_xcat_data(
        RECONSTRUCTION_PARAMS,
        RUN_SETTINGS,
        device)

dcf_op, dcf_sqrt_op, acquisition_op, AHWA_op, rhs, img_size = construct_reconstruction_operators(
    kdata_dynamic= kdata_dynamic_simul,
    kdata=kdata_vol,
    device=device,
)

(rhs,) = (acquisition_op.H @ dcf_op)(kdata_dynamic_simul.data)

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

# t1_parameter_estimator = T1ParameterEstimator(kdata=kdata_dynamic, opt_type=RECONSTRUCTION_PARAMS['fitting_opt_type'])

t1_parameter_estimator = T1ParameterEstimator(
    sampling_time=sampling_time,
    repetition_time=repetition_time,
    nominal_flip_angle=nominal_fa,
    device=device)

# Initial T1 parameter fitting with ECG-based mask for motion-compensated references.
p_0_ref, x_0_ref, rr_expansion_offsets = find_valid_rr_threshold(
    rr_idx, x_moving, t1_parameter_estimator,
)

if p_0_ref is None or x_0_ref is None:
    raise RuntimeError("Could not find valid RR threshold.")

# Store rr_idx expansion pattern for iterative updates.
base_rr_idx = rr_idx[0].item()
current_rr_expansion = rr_expansion_offsets.copy()

mask = MaskImgSize()
mask_file = f'{RUN_SETTINGS["data_root_xcat"]}/mask_parameters_{patient_prefix}.json'
mask.load_params(mask_file)
# print(f'Mask loaded from {mask_file}')

motion_op = MotionOperator(
    times=num_dynamics,
    img_size=img_size,
    device=device,
)

if RUN_SETTINGS['moco']:
    motion_estimation = MotionEstimationWithInverse(
            kdata_dynamic=kdata_dynamic_simul.to(device),
            acquisition_op=acquisition_op.to(device),
            dcf_sqrt_op=dcf_sqrt_op.to(device),
            motion_op=motion_op,
            img_size=img_size,
            times=num_dynamics,
            mask_roi=mask,
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

print('x_registered shape:', x_registered.shape)
print('x current shape:', x_current.shape)

# Compute mask selection from sinusoidal fit to exclude 15% systolic frame
optimal_phi = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid)
mask_85_percent = T1ParameterEstimator.select_sinusoidal_mask(
    optimal_phi.detach().cpu()
)
if not RUN_SETTINGS['moco']:
    mask_85_percent = None
    
# Initial motion-corrected T1 parameter fitting
p, qp = t1_parameter_estimator.fit(x_current, mask=mask_85_percent, 
                            recon_params=RECONSTRUCTION_PARAMS)

# Initial SVD-based computation
x_reg_svd, U_op, _ = compute_svd(x_current, mask)
x_reg_svd = x_reg_svd.detach().unsqueeze(1).unsqueeze(1)

if RUN_SETTINGS['save_intermediate_results']:
    output_dir = os.path.join(RUN_SETTINGS['output_dir'], RUN_SETTINGS['note_output_dir'], patient_prefix, 'it_0')
    os.makedirs(output_dir, exist_ok=True)
    save_patient_results(
        output_dir=output_dir,
        motion_op=motion_op,
        p=p,
        x_current=x_current,
    )

# Alternate motion update, SVD regularization, and CG reconstruction, MoCo T1 estimation.
for iteration in range(1, 5):
    print(f'Iteration {iteration}/4')
    
    if RUN_SETTINGS['motion_fields_in_iter']:
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

    else:
        p_ref_updated = p_0_ref
        x_ref_updated = x_0_ref

    # Update SVD regularizer with soft-thresholding

    x_reg_svd, _ = update_svd(x_current, U_op,  
                            weight_l1=RECONSTRUCTION_PARAMS['lambda3'],
                            weight_l2=RECONSTRUCTION_PARAMS['lambda2'])

    with torch.no_grad():
        # Motion-corrected image reconstruction with updated motion fields and SVD regularization.
        # Solve the quadratic subproblem with CG.
        H = lambda y: (motion_op.adjoint(AHWA_op(motion_op.forward(y)[0])[0])[0] 
                        + 
                        RECONSTRUCTION_PARAMS["lambda1"] * y 
                        + 
                        RECONSTRUCTION_PARAMS["lambda2"] * y,)
        
        b_0 = motion_op.adjoint(acquisition_op.adjoint(dcf_op(kdata_dynamic_simul.data)[0])[0])[0]
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
        optimal_phi.detach().cpu()
    )
    if not RUN_SETTINGS['moco']:
        mask_85_percent = None

    # Motion-corrected T1 parameter fitting on updated image, using chosen initialization.
    p, qp = t1_parameter_estimator.fit(x_current, mask=mask_85_percent, 
                                    initial_p = p_init,
                                    recon_params=RECONSTRUCTION_PARAMS)

    torch.cuda.empty_cache()

    if RUN_SETTINGS['save_intermediate_results']:
        output_dir = os.path.join(RUN_SETTINGS['output_dir'], RUN_SETTINGS['note_output_dir'],patient_prefix, f'it_{iteration}')
        os.makedirs(output_dir, exist_ok=True)
        save_patient_results(
            output_dir=output_dir,
            motion_op=motion_op,
            p=p,
            x_current=x_current,
        )

if RUN_SETTINGS['save_results']:
    output_dir = os.path.join(RUN_SETTINGS['output_dir'], RUN_SETTINGS['note_output_dir'], patient_prefix)
    os.makedirs(output_dir, exist_ok=True)
    save_patient_results(
        output_dir=output_dir,
        motion_op=motion_op,
        p=p,
        x_current=x_current,
    )

print(f"Finished processing {patient_prefix}")

# %%
