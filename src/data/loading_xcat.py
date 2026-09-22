#%%
"""Simulated data from the XCAT phantom.

Ground-truth parameter maps (M0, flip angle, T1) and motion fields are loaded, turned into moving
dynamic images with the signal model, and projected to k-space using the trajectory and coil
sensitivities of a real volunteer scan (plus complex Gaussian noise).
"""
import torch
import numpy as np
from einops import rearrange
import mrpro
import os


from operators.MotionOperator import MotionOperator
from data.loading_data import get_prefix, load_patient_kdata_and_rr_idx, build_dynamic_kdata
from operators.construct_recon_operators import construct_reconstruction_operators


def get_motion_operator(data_folder, rr_idx, device):
    """Build a MotionOperator from the ground-truth XCAT motion fields.

    Args:
        data_folder: folder containing ``sa_cine_100bpm.npz``.
        rr_idx: cardiac phase index of each dynamic, used to pick the matching field.
        device: torch device.

    Returns:
        MotionOperator with forward (``mf``) and inverse (``imf``) fields, cropped to 240x240.
    """
    
    sa_cine = np.load(os.path.join(data_folder, 'sa_cine_100bpm.npz'), allow_pickle=True)
    num_dynamics = 54
    img_size = [240,240]
    
    mf_fw_true = torch.from_numpy(sa_cine['mf'])
    mf_inv_true = torch.from_numpy(sa_cine['imf'])

    mf_fw_true = mf_fw_true[120:-120,120:-120]
    mf_inv_true = mf_inv_true[120:-120,120:-120]

    mf_fw_true_reordered = torch.zeros(num_dynamics, 2, *img_size)
    mf_inv_true_reordered = torch.zeros(num_dynamics, 2, *img_size)

    for t in range(num_dynamics):
        idx = rr_idx[t]
        mf_fw_true_reordered[t] = rearrange(mf_fw_true[:,:,0,:-1,idx], 'h w ch -> ch h w')
        mf_inv_true_reordered[t] = rearrange(mf_inv_true[:,:,0,:-1,idx], 'h w ch -> ch h w')


    motion_op = MotionOperator(
                times=num_dynamics,
                img_size=img_size,
                device=device,
                phi = mf_fw_true_reordered,
                phi_inv = mf_inv_true_reordered
            )
    return motion_op

def get_p_true_values(data_folder, device='cpu'):
    """Load the ground-truth XCAT parameter maps.

    Returns:
        m0_true (complex), t1_s_true (T1 in seconds), flip_angle_deg_true (degrees), and
        nominal_fa: nominal flip angle in radians (value of a corner voxel).
    """
    sa_cine = np.load(os.path.join(data_folder, 'sa_cine_100bpm.npz'), allow_pickle=True)
    p_true_values = torch.from_numpy(sa_cine['qpar'][120:-120,120:-120,:]).float().to(device)
    p_true_values = rearrange(p_true_values, 'h w p -> p 1 1 1 h w')
    
    m0_true = p_true_values[0]
    m0_complex = m0_true + 0j
    m0_true = m0_complex
    t1_ms_true = p_true_values[3]
    t1_s_true = t1_ms_true / 1e3
    flip_angle_deg_true = p_true_values[1]
    
    nominal_fa = float(torch.deg2rad(flip_angle_deg_true.squeeze()[-1][-1]))
    return m0_true, t1_s_true, flip_angle_deg_true, nominal_fa

def correct_rot_flip(data):
    """Rotate by 90 degrees and flip so the phantom has the same orientation as the scanner data."""
    data_corrected = torch.rot90(data, k=1, dims=[-2,-1])
    data_corrected = torch.flip(data_corrected, dims=[-2])
    return data_corrected

def get_kdata(RECONSTRUCTION_PARAMS, RUN_SETTINGS, device):
    """Load a real volunteer scan to reuse its trajectory and coil sensitivities for simulation.

    Args:
        RECONSTRUCTION_PARAMS: config dict (uses the CG iterations/tolerance).
        RUN_SETTINGS: config dict (uses 'data_root' and 'xcat_trajectory_filename').
        device: torch device.

    Returns:
        kdata, kdata_dynamic, ref_max (max magnitude of the CG recon, used for scaling),
        acquisition_op, dcf_op, AHWA_op.
    """
    
    patient_name = RUN_SETTINGS['xcat_trajectory_filename']
    # Load patient k-space
    kdata, _, _ = load_patient_kdata_and_rr_idx(
        patient_name=patient_name,
        patient_prefix=get_prefix(patient_name),
        data_root=RUN_SETTINGS['data_root'],
        device=device,
    )
    
    # Build kdata for dynamic images reconstruction
    kdata_dynamic, _ = build_dynamic_kdata(kdata, device=device)

    dcf_op, _, acquisition_op, AHWA_op, rhs, _ = construct_reconstruction_operators(
        kdata_dynamic=kdata_dynamic,
        kdata=kdata,
        device=device,
    )
    
    with torch.no_grad():
            x_moving_vol, = mrpro.algorithms.optimizers.cg(
                AHWA_op,
                rhs,
                initial_value=rhs,
                max_iterations=RECONSTRUCTION_PARAMS['cg_iterations_initial'],
                tolerance=RECONSTRUCTION_PARAMS['cg_tolerance']
            )
            
    ref_max = x_moving_vol.abs().max().item()
        
    return kdata, kdata_dynamic, ref_max, acquisition_op, dcf_op, AHWA_op
        


def load_xcat_data(RECONSTRUCTION_PARAMS, RUN_SETTINGS, device):
    """Create the full simulated XCAT dataset.

    Args:
        RECONSTRUCTION_PARAMS: config dict.
        RUN_SETTINGS: config dict (uses 'xcat_data_dir', 'data_root' and 'xcat_trajectory_filename').
        device: torch device.

    Returns:
        kdata_vol: real volunteer KData (trajectory/header source).
        kdata_dynamic: dynamic KData whose data is replaced by the simulated, noisy k-space.
        rr_idx: cardiac phase index per dynamic.
        p_true: ground-truth [M0, flip angle (deg), T1 (s)] stacked along dim 0.
        sampling_time, repetition_time, tau: sequence timings for the signal model.
        nominal_fa: nominal flip angle in radians.
    """
    data_folder = RUN_SETTINGS['xcat_data_dir']

    motion_rt_cine = np.load(
        os.path.join(data_folder, 'ph_motion_xcat_100bpm_rt_cine_tvt_nr9_pe441_sl0-5_vol1.npz'),
        allow_pickle=True)

    data_pars = motion_rt_cine['pars']
    pars_dict = data_pars.item()
    rr_idx = pars_dict['rr_idx']
    rr_idx = torch.from_numpy(rr_idx)
    tau = torch.tensor(pars_dict['tau']).float()
    
    sampling_time = torch.from_numpy(pars_dict['ti']).float()
    sampling_time = sampling_time - sampling_time[0]
    repetition_time = torch.tensor(pars_dict['tr']).float()
    
    m0_true, t1_s_true, flip_angle_deg_true, nominal_fa = get_p_true_values(data_folder, device)

    motion_op = get_motion_operator(data_folder, rr_idx, device)


    model_op = mrpro.operators.models.TransientSteadyStateWithPreparation(
                            sampling_time=sampling_time,
                            repetition_time=repetition_time,
                            m0_scaling_preparation=-1.,
                            delay_after_preparation=tau
                        )
    model_op = model_op.to(device)


    x_true_fixed, = model_op(m0=m0_true, 
                t1=t1_s_true, 
                flip_angle=torch.deg2rad(flip_angle_deg_true))

    # x_true_fixed = x_true_fixed.abs()
    x_true_fixed = torch.nan_to_num(x_true_fixed, nan=0.0)
    x_true_fixed /= x_true_fixed.abs().max()  

    x_true_moving, = motion_op.inverse(x_true_fixed)

    kdata_vol, kdata_dynamic, ref_max, acquisition_op, dcf_op, AHWA_op = get_kdata(
        RECONSTRUCTION_PARAMS,
        RUN_SETTINGS,
        device)
    
    x_true_moving = correct_rot_flip(x_true_moving)
    x_true_moving = x_true_moving/x_true_moving.abs().max() * ref_max
    x_true_moving = x_true_moving.to(kdata_dynamic.data.dtype)
    
    m0_true = correct_rot_flip(m0_true)
    t1_s_true = correct_rot_flip(t1_s_true)
    flip_angle_deg_true = correct_rot_flip(flip_angle_deg_true)


    kdata_dynamic_simul, = acquisition_op(x_true_moving.to(device))
    
    sigma = 1e-3 * kdata_dynamic_simul.abs().max()
    noise = sigma * (torch.randn_like(kdata_dynamic_simul.real) 
                    + 
                    1j * torch.randn_like(kdata_dynamic_simul.real))

    kdata_dynamic_simul = kdata_dynamic_simul + noise
    kdata_dynamic.data = kdata_dynamic_simul
    
    p_true = torch.cat([m0_true, flip_angle_deg_true, t1_s_true], dim=0)
    
    return kdata_vol, kdata_dynamic, rr_idx, p_true, sampling_time, repetition_time, tau, nominal_fa

# %%
