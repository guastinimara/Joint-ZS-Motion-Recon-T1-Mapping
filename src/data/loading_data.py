"""Load in-vivo scans (ISMRMRD k-space + companion npz metadata) and split them into dynamics."""
import numpy as np
import mrpro
import torch

def get_prefix(input_string):
    """Return the subject prefix of a file name, e.g. 'pat6_meas_...' -> 'pat6'."""
    return input_string.split('_')[0]


def load_patient_kdata_and_rr_idx(
    patient_name,
    patient_prefix,
    data_root,
    device,
):
    """Load raw k-space data and RR indices for a subject.

    Args:
        patient_name: file name (without .h5) inside ``{data_root}/h5_files``.
        patient_prefix: subject prefix (e.g. 'pat6') used to find the companion npz file.
        data_root: root folder of the dataset.
        device: torch device for the returned data.

    Returns:
        kdata: mrpro KData with the ISMRMRD trajectory.
        rr_idx: tensor with the RR-interval (cardiac phase) index of each dynamic.
        tau: delay after the inversion pulse, from the npz parameters.
    """
    # Raw ISMRMRD trajectory k-space for the selected patient.
    kdata = mrpro.data.KData.from_file(
        f'{data_root}/h5_files/{patient_name}.h5',
        mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
    ).to(device)

    # Companion npz stores acquisition-derived metadata.
    data_rt = np.load(
        f'{data_root}/rt_cine_tvt_nr9_pe441_sl0-5_{patient_prefix}.npz',
        allow_pickle=True,
    )
    data_pars = data_rt['pars']
    pars_dict = data_pars.item()
    rr_idx = pars_dict['rr_idx']
    tau = pars_dict['tau']
    
    rr_idx = torch.from_numpy(rr_idx).to(device)
    
    return kdata, rr_idx, tau


def build_dynamic_kdata(
    kdata,
    device: str,
    n_lines_per_dynamic: int = 16,
    overlap: int = 8,
):
    """Sort readouts by acquisition time and split them into (overlapping) dynamics.

    Args:
        kdata: full KData of the scan.
        device: torch device.
        n_lines_per_dynamic: number of k-space lines in each dynamic image.
        overlap: number of lines shared by consecutive dynamics (sliding window).

    Returns:
        kdata_dynamic: KData with a leading dynamic dimension.
        num_dynamics: number of dynamic images.
    """
    sorted_idx = kdata.header.acq_info.acquisition_time_stamp.squeeze().argsort()
    split_idx = mrpro.utils.split_idx(sorted_idx, n_lines_per_dynamic, overlap)
    kdata_dynamic = kdata[..., split_idx, :]
    kdata_dynamic = kdata_dynamic.to(device)
    num_dynamics = kdata_dynamic.data.shape[0]
    return kdata_dynamic, num_dynamics


def dynamic_kdata_to_kdata(
    kdata_dynamic,
    overlap_mode: str = 'first',
):
    """Inverse of ``build_dynamic_kdata``: merge the dynamics back into a single KData.

    The dynamic dimension is flattened into k1 and duplicated lines (shared by consecutive
    dynamics because of the sliding-window overlap) are merged, using the acquisition time stamp
    to identify each line. Trajectory and header are taken from the dynamic KData itself.

    Note: lines that were not in any dynamic (the tail left out by ``split_idx``) cannot be
    recovered, so the result may have slightly fewer k1 lines than the original KData. Lines are
    returned sorted by acquisition time.

    Args:
        kdata_dynamic: KData with a leading dynamic dimension, as returned by ``build_dynamic_kdata``.
        overlap_mode: how to merge the data of a line that appears in more than one dynamic:
            'first' keeps the copy from the earliest dynamic, 'last' from the latest one and
            'mean' averages all copies.

    Returns:
        kdata: KData of shape (other, coils, k2, k1, k0) with one entry per acquired line.
    """
    if overlap_mode not in ('first', 'last', 'mean'):
        raise ValueError(f"overlap_mode must be 'first', 'last' or 'mean', got {overlap_mode!r}")

    # (dyn other coils k2 k1 k0) -> (other coils k2 (dyn k1) k0), dynamic-major so that the
    # flattened position increases with the dynamic index.
    kdata_flat = kdata_dynamic.rearrange('dyn other coils k2 k1 k0 -> other coils k2 (dyn k1) k0')

    time_stamp = kdata_flat.header.acq_info.acquisition_time_stamp
    if time_stamp.shape[:-2].numel() != 1:
        raise ValueError('Expected one acquisition time stamp per k1 line (other=k2=1).')
    time_stamp = time_stamp.flatten()

    # One entry per unique line, sorted by acquisition time; `inverse` maps each flattened
    # position to its unique line.
    _, inverse = torch.unique(time_stamp, sorted=True, return_inverse=True)
    num_lines = int(inverse.max()) + 1
    position = torch.arange(time_stamp.numel(), device=time_stamp.device)
    reduce = 'amax' if overlap_mode == 'last' else 'amin'
    line_idx = torch.full((num_lines,), -1 if reduce == 'amax' else time_stamp.numel(),
                          dtype=position.dtype, device=position.device)
    line_idx = line_idx.scatter_reduce(0, inverse, position, reduce=reduce)

    # Traj/header of duplicated lines are identical, so selecting one copy per line is enough.
    kdata = kdata_flat[..., line_idx, :]

    if overlap_mode == 'mean':
        data = kdata_flat.data
        summed = torch.zeros_like(kdata.data).index_add_(-2, inverse, data)
        counts = torch.bincount(inverse, minlength=num_lines).to(data.real.dtype)
        kdata.data = summed / counts[:, None]

    return kdata


def get_prefix_long_pats(input_string):
    """Return patient prefix with first three underscore-separated tokens."""
    return '_'.join(input_string.split('_')[:3])


