"""Local data paths and experiment file names.

Copy this file to ``configs/local_paths.py`` (gitignored) and fill in your own values:

    cp configs/local_paths_example.py configs/local_paths.py
"""
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")


# Names of the ISMRMRD files (without extension) found in PATHS['data_root']
EXPERIMENT_FILENAMES = [
    'subject0_example_ismrmrd_traj_2s',
]

PATHS = {
    'mask_dir': '/path/to/masks/',
    'data_root': '/path/to/data',
    'note_output_dir': 'rerun/',
    'output_dir': f'/path/to/results/{today}/',
    # XCAT simulation (src/data/loading_xcat.py)
    # folder with sa_cine_100bpm.npz and ph_motion_xcat_100bpm_rt_cine_tvt_nr9_pe441_sl0-5_vol1.npz
    'xcat_data_dir': '/path/to/xcat_data/',
    # in-vivo scan in data_root whose trajectory and coil sensitivities are used for the simulation
    'xcat_trajectory_filename': 'subject0_example_ismrmrd_traj_2s',
    # XCAT run (scripts/main_x_cat.py, ResultsVisualizer.from_xcat): folder with mask_parameters_xcat_100bpm.json
    'data_root_xcat': '/path/to/xcat_run_data/',
    # Fibrosis patients (scripts/main_pat_fibrosis.py): <name>.h5 files and rr_idx/<prefix>_rr_idx.npy
    'data_root_pat_fibrosis': '/path/to/fibrosis_patients_data',
}
