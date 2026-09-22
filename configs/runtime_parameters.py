"""Centralized configuration for motion-corrected reconstruction scripts."""
import os
from datetime import datetime
import math
import torch

try:
    from configs.local_paths import EXPERIMENT_FILENAMES, PATHS
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "configs/local_paths.py not found: copy configs/local_paths_example.py "
        "to configs/local_paths.py and set your data paths."
    ) from e

today = datetime.now().strftime("%Y-%m-%d")


MOTION_NET_PARAMS = {
    'phi_episodes_start': 40, 
    'phi_steps_start': 20,
    'full_loss_phi_start': 35, 
    'phi_episodes_iter': 30, 
    'phi_steps_iter': 20,
    'full_loss_phi_iter': 25, 
}

RECONSTRUCTION_PARAMS = {
    'initial_expansion' : [0,1,-1,2],
    'lambda1': 0.1, 
    'lambda2': 0.05,
    'lambda3': 0.05 * 0.5, 
    'cg_iterations_initial': 8, 
    'cg_iterations_opt_pb': 5, 
    'cg_tolerance': 1e-5,  
    'fitting_opt_type':'lbfgs', # 'adam', 
    'divide_by_n': False,
    'lbfgs_tolerance_grad'  : 1e-8, 
    'lbfgs_tolerance_change': 1e-10, 
    'lbfgs_max_iterations': 300, 
    'lbfgs_max_evaluations': 200,
    't1_constraints': (0.1, 6.0), # in seconds
    'alpha_constraints': (math.radians(1), math.radians(9)),  # alpha is constrained
    'magnitude_model_op':False,
}


RUN_SETTINGS = {
    'run_visualization': True,
    'save_results': True,
    'print_logs': True,
    **PATHS,
}

LOG_SETTINGS = {
    'enabled': True,
    'log_dir': os.path.join(RUN_SETTINGS['output_dir'], 'logs'),
}

