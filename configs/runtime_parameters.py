"""Centralized configuration for motion-corrected reconstruction scripts."""
import os
from datetime import datetime
import math
import torch

try:
    from configs.local_paths import PATHS
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "configs/local_paths.py not found: copy configs/local_paths_example.py "
        "to configs/local_paths.py and set your output paths."
    ) from e

today = datetime.now().strftime("%Y-%m-%d")


RECONSTRUCTION_PARAMS = {
    'initial_expansion' : [0,1,-1,2],
    'lambda1': 0.1, 
    'lambda2': 0.05,
    'lambda3': 0.05 * 0.5, 
    'cg_iter_initial': 8, 
    'cg_iter_opt_pb': 5, 
    'cg_tol': 1e-5,  
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
    'save_results': False,
    'print_logs': False,
    **PATHS,
}

LOG_SETTINGS = {
    'enabled': True,
    'log_dir': os.path.join(RUN_SETTINGS['output_dir'], 'logs'),
}

