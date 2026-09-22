"""Save and load per-subject results (images, parameter maps, motion fields)."""
import torch
from operators.MotionOperator import MotionOperator
import os
import re
from typing import cast
# from runtime_parameters import RUN_SETTINGS


def save_patient_results(output_dir, 
                         motion_op=None, 
                         p=None, x_current=None):
    """Save final outputs for one patient as .pt files in output_dir.

    Args:
        output_dir: destination folder (must exist).
        motion_op: MotionOperator; saved as forward_motion_field.pt / inverse_motion_field.pt.
        p: parameter maps; saved as final_p.pt.
        x_current: images; saved as final_x.pt.
    """
    
    if motion_op is not None:
        forward_field = MotionOperator.normalized_grid_to_field(motion_op.M_op.grid.detach())
        inverse_field = MotionOperator.normalized_grid_to_field(motion_op.M_op_inv.grid.detach())

        torch.save(forward_field.cpu(), os.path.join(output_dir, 'forward_motion_field.pt'))
        torch.save(inverse_field.cpu(), os.path.join(output_dir, 'inverse_motion_field.pt'))
    
    if p is not None:
        torch.save(p.detach().cpu(), os.path.join(output_dir, 'final_p.pt'))
    
    if x_current is not None:
        torch.save(x_current.detach().cpu(), os.path.join(output_dir, 'final_x.pt'))


def load_patient_iteration_results(
    patient_prefix,
    RUN_SETTINGS,
    map_location='cpu',
    nomoco=False
):
    """Load and stack saved tensors for one patient.

    Reads ``{output_dir}/{note_dir}/<patient_prefix>/it_*/`` in iteration order (or the files in
    the patient folder itself if there are no iteration folders) and returns:
        ``all_x``, ``all_p``, ``all_forward_motion``, ``all_inverse_motion``.
    """
    if RUN_SETTINGS is None:
        raise ValueError("RUN_SETTINGS must be provided.")
    output_root = cast(str, RUN_SETTINGS['output_dir'])
    note_dir = cast(str, RUN_SETTINGS['note_dir'])
    patient_output_dir = os.path.join(output_root, note_dir, patient_prefix)
    if not os.path.isdir(patient_output_dir):
        raise FileNotFoundError(f'Patient output directory does not exist: {patient_output_dir}')

    pattern = re.compile(r'^it_(\d+)$')
    iteration_ids = []
    for entry in os.listdir(patient_output_dir):
        match = pattern.match(entry)
        if match is not None:
            iteration_ids.append(int(match.group(1)))
    iteration_ids.sort()

    if not iteration_ids:
        final_x_path = os.path.join(patient_output_dir, 'final_x.pt')
        final_p_path = os.path.join(patient_output_dir, 'final_p.pt')
        forward_motion_path = os.path.join(patient_output_dir, 'forward_motion_field.pt')
        inverse_motion_path = os.path.join(patient_output_dir, 'inverse_motion_field.pt')

        missing_files = [
            path
            for path in (final_x_path, final_p_path, forward_motion_path, inverse_motion_path)
            if not os.path.isfile(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                'No iteration folders found and missing root-level result files in: '
                f'{patient_output_dir}. Missing: {missing_files}'
            )

        print(
            'No iteration folders found in '
            f'{patient_output_dir}. Falling back to root-level result files '
            '(single motion field / single pseudo-iteration).'
        )

        root_x = torch.load(final_x_path, map_location=map_location)
        root_p = torch.load(final_p_path, map_location=map_location)
        root_forward_motion = torch.load(forward_motion_path, map_location=map_location)
        root_inverse_motion = torch.load(inverse_motion_path, map_location=map_location)

        return (
            torch.stack([root_x], dim=0),
            torch.stack([root_p], dim=0),
            torch.stack([root_forward_motion], dim=0),
            torch.stack([root_inverse_motion], dim=0),
        )

    all_x = []
    all_p = []
    all_forward_motion = []
    all_inverse_motion = []

    for iteration_id in iteration_ids:
        iteration_dir = os.path.join(patient_output_dir, f'it_{iteration_id}')
        final_x_path = os.path.join(iteration_dir, 'final_x.pt')
        final_p_path = os.path.join(iteration_dir, 'final_p.pt')
        forward_motion_path = os.path.join(iteration_dir, 'forward_motion_field.pt')
        inverse_motion_path = os.path.join(iteration_dir, 'inverse_motion_field.pt')

        all_x.append(torch.load(final_x_path, map_location=map_location))
        all_p.append(torch.load(final_p_path, map_location=map_location))
        all_forward_motion.append(torch.load(forward_motion_path, map_location=map_location))
        all_inverse_motion.append(torch.load(inverse_motion_path, map_location=map_location))

    return (
        torch.stack(all_x, dim=0),
        torch.stack(all_p, dim=0),
        torch.stack(all_forward_motion, dim=0),
        torch.stack(all_inverse_motion, dim=0),
    )

