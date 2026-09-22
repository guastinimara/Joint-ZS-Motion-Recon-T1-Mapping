"""Figures of the saved results of a run (per iteration), for a patient/volunteer or the XCAT phantom."""
import os
from datetime import datetime

import mrpro
import numpy as np
import torch
from einops import rearrange

from data.loading_data import get_prefix, get_prefix_long_pats, load_patient_kdata_and_rr_idx, build_dynamic_kdata
from data.loading_xcat import load_xcat_data
from operators.MotionOperator import MotionOperator
from operators.construct_recon_operators import construct_reconstruction_operators
from parameters_estimation.T1ParameterEstimator import T1ParameterEstimator
from utils.MaskImgSize import MaskImgSize
from utils.rr_threshold_functions import find_valid_rr_threshold
from utils.save import load_patient_iteration_results
from visualization.DynamicViewer import DynamicViewer
from visualization.MotionFieldAnalyzer import MotionFieldAnalyzer
from visualization import iterations

TIMES_TO_VISUALIZE = [0,2,3,4,6,7,8,10,12,13,15,18,20,25,28,30,35,40,43]
SLICES_TO_VISUALIZE = [5,10,20,22,25,35,40,43,46,50,55,60,65,70,75,80,85,90]

XCAT_PREFIX = 'xcat_100bpm'

# Patients with long prefixes (fibrosis data) have no tau stored with the data;
# this is the value used for them in scripts/main_pat_fibrosis.py.
LONG_PREFIX_TAU = 0.028


class ResultsVisualizer:
    """Figures for the saved iteration results of one dataset.

    Build it with `from_patient` or `from_xcat`, which reconstruct x_moving and the initial
    reference, then call `run()` to write every figure to
    `{output_dir}/{note_dir}/{patient_prefix}/figures_{today}/`.
    """

    def __init__(self, patient_prefix, x_moving, x_0_ref, t1_parameter_estimator, mask, RUN_SETTINGS,
                 clim=[0,2000], times_to_visualize=TIMES_TO_VISUALIZE, slices_to_visualize=SLICES_TO_VISUALIZE,
                 plot_fields=False, plot_slices=True, plot_registrations=True, device='cpu'):
        """
        Args:
            patient_prefix: subject prefix, used to find the results and name the figure folder.
            x_moving: initial CG reconstruction of the dynamics.
            x_0_ref: initial reference from the RR-selected T1 fit.
            t1_parameter_estimator: T1ParameterEstimator (used for the correlation heatmaps).
            mask: MaskImgSize crop applied to all images.
            RUN_SETTINGS: config dict with output_dir and note_dir.
            clim: T1 colour range in ms.
            times_to_visualize, slices_to_visualize: time points and row/column indices to plot.
            plot_fields, plot_slices, plot_registrations: which optional figures to create.
            device: torch device.
        """
        self.patient_prefix = patient_prefix
        self.x_moving = x_moving
        self.x_0_ref = x_0_ref
        self.t1_parameter_estimator = t1_parameter_estimator
        self.mask = mask
        self.clim = clim
        self.times_to_visualize = times_to_visualize
        self.slices_to_visualize = slices_to_visualize
        self.plot_fields = plot_fields
        self.plot_slices = plot_slices
        self.plot_registrations = plot_registrations
        self.device = device

        today = datetime.now().strftime("%Y-%m-%d")
        self.figure_dir = f'{RUN_SETTINGS["output_dir"]}/{RUN_SETTINGS["note_dir"]}/{patient_prefix}/figures_{today}/'
        os.makedirs(self.figure_dir, exist_ok=True)

        (self.all_x,
         self.all_p,
         self.all_forward_motion,
         self.all_inverse_motion) = load_patient_iteration_results(patient_prefix, RUN_SETTINGS)

        print('Loaded results for patient:', patient_prefix)
        print(
            f"{patient_prefix}: "
            f"all_x {tuple(self.all_x.shape)}, "
            f"all_p {tuple(self.all_p.shape)}, "
            f"all_forward_motion {tuple(self.all_forward_motion.shape)}, "
            f"all_inverse_motion {tuple(self.all_inverse_motion.shape)}"
        )

        self.x_registered_all_iter = None

    # Construction

    @classmethod
    def from_patient(cls, patient_name, RUN_SETTINGS, RECONSTRUCTION_PARAMS, long_prefix=False, **kwargs):
        """Load a patient/volunteer and reconstruct x_moving.

        Returns None when no valid RR threshold is found for the initial reference.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if long_prefix:
            patient_prefix = get_prefix_long_pats(patient_name)
            kdata = mrpro.data.KData.from_file(
                f'{RUN_SETTINGS["data_root_pat_fibrosis"]}/{patient_name}.h5',
                mrpro.data.traj_calculators.KTrajectoryIsmrmrd(),
            ).to(device)

            kdata.traj.ky = kdata.traj.ky / 240. * 120.
            kdata.traj.kx = kdata.traj.kx / 240. * 120.

            rr_idx = torch.from_numpy(np.load(
                f'{RUN_SETTINGS["data_root_pat_fibrosis"]}/rr_idx/{patient_prefix}_rr_idx.npy')).to(device)
            tau = LONG_PREFIX_TAU
        else:
            patient_prefix = get_prefix(patient_name)
            kdata, rr_idx, tau = load_patient_kdata_and_rr_idx(
                patient_name=patient_name,
                patient_prefix=patient_prefix,
                data_root=RUN_SETTINGS['data_root'],
                device=device,
            )

        # Build kdata for dynamic images reconstruction.
        kdata_dynamic, _ = build_dynamic_kdata(kdata, device=device)

        _, _, _, AHWA_op, rhs, _ = construct_reconstruction_operators(
            kdata_dynamic=kdata_dynamic,
            kdata=kdata,
            device=device,
        )

        return cls._from_reconstruction(
            patient_prefix, kdata_dynamic, rr_idx, tau, AHWA_op, rhs,
            mask_file=f'{RUN_SETTINGS["mask_dir"]}/mask_parameters_{patient_prefix}.json',
            RUN_SETTINGS=RUN_SETTINGS, RECONSTRUCTION_PARAMS=RECONSTRUCTION_PARAMS, device=device, **kwargs,
        )

    @classmethod
    def from_xcat(cls, RUN_SETTINGS, RECONSTRUCTION_PARAMS, data_root=None, **kwargs):
        """Load the XCAT phantom and reconstruct x_moving. Returns None as in `from_patient`.

        data_root: folder with the XCAT mask file; defaults to RUN_SETTINGS['data_root_xcat'].
        """
        if data_root is None:
            data_root = RUN_SETTINGS['data_root_xcat']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        (kdata,
         kdata_dynamic, rr_idx,
         _p_true,
         _sampling_time, _repetition_time, tau, _nominal_fa) = load_xcat_data(RECONSTRUCTION_PARAMS, RUN_SETTINGS, device)

        dcf_op, _, acquisition_op, AHWA_op, _, _ = construct_reconstruction_operators(
            kdata_dynamic=kdata_dynamic,
            kdata=kdata,
            device=device,
        )
        (rhs,) = (acquisition_op.H @ dcf_op)(kdata_dynamic.data)

        return cls._from_reconstruction(
            XCAT_PREFIX, kdata_dynamic, rr_idx, tau, AHWA_op, rhs,
            mask_file=f'{data_root}/mask_parameters_{XCAT_PREFIX}.json',
            RUN_SETTINGS=RUN_SETTINGS, RECONSTRUCTION_PARAMS=RECONSTRUCTION_PARAMS, device=device, **kwargs,
        )

    @classmethod
    def _from_reconstruction(cls, patient_prefix, kdata_dynamic, rr_idx, tau, AHWA_op, rhs, mask_file,
                             RUN_SETTINGS, RECONSTRUCTION_PARAMS, device, **kwargs):
        """CG recon of x_moving, initial reference fit and mask loading shared by from_patient/from_xcat."""
        # Compute an initial moving dynamic with CG reconstruction.
        with torch.no_grad():
            x_moving, = mrpro.algorithms.optimizers.cg(
                AHWA_op,
                rhs,
                initial_value=rhs,
                max_iterations=RECONSTRUCTION_PARAMS['cg_iterations_initial'],
                tolerance=RECONSTRUCTION_PARAMS["cg_tolerance"],
            )
        torch.cuda.empty_cache()

        t1_parameter_estimator = T1ParameterEstimator(kdata=kdata_dynamic, opt_type=RECONSTRUCTION_PARAMS['fitting_opt_type'], tau=tau)

        # Initial T1 parameter fitting with ECG-based mask for motion-compensated references.
        p_0_ref, x_0_ref, _ = find_valid_rr_threshold(
            rr_idx, x_moving, t1_parameter_estimator, logger=None
        )
        if p_0_ref is None or x_0_ref is None:
            return None

        mask = MaskImgSize()
        mask.load_params(mask_file)

        return cls(patient_prefix, x_moving, x_0_ref, t1_parameter_estimator, mask, RUN_SETTINGS,
                   device=device, **kwargs)

    # Figures

    def run(self):
        """Write every figure, in the same order as the original pipeline."""
        self.plot_correlation_heatmaps()
        self.plot_initial_registration()
        if self.plot_registrations:
            self.compute_registered_over_iterations()
        self.plot_final_frames()
        self.plot_t1_over_iterations()
        if self.plot_slices:
            self.plot_line_profiles()
        if self.plot_fields:
            self.plot_motion_fields()
        if self.plot_registrations:
            self.plot_registration_over_iterations()

    def plot_correlation_heatmaps(self):
        """Temporal correlation maps of the motion fields at every iteration."""
        iterations.plot_correlation_heatmaps_over_iterations(
            self.t1_parameter_estimator,
            self.mask.apply(self.all_forward_motion),
            self.mask.apply(self.all_inverse_motion),
            self.figure_dir,
        )

    def plot_initial_registration(self):
        """x moving, the initial reference and x at iteration 0, for each time point."""
        x_moving = DynamicViewer(self.mask.apply(self.x_moving), 'x moving')
        out_dir = f'{self.figure_dir}/iteration_0/registration/'
        os.makedirs(out_dir, exist_ok=True)
        for t in self.times_to_visualize:
            x_moving.compare(
                [self.mask.apply(self.x_0_ref), self.mask.apply(self.all_x[0])],
                t,
                titles=['x_0_ref', 'x registered = x(0)'],
                suptitle=f'Iteration 0, time={t}',
                filepath=f'{out_dir}/time_{t}.png',
            )

    def compute_registered_over_iterations(self):
        """x moving registered with the inverse motion of every iteration."""
        x_registered_all_iter = []
        for iteration in range(self.all_forward_motion.shape[0]):
            motion_op = MotionOperator(
                times=self.x_moving.shape[0],
                img_size=self.x_moving.shape[-2:],
                device=self.device,
                phi=self.all_forward_motion[iteration],
                phi_inv=self.all_inverse_motion[iteration],
            )
            x_registered, = motion_op.inverse(self.x_moving)
            x_registered_all_iter.append(x_registered.cpu())

        self.x_registered_all_iter = torch.stack(x_registered_all_iter, dim=0)
        return self.x_registered_all_iter

    def plot_final_frames(self):
        """Frames of the final x at the selected time points."""
        DynamicViewer(self.mask.apply(self.all_x[-1]), 'x final').plot_frames(
            self.times_to_visualize, f'{self.figure_dir}/final_x_over_time.png')

    def plot_t1_over_iterations(self):
        """T1 maps of every iteration."""
        iterations.plot_t1_over_iterations(self.mask.apply(self.all_p).abs(), f'{self.figure_dir}/t1_over_iterations.png', clim=self.clim)

    def plot_line_profiles(self):
        """Rows and columns of x over time, across iterations (and registered x_moving, if computed)."""
        x_moving = self.mask.apply(self.x_moving).abs().detach().cpu()
        series = {'': self.mask.apply(self.all_x).abs().detach().cpu()}
        if self.x_registered_all_iter is not None:
            series['registered_iters_'] = self.mask.apply(self.x_registered_all_iter).abs().detach().cpu()

        for axis in ('row', 'col'):
            out_dir = f'{self.figure_dir}/x_over_iterations_{axis}/'
            os.makedirs(out_dir, exist_ok=True)
            for index in self.slices_to_visualize:
                for name_prefix, all_x in series.items():
                    iterations.plot_line_profile_over_iterations(x_moving, all_x, f'{out_dir}/{name_prefix}{axis}_{index}.png', index, axis=axis)

    def plot_motion_fields(self):
        """Deformed grids and Jacobian determinants of the last iteration's motion fields."""
        grid_dir = f'{self.figure_dir}/motion_grids_last_iter/'
        det_dir = f'{self.figure_dir}/motion_determinants_last_iter/'
        os.makedirs(grid_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)

        forward = rearrange(self.mask.apply(self.all_forward_motion[-1]), 't ch h w -> ch t h w').detach().cpu()
        inverse = rearrange(self.mask.apply(self.all_inverse_motion[-1]), 't ch h w -> ch t h w').detach().cpu()
        analyzer = MotionFieldAnalyzer()

        for t in self.times_to_visualize:
            analyzer.plot_grid(forward, t=t, step_plot=1, title=f'fw {t}', linewidth=0.1,
                               save_filename=os.path.join(grid_dir, f'phi_t{t}.png'))
            analyzer.plot_grid(inverse, t=t, step_plot=1, title=f'inv {t}', linewidth=0.1,
                               save_filename=os.path.join(grid_dir, f'phi_inv_t{t}.png'))
            analyzer.plot_determinant(inverse, t=t, negative_det=True, suptitle=f'det inv t{t}',
                                      save_filename=os.path.join(det_dir, f'phi_inv_det_t{t}.png'))
            analyzer.plot_determinant(forward, t=t, negative_det=True, suptitle=f'det t{t}',
                                      save_filename=os.path.join(det_dir, f'phi_det_t{t}.png'))

    def plot_registration_over_iterations(self):
        """x moving next to its registered version at every iteration, for the selected time points."""
        if self.x_registered_all_iter is None:
            self.compute_registered_over_iterations()
        registration_dir = f'{self.figure_dir}/registration_all_iter/'
        os.makedirs(registration_dir, exist_ok=True)
        for t in self.times_to_visualize:
            iterations.plot_registration_over_iterations(self.mask.apply(self.x_moving), self.mask.apply(self.x_registered_all_iter),
                                              t, save_plots=True, dir_fig=registration_dir)
