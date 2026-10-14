"""Voxel-wise estimation of M0, T1 and flip angle from the dynamic images.

Uses the mrpro inversion-recovery transient steady state model
and fits it with L-BFGS or Adam under box constraints.
"""
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from mrpro.operators import ConstraintsOp
from mrpro.operators.models import TransientSteadyStateWithPreparation
from mrpro.operators.functionals import MSE
from mrpro.algorithms.optimizers import lbfgs, adam
from einops import rearrange
from scipy.optimize import curve_fit
import mrpro


class T1ParameterEstimator:
    def __init__(self, kdata=None, sampling_time = None, repetition_time = None, 
                 nominal_flip_angle=None, device='cpu', opt_type='lbfgs', tau=0.):
        """
        Initialize the estimator from kdata, or from explicit sequence timings.

        Args:
            kdata: KData to read sampling times, echo spacing (TR) and nominal flip angle from.
            sampling_time, repetition_time, nominal_flip_angle: used when kdata is None.
            device: torch device (when kdata is None).
            opt_type: 'lbfgs' or 'adam'.
            tau: delay after the inversion pulse, added to the sampling times.
        """
        if kdata is not None:
            self.device = kdata.data.device
            if kdata.header.echo_spacing is None:
                raise ValueError('Echo spacing needs to be defined in kdata.')
            self.sampling_time = torch.mean(kdata.header.acq_info.acquisition_time_stamp[:, 0, :, 0].to(torch.float32), dim=-1)
            self.sampling_time = self.sampling_time.mean(-1)
            self.sampling_time = self.sampling_time - self.sampling_time[0]
            self.sampling_time = self.sampling_time.squeeze().to(self.device)
            self.repetition_time = kdata.header.echo_spacing[0]
            self.nominal_flip_angle = float(kdata.header.fa[0])
        
        else:
            self.device = device
            if sampling_time is None or repetition_time is None or nominal_flip_angle is None:
                raise ValueError('If kdata is not provided, sampling_time, repetition_time and nominal_flip_angle must be provided.')
            self.sampling_time = sampling_time - sampling_time[0]
            self.sampling_time = self.sampling_time.to(torch.float32).to(self.device)
            self.repetition_time = repetition_time
            self.nominal_flip_angle = nominal_flip_angle
        
        self.tau = tau
        self.sampling_time += self.tau
        
        # print('=' * 50)
        # print('Sampling time:', self.sampling_time)
        # print('Repetition time:', self.repetition_time)
        # print('Nominal flip angle:', self.nominal_flip_angle)
        # print('=' * 50)
        
        self.model_op_full_standard = TransientSteadyStateWithPreparation(
            sampling_time=self.sampling_time,
            repetition_time=self.repetition_time,
            m0_scaling_preparation=-1.0,
            delay_after_preparation=tau
        ).to(self.device)
        
        
        self.opt_type = opt_type
        if self.opt_type not in ['lbfgs', 'adam']:
            raise ValueError('opt_type must be either "lbfgs" or "adam".')
        

    @staticmethod
    def construct_model_op(sampling_time, repetition_time, tau=0., device='cpu', magnitude_model=False):
        """
        Construct the signal model operator for a given sampling time.

        Args:
            sampling_time, repetition_time, tau: sequence timings.
            device: torch device.
            magnitude_model: fit the magnitude of the signal instead of the complex signal.
        """
        
        model_op = TransientSteadyStateWithPreparation(
            sampling_time=sampling_time,
            repetition_time=repetition_time,
            m0_scaling_preparation=-1.0,
            delay_after_preparation=tau
        ).to(device)
       
        if magnitude_model:
            return mrpro.operators.MagnitudeOp() @ model_op
        else:
            return model_op
    
    @staticmethod
    def construct_functional(x_m, model_op,
                             recon_params=None,
                             magnitude_model=False):
        """
        Construct the functional for optimization: MSE(x_m, model(constraints(p))).

        Args:
            x_m: normalized target images.
            model_op: signal model from ``construct_model_op``.
            recon_params: config dict with 't1_constraints' and 'alpha_constraints'; defaults if None.
            magnitude_model: compare magnitudes.

        Returns:
            functional, constraints_op.
        """
    

        if recon_params is None:
            constraints_op = ConstraintsOp(
                bounds=(
                    (None, None),  # M0 is not constrained
                    (0.1, 6.),  # T1 is constrained
                    (math.radians(1), math.radians(9))  # alpha is constrained
                ),
                beta_sigmoid=2.0,
                eps=1e-7
            ).to(x_m.device)
        else:
            constraints_op = ConstraintsOp(
                                bounds=(
                                    (None, None),  # M0 is not constrained
                                    recon_params['t1_constraints'],  # T1 is constrained
                                    recon_params['alpha_constraints']  # alpha is constrained
                                ),
                                beta_sigmoid=2.0,
                                eps=1e-7
                            ).to(x_m.device)
        
        if magnitude_model:
            mse_loss = MSE(x_m.abs()).to(x_m.device)
        else:
            mse_loss = MSE(x_m).to(x_m.device)
            
        functional = mse_loss @ model_op @ constraints_op
        return functional, constraints_op

    def fit(self, x, mask=None,
                initial_p=None,
                callback=None,
                recon_params=None):
        """
        Fit the signal model to the (motion-corrected) dynamic images.

        Args:
            x: dynamic images (t, ..., ny, nx).
            mask: optional boolean selection of dynamics to use in the fit.
            initial_p: optional (m0, flip_angle, t1) starting values; otherwise first image, T1=1 s
                and the nominal flip angle.
            beta_sigmoid: sharpness of the constraint sigmoid.
            callback: optional optimizer callback.
            recon_params: config dict with L-BFGS settings and constraints.

        Returns:
            p_moco: stacked [m0, flip_angle, t1.
            qp_moco: model images for all dynamics, rescaled to the input intensity.
        """
        # Compute sampling times (assumes mask_function is provided)
        x = x.to(self.device)
        
        if recon_params is not None:
            magnitude_model = recon_params['magnitude_model_op']
        else:
            magnitude_model = False
            
        
        if mask is not None:
            sampling_time = self.sampling_time[mask]
            x_masked = x[mask]
        else:
            sampling_time = self.sampling_time
            x_masked = x
        
        model_op = T1ParameterEstimator.construct_model_op(
            sampling_time, 
            self.repetition_time, 
            tau=self.tau,device=self.device, 
            magnitude_model=magnitude_model)
        
        

        # Construct functional
        x_m = x_masked
        factor = x_masked.abs().max()
        x_m = x_m / factor
        functional, constraints_op = T1ParameterEstimator.construct_functional(x_m, model_op,
                                                               recon_params=recon_params,
                                                               magnitude_model=magnitude_model)

        # Initialize parameters
        if initial_p is None:
            m0_start = (x_m)[0]
            if magnitude_model:
                m0_start = m0_start.abs()
            
            #m0_start = torch.view_as_real(m0_start)
            t1_start = torch.ones(m0_start.shape).to(self.device)
            flip_angle_start = torch.ones_like(t1_start) * torch.as_tensor(self.nominal_flip_angle)
            flip_angle_start = flip_angle_start.to(self.device)
        
        else:
            m0_start, flip_angle_start, t1_start = initial_p
            m0_start = m0_start.to(self.device)
            flip_angle_start = flip_angle_start.to(self.device)
            t1_start = t1_start.to(self.device)
       
        initial_parameters = constraints_op.inverse(m0_start, t1_start, flip_angle_start)
            
        
        if self.opt_type == 'adam':
            result = adam(
                functional,
                initial_parameters=initial_parameters,
                max_iterations=100,
                tolerance_change=1e-8,
                learning_rate=1e-2,
                weight_decay=1e-3,
                decoupled_weight_decay=True,
                amsgrad=True,
                callback=callback
            )

        # Perform lbfgs optimization
        else:
            if recon_params is not None:
                result = lbfgs(
                    functional,
                    initial_parameters=initial_parameters,
                    max_iterations= recon_params['lbfgs_max_iterations'],
                    max_evaluations=recon_params['lbfgs_max_evaluations'],
                    line_search_fn='strong_wolfe',
                    callback=callback,
                    tolerance_grad=recon_params['lbfgs_tolerance_grad'], 
                    tolerance_change=recon_params['lbfgs_tolerance_change'],
                )
            else:
                result = lbfgs(
                    functional,
                    initial_parameters=initial_parameters,
                    max_iterations= 300,
                    max_evaluations=200,
                    line_search_fn='strong_wolfe',
                    callback=callback,
                    tolerance_grad=1e-08,
                    tolerance_change=1e-10,
                )
        # Extract parameters
        
        m0, t1, flip_angle = (p.detach().squeeze() for p in constraints_op(*result))
       
        m0 = m0.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        t1 = t1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        flip_angle = flip_angle.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        p_moco = torch.stack([m0, flip_angle, t1], dim=0)
        
        qp_moco, = self.model_op_full_standard.forward(m0=m0, t1=t1, flip_angle=flip_angle)
        qp_moco = qp_moco * factor
        return p_moco, qp_moco
    
    def get_params(self, p):
        """
        Split stacked parameters into m0, flip_angle, t1.
        """
        p = p.squeeze()
        m0, flip_angle, t1 = p[0], p[1], p[2]
        return m0, flip_angle, t1

    @staticmethod
    def _sinusoidal(t, A, B, C, D):
        """A sin(B t + C) + D."""
        return A * np.sin(B * t + C) + D

    @staticmethod
    def _compute_correlation_map(fields, show_heatmap=False):
        """Normalized (h, w) map of the lag-1 temporal autocorrelation of motion fields (t, 2, h, w)."""
        def compute_autocorrelation(motion_field, lag=1):
            autocorr = torch.zeros_like(motion_field)
            for t in range(motion_field.size(0) - lag):
                autocorr += motion_field[t] * motion_field[t + lag]
            autocorr /= (motion_field.size(0) - lag)
            return autocorr

        correlation_factor = compute_autocorrelation(fields, lag=1)
        correlation_magnitude = torch.sqrt(correlation_factor[:, 0] ** 2 + correlation_factor[:, 1] ** 2)
        heatmap = torch.mean(correlation_magnitude, dim=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        if show_heatmap:
            plt.imshow(heatmap.cpu().numpy())
            plt.colorbar()
            plt.show()
        return heatmap
    
    @staticmethod
    def get_mean_heat_map(forward_motion, inverse_motion):
        """Average of the correlation maps of forward and inverse fields."""
        heat_map_fw = T1ParameterEstimator._compute_correlation_map(forward_motion, show_heatmap=False)
        heat_map_inv = T1ParameterEstimator._compute_correlation_map(inverse_motion, show_heatmap=False)
        return (heat_map_fw + heat_map_inv) / 2

    @staticmethod
    def select_sinusoidal_mask(masked_motion_fields, percentage=0.85, plot=False):
        """Select dynamics with low cardiac motion.

        Fits a sinusoid to the mean motion amplitude over time and keeps the dynamics below the
        ``percentage`` percentile of the fitted curve.

        Args:
            masked_motion_fields: motion fields (t, 2, h, w), usually cropped around the heart.
            percentage: fraction of dynamics to keep (e.g. 0.85).
            plot: show and save the selection plots.

        Returns:
            Boolean array (t,) of selected dynamics, or None if the fit fails.
        """
        # Compute the mean amplitude of the motion fields.
        amplitudes = torch.sqrt(torch.sum(masked_motion_fields ** 2, dim=1))
        corr_heatmap = T1ParameterEstimator._compute_correlation_map(masked_motion_fields, show_heatmap=False)
        mean_amplitudes = torch.mean(corr_heatmap.unsqueeze(0) * amplitudes, dim=(1, 2))
        mean_amplitudes = mean_amplitudes.cpu().numpy()
        time = np.arange(len(mean_amplitudes))

        # Initial guesses for sinusoidal parameters.
        A_guess = (np.max(mean_amplitudes) - np.min(mean_amplitudes)) / 2
        B_guess = 2 * np.pi / (len(mean_amplitudes) / 2)
        C_guess = 0
        D_guess = np.mean(mean_amplitudes)
        initial_guesses = [A_guess, B_guess, C_guess, D_guess]

        fitted_curve = None
        idx_constraint_sinusoidal_mask = None
        try:
            params, _ = curve_fit(T1ParameterEstimator._sinusoidal, time, mean_amplitudes, p0=initial_guesses)
            A, B, C, D = params
            fitted_curve = T1ParameterEstimator._sinusoidal(time, A, B, C, D)

            threshold = np.percentile(fitted_curve, percentage * 100)
            idx_constraint_sinusoidal_mask = fitted_curve < threshold
        except Exception as e:
            print(f"Curve fitting failed: {e}. Proceeding without constraints.")

        if plot and fitted_curve is not None and idx_constraint_sinusoidal_mask is not None:
            T1ParameterEstimator.plot_sinusoidal_selection(
                mean_amplitudes,
                time,
                fitted_curve,
                time[idx_constraint_sinusoidal_mask],
                mean_amplitudes[idx_constraint_sinusoidal_mask],
                show_plots=True,
                save_plots=True,
            )

        return idx_constraint_sinusoidal_mask

    @staticmethod
    def plot_sinusoidal_selection(
        mean_amplitudes,
        time,
        fitted_curve,
        selected_times,
        selected_amplitudes,
        show_plots=False,
        save_plots=False,
        save_dir='',
    ):
        """Plot mean amplitudes, the fitted sinusoid and the selected dynamics."""
        plt.figure(figsize=(10, 5))
        plt.plot(mean_amplitudes, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Mean Amplitude')
        plt.title('Mean Amplitude of Motion Fields over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(time, mean_amplitudes, marker='o', label='Amplitudes')
        plt.plot(time, fitted_curve, 'g--', label='Fitted sinusoid')
        plt.title('Sinusoidal fit to amplitudes')
        plt.xlabel('Time')
        plt.ylabel('Mean Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{save_dir}/mean_amplitudes_sinus_graph.png')
            if show_plots:
                plt.show()
            else:
                plt.close()
        else:
            plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(mean_amplitudes, marker='o')
        plt.plot(selected_times, selected_amplitudes, 'rx')
        plt.title('Selected Amplitudes excluding peaks of fitted curve')
        plt.xlabel('Time')
        plt.ylabel('Mean Amplitude')
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{save_dir}/mean_amplitudes_sinus_picked.png')
            if show_plots:
                plt.show()
            else:
                plt.close()
        else:
            plt.close()
    
    