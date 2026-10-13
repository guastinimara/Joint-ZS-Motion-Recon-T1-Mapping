"""Training loop of the motion network (zero-shot: optimized on the scan itself, no training data)."""

from .NetworkMotionFieldsWithInverse import NetworkMotionFieldsWithInverse
import torch

import matplotlib.pyplot as plt
from operators.MotionOperator import MotionOperator
import mrpro
from operators.GradOperators import GradOperators
from losses.DeformationFieldRegularizer import DeformationFieldRegularizer
from losses.MutualInformation import MutualInformation

class MotionEstimationWithInverse:
    """Estimates forward/inverse motion fields between moving dynamics and a reference image.

    The loss combines mutual information between warped and target images, an optional
    k-space data consistency term, and gradient + divergence smoothness penalties on the fields.
    The estimated fields are written into ``motion_op``.
    """
    def __init__(self,
                 kdata_dynamic,
                 acquisition_op,
                 dcf_sqrt_op,
                 motion_op,
                 img_size, 
                 times, 
                 mask_roi,
                 lr=1e-4, 
                 weight_decay=1e-5, 
                 optimizer='adam', 
                 ch_out_unet=32, 
                 n_enc_stages=4, 
                 n_filters=32,
                 weight_dc=0.5,
                 weight_mi=0.5,
                 lambd=0.05,
                 mu=0.01,
                 device='cpu'):
        """
        Args:
            kdata_dynamic: measured dynamic k-space (KData or tensor), for the data consistency term.
            acquisition_op, dcf_sqrt_op: acquisition operator A and sqrt of the density compensation.
            motion_op: MotionOperator updated with the estimated fields.
            img_size: full image size (ny, nx).
            times: number of dynamics.
            lr, weight_decay, optimizer: optimizer settings ('adam' or 'adamw'; anything else falls back to 'adam').
            mask_roi: MaskImgSize crop around the heart; the network works on the cropped images.
            ch_out_unet, n_enc_stages, n_filters: U-Net size.
            weight_dc, weight_mi: weights of data consistency and similarity terms.
            lambd, mu: weights of gradient and divergence regularization of the fields.
            device: torch device.
        """

        self.device = device
        self.mask_roi = mask_roi

        if mask_roi is not None:
            self.imshape = mask_roi.get_masked_shape()
            self.imshape_original = img_size
        else:
            self.imshape = img_size
        self.T = times

        self.net = NetworkMotionFieldsWithInverse(dim=2,
                                            inshape=self.imshape,
                                            src_feats=self.T,
                                            out_feats=self.T * 2,
                                            ch_out_unet=ch_out_unet,
                                            n_enc_stages=n_enc_stages,
                                            int_steps=7,
                                            int_downsize=2,
                                            n_filters=n_filters).to(self.device)

        self.lr = lr

        if optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr,
                                            weight_decay=weight_decay)
        else:
            if optimizer != 'adam':
                print(f"Unknown optimizer '{optimizer}', using 'adam'.")
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,
                                            weight_decay=weight_decay)

        self.motion_op = motion_op

        self.acquisition_op = acquisition_op
        self.dcf_sqrt_op = dcf_sqrt_op
        if isinstance(kdata_dynamic, mrpro.data.KData):
            self.kdata_dynamic = kdata_dynamic.data.to(self.device)
        else:
            self.kdata_dynamic = kdata_dynamic.to(self.device)

        self.mutual_info = MutualInformation(
                                    sigma=0, num_bins=128,
                                    patch_size=self.imshape[0],
                                    overlap=0,normalize=True, device=self.device)

        self.mse = torch.nn.MSELoss().to(self.device)
        self.grad_loss = GradOperators(dim=2, loss_mult=2).to(self.device).loss
        self.lambd = lambd

        self.div_loss = DeformationFieldRegularizer(self.imshape, device=self.device)
        self.mu = mu

        self.weight_dc = weight_dc
        self.weight_mi = weight_mi

        self.total_loss = []

    def train_net(self, imgs_m, imgs_f, 
                  episodes=1,
                  steps=100, 
                  full_loss_phi=None, 
                  patience = 10):
        """Optimize the network to register the dynamics.

        Args:
            imgs_m: moving dynamic images [nt, 1, 1, ny, nx]; network input, and target of the
                reference images warped with phi.
            imgs_f: reference (motion-free) images, warped with phi to match ``imgs_m``.
            episodes, steps: number of episodes and optimization steps per episode.
            full_loss_phi: episode from which the k-space data consistency term is added (None: never).
            patience: steps without improvement of the similarity loss before early stopping the episode.
        """
        # images shape [nt, ncoils=1, 1, ny, nx]
        self.net.train()

        imgs_m = imgs_m.squeeze().unsqueeze(0)
        imgs_f = imgs_f.squeeze().unsqueeze(0)
        imgs_m_clone = imgs_m.clone()

        initial_inputs = self._network_input(imgs_m)
        y_in_loss = torch.view_as_real(self.dcf_sqrt_op(self.kdata_dynamic)[0])

        for episode in range(episodes):
            best_loss = float('inf')
            no_improvement_steps = 0
            inputs_net = initial_inputs.clone().detach()
            use_dc_loss = full_loss_phi is not None and episode >= full_loss_phi

            for step in range(steps):
                preint_fields, motion_fields = self.net(inputs_net)
                pos_motion_fields, neg_motion_fields = self._set_motion_fields(motion_fields)
                Mx, M_inv_Mx = self._warp(imgs_f, imgs_m)

                loss = 0.0
                matching_loss = None
                if use_dc_loss:
                    matching_loss = self._data_consistency_loss(Mx, y_in_loss)
                    loss += matching_loss

                reg_matching_loss = self._similarity_loss(Mx, M_inv_Mx, imgs_m_clone, imgs_f)
                loss += reg_matching_loss

                grad_loss, divergence_loss = self._regularization_loss(preint_fields)
                loss += grad_loss + divergence_loss

                # Perform backpropagation and update the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._log_step(loss, reg_matching_loss, grad_loss, divergence_loss, matching_loss,
                               done_steps=(steps * episode) + (step + 1), total_steps=episodes * steps)

                if reg_matching_loss < best_loss:
                    best_loss = reg_matching_loss
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1

                if no_improvement_steps >= patience:
                    print(f" -> early stopping at step: {step}, next episode.")
                    break

                # Next input: the moving images registered with the current inverse fields
                inputs_net = self._network_input(M_inv_Mx)
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        self._store_motion_fields(pos_motion_fields, neg_motion_fields)

    # Helpers of train_net

    def _network_input(self, imgs):
        """Network input from images (..., h, w): magnitude, cropped to the ROI, shaped (1, t, h, w), in [0, 1]."""
        imgs = imgs.detach().abs().squeeze().unsqueeze(0)
        return normalize(self.mask_roi.apply(imgs))

    def _set_motion_fields(self, motion_fields):
        """Pad the network fields back to the full image size and set them in motion_op.

        Returns the full-size forward and inverse fields (t, 2, ny, nx).
        """
        pos_motion_fields = self.mask_roi.inverse_mask_on_mf(motion_fields[0], output_shape=self.imshape_original)
        neg_motion_fields = self.mask_roi.inverse_mask_on_mf(motion_fields[1], output_shape=self.imshape_original)

        self.motion_op.M_op.grid = MotionOperator.field_to_normalized_grid(pos_motion_fields)
        self.motion_op.M_op_inv.grid = MotionOperator.field_to_normalized_grid(neg_motion_fields)
        return pos_motion_fields, neg_motion_fields

    def _warp(self, imgs_f, imgs_m):
        """Warp the reference with phi (Mx) and the moving images with phi_inv (M_inv_Mx), shaped (t, 1, ny, nx)."""
        (Mx,) = self.motion_op.M_op(imgs_f.squeeze().unsqueeze(1))
        (M_inv_Mx,) = self.motion_op.M_op_inv(imgs_m.squeeze().unsqueeze(1))
        return Mx, M_inv_Mx

    def _data_consistency_loss(self, Mx, y_in_loss):
        """Weighted k-space MSE between sqrt(W) A Mx and the measured sqrt(W) y."""
        (AMx,) = self.acquisition_op(Mx.unsqueeze(1).unsqueeze(1))
        return self.mse(torch.view_as_real(self.dcf_sqrt_op(AMx)[0]), y_in_loss) * self.weight_dc

    def _roi_magnitude(self, imgs):
        """Magnitude cropped to the ROI, shaped (t, 1, h, w) and normalized to [0, 1], for the MI loss."""
        return normalize(self.mask_roi.apply(imgs).abs().squeeze().unsqueeze(1))

    def _similarity_loss(self, Mx, M_inv_Mx, imgs_m, imgs_f):
        """Weighted negative MI in the ROI: Mx vs the moving images, plus M_inv_Mx vs the reference."""
        reg_matching_loss_1 = self.mutual_info(self._roi_magnitude(Mx), self._roi_magnitude(imgs_m)).mean()
        reg_matching_loss_2 = self.mutual_info(self._roi_magnitude(M_inv_Mx), self._roi_magnitude(imgs_f)).mean()
        return (reg_matching_loss_1 + reg_matching_loss_2) * self.weight_mi

    def _regularization_loss(self, preint_fields):
        """Weighted gradient (smoothness) and divergence losses of the velocity field."""
        grad_loss = self.lambd * torch.mean(self.grad_loss(preint_fields))
        divergence_loss = self.mu * torch.mean(self.div_loss.divergence_loss(preint_fields))
        return grad_loss, divergence_loss

    def _log_step(self, loss, reg_matching_loss, grad_loss, divergence_loss, matching_loss, done_steps, total_steps):
        """Store the losses in total_loss and print the progress line (DC only when it is used)."""
        self.total_loss.append((loss.item(),
                reg_matching_loss.item(),
                (grad_loss + divergence_loss).item()
                ))
        message = f'\r{done_steps}/{total_steps}: TOT: {loss.item():.4f}, MI: {reg_matching_loss.item():.4f}, REG (no w) {(grad_loss.item()/ self.lambd):.4f}'
        if matching_loss is not None:
            message += f', DC (w): {matching_loss.item():.4f}'
        print(message, end='')

    def _store_motion_fields(self, pos_motion_fields, neg_motion_fields):
        """Write the final (detached) fields into motion_op."""
        self.motion_op.phi = pos_motion_fields.detach()
        self.motion_op.phi_inv = neg_motion_fields.detach()

        self.motion_op.M_op.grid = MotionOperator.field_to_normalized_grid(pos_motion_fields.detach())
        self.motion_op.M_op_inv.grid = MotionOperator.field_to_normalized_grid(neg_motion_fields.detach())

    def plot_loss(self, position=0):
        """Plot a loss component over steps (0: total, 1: similarity, 2: regularization)."""
        loss_to_plot = [loss[position] for loss in self.total_loss]
        plt.plot(loss_to_plot)
        plt.xlabel('Epoch')
        plt.ylabel(f'Loss[{position}]')
        plt.show()


def normalize(tensor):
    """Min-max normalize tensor to [0, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val + 1e-10)
    return normalized
