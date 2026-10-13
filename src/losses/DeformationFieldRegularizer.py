"""Regularizers for deformation fields used in motion-estimation losses."""

import torch
from einops import rearrange
from operators.GradOperators import GradOperators

class DeformationFieldRegularizer:
    """Regularization utilities for 2D deformation/displacement fields.

    The class provides Jacobian- and divergence-based operators that can be used
    as smoothness and topology-preserving penalty terms during motion estimation.
    """

    @staticmethod
    def _identity_map(shape):
        """Return a 2x2 identity tensor field over a 2D grid of shape (h, w)."""
        h, w = shape
        ones = torch.ones(size=(h, w, 1, 1))
        zeros = torch.zeros(size=(h, w, 1, 1))
        g0 = torch.cat((ones, zeros), dim=2)
        g1 = torch.cat((zeros, ones), dim=2)
        identity = torch.stack([g0, g1], dim=2).squeeze()
        return identity.squeeze()

    def __init__(self, shape, device='cpu'):
        """Initialize regularization operators for a fixed spatial grid.

        Args:
            shape: Spatial shape ``(h, w)`` of the deformation field.
            device: Torch device where internal tensors/operators are allocated.
        """
        self.shape = shape
        self.identity_map = self._identity_map(self.shape).to(device)
        self.finite_differences = GradOperators(len(self.shape)).to(device)
        self.device = device
        
        
    def determinant_jacobian(self, disp):  # b ch t h w
        """Compute Jacobian determinants of the deformation map associated to displacement.

        The deformation gradient is built as ``I + grad(disp)`` and the
        determinant is computed per spatial location.

        Args:
            disp: Displacement field tensor with shape compatible with the
                configured finite-difference operator.

        Returns:
            Tensor of local Jacobian determinants.
        """
        identity = self.identity_map
        grad_disp = self.finite_differences(disp)
        grad_disp = rearrange(grad_disp, 'bt dx dy h w -> bt h w dx dy')

        grad_deformation = grad_disp + identity.unsqueeze(0)
        det = grad_deformation.det()
        return det


    def divergence(self, disp):
        """Compute discrete divergence of a 2D displacement field.

        Uses first-order forward differences and pads borders to preserve
        the input spatial size.

        Args:
            disp: Displacement tensor with shape ``(n, 2, h, w)``.

        Returns:
            Divergence map with shape ``(n, h, w)``.
        """
        # Channel 0 displaces along h (y), channel 1 along w (x), as in the SpatialTransformer.
        v_y = disp[:, 0, :, :]
        v_x = disp[:, 1, :, :]

        # div(v) = d v_y / dy + d v_x / dx
        d_v_y_dy = v_y[:, 1:, :] - v_y[:, :-1, :]
        d_v_x_dx = v_x[:, :, 1:] - v_x[:, :, :-1]

        d_v_y_dy = torch.nn.functional.pad(d_v_y_dy, (0, 0, 0, 1))
        d_v_x_dx = torch.nn.functional.pad(d_v_x_dx, (0, 1, 0, 0))

        divergence = d_v_y_dy + d_v_x_dx
        return divergence
    
    def divergence_loss(self, disp, mag=False):
        """Return mean-squared divergence as a smoothness regularizer.

        Args:
            disp: Displacement tensor with shape ``(n, 2, h, w)``.
            mag: If ``True``, weight divergence by normalized displacement
                magnitude before reduction.

        Returns:
            Scalar divergence regularization loss.
        """
        divergence = self.divergence(disp)
        if mag:
            magnitude = torch.sqrt(disp[:, 0] ** 2 + disp[:, 1] ** 2)
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            divergence = divergence * magnitude
        return (divergence ** 2).mean()
    
    def determinant_loss(self, disp):
        """Penalize deviation of Jacobian determinant from 1.

        This encourages locally volume-preserving deformations.

        Args:
            disp: Displacement field tensor.

        Returns:
            Scalar mean-squared penalty ``E[(det(J) - 1)^2]``.
        """
        determinant = self.determinant_jacobian(disp)
        return ((determinant - 1) ** 2).mean()
        
    def determinant_negative_penalty(self, disp):
        """Penalize negative Jacobian determinants.

        Negative determinants indicate local foldings/non-invertible mappings.

        Args:
            disp: Displacement field tensor.

        Returns:
            Scalar penalty based on squared negative part of ``det(J)``.
        """
        determinant = self.determinant_jacobian(disp)
        return torch.mean(torch.square(torch.clamp(determinant, max=0.0)))

    
  