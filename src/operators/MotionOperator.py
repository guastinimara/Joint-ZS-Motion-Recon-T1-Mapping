"""Image warping operator built from displacement fields (forward/inverse motion)."""
import torch
from einops import rearrange    
from operators.GradOperators import GradOperators
import mrpro

class MotionOperator:
    """Warps dynamic images with displacement fields.

    ``forward`` applies the motion phi (moving -> reference), ``inverse`` applies phi_inv, and
    ``adjoint`` approximates M^H with the inverse warp scaled by the Jacobian determinant.
    Fields have shape (times, 2, h, w) in pixels, channel 0 = y and channel 1 = x.
    """
    @staticmethod
    def field_to_normalized_grid(field):
        """Convert a pixel displacement field (b, ch, h, w[, z]) to a sampling grid in [-1, 1] for grid_sample."""
        dim = field.shape[1]

        shape = field.shape[-dim:]
        b = field.shape[0]
        unit_vectors = [torch.arange(0, s) for s in shape]
        grid = torch.meshgrid(unit_vectors)
        unit_grid = torch.stack(grid).unsqueeze(0).type(torch.FloatTensor).to(field.device)
        if dim == 2:
            unit_grid = unit_grid.repeat(b, 1, 1, 1)
        if dim == 3:
            unit_grid = unit_grid.repeat(b, 1, 1, 1, 1)
        warped_grid = unit_grid + field

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            warped_grid[:, i, ...] = 2 * (warped_grid[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # channels need to be reversed
        if len(shape) == 2:
            warped_grid = rearrange(warped_grid, 'b ch h w -> b h w ch')
            warped_grid = warped_grid[..., [1, 0]]
        elif len(shape) == 3:
            warped_grid = rearrange(warped_grid, 'b ch h w z -> b h w z ch')
            warped_grid = warped_grid[..., [2, 1, 0]]

        return warped_grid

    @staticmethod
    def normalized_grid_to_field(grid):
        """
        Inverts field_to_normalized_grid.
        grid: (B, H, W, C) or (B, H, W, Z, C)
        Returns: field (B, C, H, W) or (B, C, H, W, Z)
        """
        shape = grid.shape[1:-1]
        assert len(shape) in (2, 3), f"Expected 2D or 3D grid, got shape {grid.shape}"
        # Reverse channel order
        if len(shape) == 2:
            grid = grid[..., [1, 0]]
        elif len(shape) == 3:
            grid = grid[..., [2, 1, 0]]
        # Move channels to first dimension
        if len(shape) == 2:
            grid = rearrange(grid, 'b h w ch -> b ch h w')
        elif len(shape) == 3:
            grid = rearrange(grid, 'b h w z ch -> b ch h w z')
        # Undo normalization
        for i, s in enumerate(shape):
            grid[:, i, ...] = ((grid[:, i, ...] + 1) / 2) * (s - 1)
        # Subtract unit grid
        unit_vectors = [torch.arange(0, s, device=grid.device, dtype=grid.dtype) for s in shape]
        mesh = torch.meshgrid(*unit_vectors, indexing='ij')
        unit_grid = torch.stack(mesh).unsqueeze(0)
        field = grid - unit_grid
        return field

    def __init__(
        self,
        M_op=None,
        M_op_inv=None,
        phi=None,
        phi_inv=None,
        times=None,
        img_size=None,
        device='cuda',
        interpolation_mode='bicubic',
        padding_mode='zeros',
    ):
        """
        Args:
            M_op, M_op_inv: existing mrpro GridSamplingOp for the forward/inverse warp (optional).
            phi, phi_inv: displacement fields (times, 2, h, w). If only one is given, the other is
                approximated by its negative.
            times, img_size: used to build zero (identity) fields when nothing else is given.
            device: torch device.
            interpolation_mode, padding_mode: passed to GridSamplingOp.
        """
        self.device = device
        if M_op is None or M_op_inv is None:
            if phi is None and phi_inv is None:
                if times is None or img_size is None:
                    raise ValueError(
                        'If M_op/M_op_inv are not provided, pass times and img_size, or provide phi/phi_inv.'
                    )
                phi = torch.zeros((int(times), 2, *img_size), device='cpu')
                phi_inv = -phi
            elif phi is None:
                phi = -phi_inv
            elif phi_inv is None:
                phi_inv = -phi

            M_op = mrpro.operators.GridSamplingOp.from_displacement(
                displacement_z=None,
                displacement_y=phi[:, 0],
                displacement_x=phi[:, 1],
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
            ).to(self.device)

            M_op_inv = mrpro.operators.GridSamplingOp.from_displacement(
                displacement_z=None,
                displacement_y=phi_inv[:, 0],
                displacement_x=phi_inv[:, 1],
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
            ).to(self.device)

        self.M_op = M_op
        self.M_op_inv = M_op_inv

        if phi is None:
            phi = self.normalized_grid_to_field(self.M_op.grid)
        if phi_inv is None:
            phi_inv = self.normalized_grid_to_field(self.M_op_inv.grid)

        self.phi = phi.to(self.device)
        self.phi_inv = phi_inv.to(self.device)
        # self.psi = self.phi_inv

        self.M_op.grid = self.field_to_normalized_grid(self.phi)
        self.M_op_inv.grid = self.field_to_normalized_grid(self.phi_inv)
        
    def forward(self, x):
        """Warp x with phi."""
        return self.M_op(x)

    def inverse(self, x):
        """Warp x with phi_inv."""
        return self.M_op_inv(x)
    
    @staticmethod
    def jacobian_determinant(disp, return_matrix=False): # ch t h w
        """Determinant of the Jacobian of the deformation identity + disp, per time point (t h w)."""
        def identity_map(shape, device='cpu'): # identity, as derivative of identity mapping
            h, w = shape
            ones = torch.ones(size=(h, w, 1, 1), device=device)
            zeros = torch.zeros(size=(h, w, 1, 1), device=device)
            g0 = torch.cat((ones, zeros), dim=2)
            g1 = torch.cat((zeros, ones), dim=2)
            identity = torch.stack([g0, g1], dim=2).squeeze()
            return identity.squeeze()

        disp = disp.unsqueeze(0)  # add batch dimension
        b, ch, t, h, w = disp.shape

        identity = identity_map((h,w), device=disp.device)
        finite_diff =  GradOperators(2).to(disp.device)
        grad_disp = finite_diff(rearrange(disp, 'b ch t h w -> (b t) h w ch'))

        grad_disp = rearrange(grad_disp, 'bt h dx w dy -> bt h w dx dy')

        grad_deformation = grad_disp + identity.unsqueeze(0)
        det = grad_deformation.det()
        if return_matrix:
            return det, grad_deformation
        return det

    def get_determinant(self, disp): # t ch h w
        """Jacobian determinant for a field given as (t, ch, h, w)."""
        return self.jacobian_determinant(rearrange(disp, 't ch h w -> ch t h w'))

    def adjoint(self, y):
        """Approximate adjoint: inverse warp weighted by |det J(phi_inv)|."""
        M_inv_y, = self.M_op_inv(y)
        det_J_inv = self.get_determinant(self.phi_inv[:, [1, 0]])
        abs_det = torch.abs(det_J_inv)
        return (M_inv_y * abs_det.unsqueeze(1).unsqueeze(1).unsqueeze(1),)
        #return (M_inv_y,)
    
    def gram(self, x):
        """M^H M x."""
        Mx, = self.M_op(x)
        MH_Mx = self.adjoint(Mx)
        return MH_Mx
    
    
