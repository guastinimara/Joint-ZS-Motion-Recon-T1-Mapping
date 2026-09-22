"""Finite-difference gradient operator (forward, adjoint and normal operator) via convolutions."""
from functools import partial
import torch

class GradOperators(torch.nn.Module):
    @staticmethod
    def diff_kernel(ndim, mode):
        """Return the (ndim, 1, 3, ..., 3) convolution kernel of the chosen difference scheme."""
        if mode == "doublecentral":
            kern = torch.tensor((-1, 0, 1))
        elif mode == "central":
            kern = (torch.tensor((-1, 0, 1)) / 2)
        elif mode == "forward":
            kern = torch.tensor((0, -1, 1))
        elif mode == "backward":
            kern = torch.tensor((-1, 1, 0))
        else:
            raise ValueError(f"mode should be one of (central, forward, backward, doublecentral), not {mode}")
        kernel = torch.zeros(ndim, 1, *(ndim * (3,)))
        for i in range(ndim):
            idx = tuple([i, 0, *(i * (1,)), slice(None), *((ndim - i - 1) * (1,))])
            kernel[idx] = kern
        return kernel

    def __init__(self, dim:int=2, mode:str="doublecentral", padmode:str = "circular", loss_mult=2):
        """
        An Operator for finite Differences / Gradients
        Implements the forward as apply_G and the adjoint as apply_GH.
        
        Args:
            dim (int, optional): Dimension. Defaults to 2.
            mode (str, optional): one of doublecentral, central, forward or backward. Defaults to "doublecentral".
            padmode (str, optional): one of constant, replicate, circular or refelct. Defaults to "circular".
            loss_mult (float, optional): scaling factor applied in ``loss``. Defaults to 2.
        """
        super().__init__()
        self.register_buffer("kernel", self.diff_kernel(dim, mode), persistent=False)
        self._dim = dim
        self._conv = (torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d)[dim - 1]
        self._convT = (torch.nn.functional.conv_transpose1d, torch.nn.functional.conv_transpose2d, torch.nn.functional.conv_transpose3d)[dim - 1]
        self._pad = partial(torch.nn.functional.pad, pad=2 * dim * (1,), mode=padmode)
        if mode == 'central':
            self._norm = (self.dim) ** (1 / 2)
        else:
            self._norm = (self.dim * 4) ** (1 / 2)
        self.loss_mult=loss_mult

    @property
    def dim(self):
        return self._dim
    
    def apply_G(self, x):
        """
        Forward: gradient G x. Input (..., *spatial), output (..., dim, *spatial). Supports complex.
        """
        if x.is_complex():
            xr = torch.view_as_real(x).moveaxis(-1, 0)
        else:
            xr = x
        xr = xr.reshape(-1, 1, *x.shape[-self.dim :])
        xp = self._pad(xr)
        y = self._conv(xp, weight=self.kernel, bias=None, padding=0)
        if x.is_complex():
            y = y.reshape(2, *x.shape[: -self.dim], self.dim, *x.shape[-self.dim :])
            y = torch.view_as_complex(y.moveaxis(0, -1).contiguous())
        else:
            y = y.reshape(*x.shape[0 : -self.dim], self.dim, *x.shape[-self.dim :])
        return y

    def apply_GH(self, x):
        """
        Adjoint: G^H x. Input (..., dim, *spatial), output (..., *spatial).
        """
        if x.is_complex():
            xr = torch.view_as_real(x).moveaxis(-1, 0)
        else:
            xr = x
        xr = xr.reshape(-1, self.dim, *x.shape[-self.dim :])
        xp = self._pad(xr)
        y = self._convT(xp, weight=self.kernel, bias=None, padding=2)
        if x.is_complex():
            y = y.reshape(2, *x.shape[: -self.dim - 1], *x.shape[-self.dim :])
            y = torch.view_as_complex(y.moveaxis(0, -1).contiguous())
        else:
            y = y.reshape(*x.shape[: -self.dim - 1], *x.shape[-self.dim :])
        return y
    
  
    def apply_GHG(self, x):
        """Normal operator G^H G x (same shape as x)."""
        if x.is_complex():
            xr = torch.view_as_real(x).moveaxis(-1, 0)
        else:
            xr = x
        xr = xr.reshape(-1, 1, *x.shape[-self.dim :])
        xp = self._pad(xr)
        tmp = self._conv(xp, weight=self.kernel, bias=None, padding=0)
        tmp = self._pad(tmp)
        y = self._convT(tmp, weight=self.kernel, bias=None, padding=2)
        if x.is_complex():
            y = y.reshape(2, *x.shape)
            y = torch.view_as_complex(y.moveaxis(0, -1).contiguous())
        else:
            y = y.reshape(*x.shape)
        return y

    def forward(self, x, direction=1):
        """Apply G (direction > 0), G^H (direction < 0) or G^H G (direction == 0)."""
        if direction>0:
            return self.apply_G(x)
        elif direction<0:
            return self.apply_GH(x)
        else:
            return self.apply_GHG(x)
        
    def loss(self, x, direction=1, weight_map=None):
        """Smoothness loss: mean squared gradient of x, times ``loss_mult``.

        Args:
            x: displacement field (b, 2, h, w).
            direction: which operator to apply, as in ``forward``.
            weight_map: None for a plain loss; 1 to weight the gradients by the normalized
                field magnitude. Other values return None.
        """
        x_mag = torch.sqrt(x[:,0]**2 + x[:,1]**2).unsqueeze(1)
        x_mag = (x_mag - x.min())/(x.max() - x.min())
        if direction>0:
            grad_results = self.apply_G(x)
        elif direction<0:
            grad_results = self.apply_GH(x)
        else:
            grad_results = self.apply_GHG(x)
        
        if weight_map is None:
            l2_norm = torch.mean(grad_results** 2)  
            return (l2_norm).mean() * self.loss_mult
        elif weight_map== 1:
            grad_results = grad_results * (x_mag).unsqueeze(1)
            l2_norm = torch.mean(grad_results** 2)  
            return (l2_norm).mean() * self.loss_mult
                

    @property
    def normGHG(self):
        return self._norm
    
    