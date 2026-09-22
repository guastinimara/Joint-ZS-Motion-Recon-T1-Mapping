"""Analysis and plots of 2D motion fields (Jacobian determinant, divergence, deformed grids, arrows)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import torch
import torch.nn as nn
from einops import rearrange

from cardiac_motion.NetworkMotionFieldsWithInverse import NetworkMotionFieldsWithInverse
from operators.MotionOperator import MotionOperator
from visualization._figure_output import finalize_figure

SpatialTransformer = NetworkMotionFieldsWithInverse.SpatialTransformer


class MotionFieldAnalyzer:
    """Quantities and plots for 2D motion (displacement) fields shaped ch t h w:
    Jacobian determinant, divergence, deformed grid and arrow plots."""

    def __init__(self,example=''):
        """example: optional name of the example/subject (not used in the plots)."""
        self.cmap = 'seismic_r'
        self.example = example

        self.pad1 = nn.ReplicationPad2d((0, 1, 0, 0))
        self.pad2 = nn.ReplicationPad2d((0, 0, 0, 1))

    def invert_axis(self,disp): # ch t h w
        """Swap the two field components (y, x) -> (x, y) and add a batch dimension."""
        disp_inv = torch.stack([disp[1, ...], disp[0, ...]], dim=0)
        return disp_inv.unsqueeze(0) # b ch t h w

    def get_determinant(self, disp, return_matrix=False): # ch t h w
        """Jacobian determinant per time point (t, h, w)."""
        return MotionOperator.jacobian_determinant(self.invert_axis(disp).squeeze(0), return_matrix=return_matrix)

    def plot_determinant(self, disp, t, negative_det=False, margins=None, suptitle=None, save_filename=None, return_fig=False):
        """Plot the Jacobian determinant at time t (colour range [-margins, margins], default 4).

        negative_det: also show where the determinant is <= 0 (folding).
        """
        det = self.get_determinant(disp)
        if not margins:
            margins = 4.0 #torch.maximum(abs(det[t]).min(), abs(det[t]).max())

        norm = TwoSlopeNorm(vmin=-margins, vcenter=0, vmax=margins)
        fig = plt.figure(figsize=(8, 5) if negative_det else (8, 8))

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(det[t], cmap=self.cmap, norm=norm)
        ax1.set_title("Determinant J")
        fig.colorbar(im1, ax=ax1, shrink=0.6)
        ax1.axis('off')

        if negative_det:
            ax2 = fig.add_subplot(1, 2, 2)
            im2 = ax2.imshow(det[t] <= 0, cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
            ax2.set_title("Determinant J <= 0")
            fig.colorbar(im2, ax=ax2, shrink=0.6)
            ax2.axis('off')

        plt.tight_layout()
        if suptitle:
            fig.suptitle(suptitle)

        return finalize_figure(fig, save_filename, return_fig=return_fig, dpi=100, bbox_inches='tight')

    def get_divergence(self, vector_field):
        """Divergence per time point (t, h, w) of a field (ch, t, h, w)."""
        vector_field = self.invert_axis(vector_field)
        vector_field = rearrange(vector_field, "b ch t h w -> (b t) ch h w")

        Vx = vector_field[:, 0, :, :]
        Vy = vector_field[:, 1, :, :]

        dVx_dx = Vx[:, :, 1:] - Vx[:, :, :-1]
        dVy_dy = Vy[:, 1:, :] - Vy[:, :-1, :]

        dVx_dx = torch.nn.functional.pad(dVx_dx, (0, 1, 0, 0))
        dVy_dy = torch.nn.functional.pad(dVy_dy, (0, 0, 0, 1))
        divergence = dVx_dx + dVy_dy

        return divergence

    def plot_divergence(self, disp, t, margins=None, and_det=False, suptitle=None, save_filename=None):
        """Plot the divergence at time t, optionally next to the determinant (and_det)."""
        div = self.get_divergence(disp)

        if not margins:
            margins = 4.0 #torch.maximum(abs(div[t]).min(), abs(div[t]).max())

        norm = TwoSlopeNorm(vmin=-margins, vcenter=0, vmax=margins)

        fig = plt.figure(figsize=(8, 5) if and_det else (5, 5))

        plt.subplot(1, 2, 1)
        im1 = plt.imshow(div[t], cmap=self.cmap, norm=norm)
        plt.title("Divergence")
        plt.colorbar(im1, shrink=0.6)
        plt.axis('off')

        if and_det:
            det = self.get_determinant(disp)
            plt.subplot(1, 2, 2)
            im2 = plt.imshow(det[t], cmap=self.cmap, norm=norm)
            plt.title("Determinant")
            plt.colorbar(im2, shrink=0.6)
            plt.axis('off')

        plt.tight_layout()
        if suptitle:
            plt.suptitle(suptitle)

        finalize_figure(fig, save_filename, dpi=100, bbox_inches='tight')

    def get_deformation(self, warp_disp):
        """Absolute deformation (identity grid + displacement), shaped (1, ch, t, h, w)."""
        warp_disp = warp_disp.unsqueeze(0)
        b, ch, t, h, w = warp_disp.shape
        transformer = SpatialTransformer([h,w])
        grid = transformer.get_grid().unsqueeze(2)
        deformation = warp_disp + grid
        return deformation

    def plot_grid(self, disp, t, step_plot=2,  title='', linewidth=0.3, suptitle=None, save_filename=None, return_fig=False):
        """Plot the deformed grid at time t."""
        deformation = rearrange(self.get_deformation(disp).squeeze(), "ch t h w -> t h w ch")
        warp = deformation[t]

        h,w,ch = warp.shape

        def auxiliary_plot_grid(x, y, ax=None, axis_off=True, **kwargs):
            ax = ax or plt.gca()
            segs1 = np.stack((x, y), axis=2)
            segs2 = segs1.transpose(1, 0, 2)
            if axis_off:
                ax.axis('off')
            ax.add_collection(LineCollection(segs1, **kwargs))
            ax.add_collection(LineCollection(segs2, **kwargs))
            ax.autoscale()

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        singular_grid = warp

        u = singular_grid[..., 1]
        v = singular_grid[..., 0]

        nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
        u[np.where(nanIdx)] = 0
        v[np.where(nanIdx)] = 0

        aspect_ratio = float(u.shape[0]) / u.shape[1]
        ax.set_aspect(aspect_ratio)
        plt.subplots_adjust(left=0.0, top=1, right=1, bottom=0., hspace=0,
                            wspace=0)
        alpha = [0]*step_plot
        alpha[0] = 1
        auxiliary_plot_grid(u, v, ax=ax, color='black', linewidth=linewidth,
                            alpha=alpha)
        ax.set_title(title + f'{t}', fontsize=12)
        ax.axis('off')

        ax.set_ylim(w,0)
        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

        plt.tight_layout()

        return finalize_figure(fig, save_filename, return_fig=return_fig, dpi=100, bbox_inches='tight')

    def plot_vf(self,vector_field, t, step_plot=4, scale=1,  title=None, suptitle=None, save_filename=None):
        """Arrow (quiver) plot of the field at time t, one arrow every step_plot pixels."""
        fig, ax = plt.subplots(figsize=(5, 5))
        vector_field = rearrange(vector_field, 'ch t h w -> t h w ch')[t]
        h, w, _ = vector_field.shape
        x = np.arange(0, w, step_plot)
        y = np.arange(0, h, step_plot)
        x, y = np.meshgrid(x, y)
        ax.set_aspect('equal')

        ax.quiver(
            x,
            y,
            vector_field[::step_plot, ::step_plot, 0],
            vector_field[::step_plot, ::step_plot, 1],
            angles='xy', scale_units='xy', scale=scale, color='black',
            headwidth=2
        )

        ax.set_ylim(w,0)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=12)
        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()

        finalize_figure(fig, save_filename, dpi=100, bbox_inches='tight')
