"""Figures comparing results across the iterations of the joint reconstruction."""
import os

import matplotlib.pyplot as plt

from visualization._figure_output import finalize_figure
from visualization.colormaps import cmap_t1
from visualization.images import _line_profile, _magnitude


def plot_t1_over_iterations(all_p, filename, cmap=cmap_t1, clim=[0,2000]):
    """T1 map (in ms) at every iteration, side by side."""
    all_t1 = all_p.squeeze()[:,-1]
    num_imgs = all_t1.shape[0]
    fig, axes = plt.subplots(1, num_imgs, figsize=(4*num_imgs, 4))
    fig.suptitle('T1 over iterations')
    for i in range(num_imgs):
        ax = axes[i] if num_imgs > 1 else axes
        im = ax.imshow(all_t1[i]*1e3, clim=clim, cmap=cmap)
        ax.set_title(f'it {i}')
        ax.axis('off')
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02)
    finalize_figure(fig, filename, bbox_inches='tight')


def plot_line_profile_over_iterations(x_moving, all_x, filename, index, axis='row'):
    """One row or column of x followed over time: x_moving first, then every iteration.

    x_moving is (t, h, w) and all_x is (iterations, t, h, w) once squeezed; each is
    normalised by its own maximum.
    """
    x_moving = x_moving.squeeze().abs()
    x_moving = x_moving/x_moving.max()

    all_x = all_x.squeeze().abs()
    all_x = all_x/all_x.max()

    lines = [x_moving] + list(all_x)
    titles = ['x moving'] + [f'it {i}' for i in range(1, len(lines))]

    fig, axes = plt.subplots(1, len(lines), figsize=(4*len(lines), 4))
    fig.suptitle(f'x over iterations - {axis} {index}')
    for ax, img, title in zip(axes, lines, titles):
        im = ax.imshow(_line_profile(img, index, axis), cmap='gray', vmin=0, vmax=0.6)
        ax.set_title(title)
        ax.axis('off')
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02)
    finalize_figure(fig, filename, bbox_inches='tight')


def plot_registration_over_iterations(x_moving, all_x_registered, t, save_plots=False, dir_fig=None):
    """x_moving at time t next to its registered version at every iteration."""
    num_iters = all_x_registered.shape[0]
    fig, axs = plt.subplots(1, num_iters+1, figsize=(5*(num_iters+1), 5))

    images = [x_moving[t]] + [all_x_registered[i][t] for i in range(num_iters)]
    titles = [f'x moving, t={t}'] + [f'x registered iter {i}, t={t}' for i in range(num_iters)]
    for ax, img, title in zip(axs, images, titles):
        img = _magnitude(img.squeeze())
        ax.imshow(img, clim=[0, img.max().item()], cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    filepath = os.path.join(dir_fig, f'x_registered_all_iters_t{t}.png') if save_plots and dir_fig is not None else None
    finalize_figure(fig, filepath)


def plot_correlation_heatmaps_over_iterations(t1_parameter_estimator, all_forward_motion, all_inverse_motion, figure_dir):
    """Temporal correlation maps of the forward, inverse and mean motion fields at every iteration."""
    heat_maps = {'Forward Motion': [], 'Inverse Motion': [], 'Mean Motion': []}
    for i in range(all_forward_motion.shape[0]):
        heat_map_fw = t1_parameter_estimator._compute_correlation_map(all_forward_motion[i], show_heatmap=False)
        heat_map_inv = t1_parameter_estimator._compute_correlation_map(all_inverse_motion[i], show_heatmap=False)
        heat_maps['Forward Motion'].append(heat_map_fw)
        heat_maps['Inverse Motion'].append(heat_map_inv)
        heat_maps['Mean Motion'].append((heat_map_fw + heat_map_inv) / 2)

    num_heatmaps = all_forward_motion.shape[0]
    fig, axes = plt.subplots(3, num_heatmaps, figsize=(4*num_heatmaps, 8))

    for row, (label, row_maps) in enumerate(heat_maps.items()):
        for idx, heat_map in enumerate(row_maps):
            im = axes[row, idx].imshow(heat_map.cpu().numpy() if hasattr(heat_map, 'cpu') else heat_map, cmap='viridis')
            axes[row, idx].set_title(f'Iteration {idx}')
            axes[row, idx].axis('off')
            plt.colorbar(im, ax=axes[row, idx])
        axes[row, 0].text(-0.3, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=12, va='center', ha='right', weight='bold')

    heatmap_filepath = f'{figure_dir}/correlation_heatmaps_all_iterations.png'
    os.makedirs(os.path.dirname(heatmap_filepath), exist_ok=True)
    plt.tight_layout()
    finalize_figure(fig, heatmap_filepath, dpi=150, bbox_inches='tight')
