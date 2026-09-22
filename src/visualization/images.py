"""Single-image plots (T1 maps) and shared helpers for magnitude images and line profiles."""
import matplotlib.pyplot as plt

from visualization._figure_output import finalize_figure
from visualization.colormaps import cmap_t1


# Shared building blocks

def _magnitude(img):
    """Detached magnitude on CPU."""
    return img.detach().cpu().abs()


def _imshow_shared_scale(axs, images, titles, cmap='gray'):
    """Show magnitude images on the given axes with one shared [0, max] scale."""
    vmax = max(img.max().item() for img in images)
    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img, clim=[0, vmax], cmap=cmap)
        ax.set_title(title)
        ax.axis('off')


def _line_profile(img, index, axis):
    """x-t image along one row or one column of img (t, h, w)."""
    if axis == 'row':
        return img[..., index, :]
    if axis == 'col':
        return img[..., index]
    raise ValueError(f"axis should be 'row' or 'col', not {axis}")


def _show_reference_with_line(ax, reference, index, axis):
    """Show one frame with a red line marking the row/column used for the line profile."""
    ax.imshow(reference, clim=[0, reference.max().item()], cmap='gray')
    if axis == 'row':
        ax.axhline(y=index, color='red', linewidth=2, linestyle='-')
    else:
        ax.axvline(x=index, color='red', linewidth=2, linestyle='-')
    ax.set_title(f'Red line: {axis}={index}')
    ax.axis('off')


# T1 maps

def plot_t1_map(p, save_plots=False, filepath=None, return_fig=False, clim=[0,2],
                title='p', run=None, it=0):
    """Plot the T1 map (last parameter of p, shaped (params, 1, 1, 1, h, w)) with the T1 colormap.

    Args:
        p: stacked parameter maps.
        save_plots, filepath: save the figure to filepath.
        return_fig: return the figure instead of closing it.
        clim: colour range (in the units of p, seconds by default).
        title: figure title.
        run, it: wandb run and step to log the figure to.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(p[-1, 0, 0, 0].abs().detach().cpu(), cmap=cmap_t1, clim=clim)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return finalize_figure(fig, filepath if save_plots else None,
                           run=run, wandb_key=f'plots/{filepath}', it=it, return_fig=return_fig)
