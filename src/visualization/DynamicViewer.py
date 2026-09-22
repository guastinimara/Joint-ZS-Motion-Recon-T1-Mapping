"""Viewer for one dynamic image series: frames, comparisons, line profiles and videos."""
import math
import os

import matplotlib.pyplot as plt
from matplotlib import animation

from visualization._figure_output import finalize_figure
from visualization.images import _imshow_shared_scale, _line_profile, _magnitude, _show_reference_with_line


class DynamicViewer:
    """Frames, comparisons, line profiles and videos of one dynamic image series.

    x is shaped (t, ..., h, w), with only singleton dimensions in between, e.g. (t, 1, 1, h, w).
    It may be complex; everything is shown as magnitude. `name` is used in titles.

    Usage:
        viewer = DynamicViewer(mask.apply(x_moving), name='x moving')

        viewer.plot_frames([0, 5, 10])                            # grid of frames
        viewer.compare([x_0_ref, x_final], t=5,                   # side by side, shared grey scale
                       titles=['x_0_ref', 'x final'])
        viewer.plot_line_profile(75, axis='row')                  # one row ('row') or column ('col') over time
        viewer.video()                                            # animation to display in a notebook
        viewer.video('results/x_moving.mp4', fps=15)              # save the video instead
        viewer.frame(5), viewer.frames()                          # magnitude (h, w) / (t, h, w)

    The plotting methods show the figure by default; pass `filepath` to save it, `return_fig=True`
    to get it back, or `run`/`it` to log it to wandb. Images in `compare` can be dynamics (frame t
    is used), other DynamicViewers, or single (h, w) images.
    """

    def __init__(self, x, name='x'):
        self.x = x
        self.name = name

    def __len__(self):
        return self.x.shape[0]

    def frame(self, t):
        """Magnitude of frame t, shaped (h, w)."""
        return _magnitude(self.x[t]).reshape(self.x.shape[-2:])

    def frames(self):
        """Magnitude of every frame, shaped (t, h, w)."""
        return _magnitude(self.x).reshape(len(self), *self.x.shape[-2:])

    def plot_frames(self, times, filepath=None, return_fig=False):
        """Grid of the frames at the given times, each on its own grey scale."""
        num_cols = min(len(times), 4)
        num_rows = math.ceil(len(times) / num_cols)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        axs = axs.flatten() if num_rows > 1 or num_cols > 1 else [axs]

        for ax, t in zip(axs, times):
            frame = self.frame(t)
            ax.imshow(frame, clim=[0, frame.max().item()], cmap='gray')
            ax.set_title(f'{self.name}, t={t}')
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        return finalize_figure(fig, filepath, return_fig=return_fig)

    def compare(self, others, t, titles=None, suptitle=None, filepath=None,
                run=None, wandb_key=None, it=0, return_fig=False):
        """Frame t next to other images, all on one shared [0, max] grey scale.

        Each entry of `others` is a dynamic (frame t is taken), a DynamicViewer, or a single (h, w) image.
        """
        images = [self.frame(t)] + [self._frame_of(other, t) for other in others]
        titles = [self.name] + list(titles if titles is not None else [f'image {i+1}' for i in range(len(others))])

        fig, axs = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        _imshow_shared_scale(axs, images, titles)
        fig.suptitle(suptitle if suptitle is not None else f't={t}')
        plt.tight_layout()
        return finalize_figure(fig, filepath, run=run, wandb_key=wandb_key, it=it, return_fig=return_fig)

    def plot_line_profile(self, index, axis='row', reference_t=5, filepath=None,
                          run=None, it=0, return_fig=False):
        """Frame `reference_t` with the chosen row/column marked, next to that line over time."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        _show_reference_with_line(axs[0], self.frame(reference_t), index, axis)

        line = _line_profile(self.frames(), index, axis)
        axs[1].imshow(line, clim=[0, line.max().item()], cmap='gray')
        axs[1].set_title(f'{axis} {index} over time')
        axs[1].axis('off')

        fig.suptitle(self.name, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return finalize_figure(fig, filepath, run=run, wandb_key=f'x_slice/x_{axis}_figure_{index}',
                               it=it, return_fig=return_fig)

    def video(self, filepath=None, fps=15, clim=None):
        """Play the frames in time.

        With `filepath` the video is saved there (e.g. .mp4) and nothing is returned; without it an
        animation is returned for display in a notebook. `clim` defaults to [0, max over all frames].
        """
        frames = self.frames()
        clim = clim if clim is not None else [0, frames.max().item()]

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(frames[0], clim=clim, cmap='gray')
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        title = ax.set_title(f'{self.name}, t=0')

        def show_frame(t):
            im.set_data(frames[t])
            title.set_text(f'{self.name}, t={t}')
            return im, title

        anim = animation.FuncAnimation(fig, show_frame, frames=len(frames), interval=1000 / fps)

        try:
            if filepath is None:
                from IPython.display import HTML
                return HTML(anim.to_jshtml(fps=fps))

            import imageio_ffmpeg  # ffmpeg binary bundled with imageio, no system install needed
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with plt.rc_context({'animation.ffmpeg_path': imageio_ffmpeg.get_ffmpeg_exe()}):
                anim.save(filepath, writer=animation.FFMpegWriter(fps=fps))
            print(f"Video saved to {filepath}")
        finally:
            plt.close(fig)

    @staticmethod
    def _frame_of(image, t):
        """Frame t of a dynamic or DynamicViewer, or the magnitude of a single (h, w) image."""
        if isinstance(image, DynamicViewer):
            return image.frame(t)
        if image.ndim == 2:
            return _magnitude(image)
        return DynamicViewer(image).frame(t)
