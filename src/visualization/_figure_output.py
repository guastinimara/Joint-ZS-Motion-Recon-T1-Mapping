"""Common output handling for figures (show, save or log to wandb)."""
import matplotlib.pyplot as plt


def finalize_figure(fig, filepath=None, run=None, wandb_key=None, it=0, return_fig=False, **savefig_kwargs):
    """Send a finished figure to wandb, to disk, or to the screen.

    - run given: log it to wandb under `wandb_key` at step `it`, then close it.
    - filepath given: save it (extra kwargs go to `fig.savefig`).
    - return_fig: return the figure, left open.
    - otherwise the figure is shown if it was not saved, then closed.
    """
    if run is not None:
        import wandb
        run.log({wandb_key: wandb.Image(fig)}, step=it)
        plt.close(fig)
        return None

    if filepath is not None:
        fig.savefig(filepath, **savefig_kwargs)
    if return_fig:
        return fig
    if filepath is None:
        plt.show()
    plt.close(fig)
    return None
