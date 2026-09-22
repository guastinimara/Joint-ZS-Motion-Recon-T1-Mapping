#%%
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'src')]

from visualization.ResultsVisualizer import ResultsVisualizer

from configs.runtime_parameters import (
    EXPERIMENT_FILENAMES,
    RUN_SETTINGS,
    RECONSTRUCTION_PARAMS,
)


def main() -> None:
    """Regenerate the figures for already saved results of every patient in EXPERIMENT_FILENAMES."""
    for patient_name in EXPERIMENT_FILENAMES:
        visualizer = ResultsVisualizer.from_patient(
            patient_name, RUN_SETTINGS, RECONSTRUCTION_PARAMS,
            plot_fields=True, plot_slices=True, plot_registrations=True,
        )
        if visualizer is None:
            continue  # No valid RR threshold: skip to the next patient
        visualizer.run()


if __name__ == '__main__':
    main()
#%%
