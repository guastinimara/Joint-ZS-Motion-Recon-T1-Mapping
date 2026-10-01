"""Local output paths.

Copy this file to ``configs/local_paths.py`` (gitignored) and fill in your own values:

    cp configs/local_paths_example.py configs/local_paths.py

The input data (simulated XCAT k-space) is downloaded from Zenodo by scripts/main.py,
so no input data paths are needed.
"""
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")

PATHS = {
    # Root folder for all results (one subfolder per day)
    'output_dir': f'/path/to/results/{today}/',
}
