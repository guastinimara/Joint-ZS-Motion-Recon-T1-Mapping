"""Colormaps for the figures."""
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

# T1 colormap used throughout the paper figures.
CMAP_T1_PATH = Path(__file__).with_name('cmap_t1_lipari.csv')
cmap_t1 = ListedColormap(np.loadtxt(CMAP_T1_PATH))
