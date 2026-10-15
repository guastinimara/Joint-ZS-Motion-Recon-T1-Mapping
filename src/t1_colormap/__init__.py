"""T1 colormap (lipari) used in the figures.

Sources:
    Crameri F. Scientific colour maps. Zenodo. https://doi.org/10.5281/zenodo.1243862
    Fuderer M, et al. Color-map recommendation for MR relaxometry maps.
    Magn Reson Med. 2025;93(2):490-506. https://doi.org/10.1002/mrm.30290
"""
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

CMAP_T1_PATH = Path(__file__).with_name('cmap_t1_lipari.csv')
cmap_t1 = ListedColormap(np.loadtxt(CMAP_T1_PATH))
