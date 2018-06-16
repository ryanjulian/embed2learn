from matplotlib.cm import get_cmap
import numpy as np


def colormap(k, name="jet"):
    cmap = get_cmap(name)
    rgb = cmap(np.linspace(0, 1, num=k), bytes=True)[::, :3]
    return [tuple(c) for c in rgb]
