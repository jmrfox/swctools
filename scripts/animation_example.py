"""
Animation example script for swctools using pyvista.
"""

import logging
from pathlib import Path
import numpy as np
from swctools import FrustaSet, animate_frusta_timeseries

logging.basicConfig(level=logging.DEBUG)

swc_filepath = Path("data/swc/TS2_s50.swc")
frusta = FrustaSet.from_swc_file(swc_filepath)

n_frusta = frusta.n_frusta
T = 10
dt = 0.1
time_domain = np.linspace(0, T, int(T / dt) + 1)
freqs = np.random.uniform(3, 30, n_frusta) / T
V = np.zeros((len(time_domain), n_frusta))
for i, t in enumerate(time_domain):
    V[i, :] = np.sin(freqs * t)

plotter = animate_frusta_timeseries(
    frusta,
    time_domain=time_domain,
    amplitudes=V,
)
