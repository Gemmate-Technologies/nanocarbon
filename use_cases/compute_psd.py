# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Pore-size distribution of an imported structure, plotted against PoreBlazer's for the
same frame.

Usage: python use_cases/compute_psd.py
"""

from pathlib import Path
import numpy as np

from nanocarbon import CarbonStructure
from nanocarbon import post_process as pp
from nanocarbon import io_functions as io

import matplotlib.pyplot as plt

data_dir = Path(__file__).parent / "data"
traj_file = data_dir / "harmonic_trajectory.lammpstrj"
psd_file = data_dir / "harmonic_psd_poreblazer.txt"


# the last frame of the trajectory
dict_lammps = io.import_from_LAMMPS(traj_file)
s_loaded = CarbonStructure.from_LAMMPS_file(dict_lammps, 
                                            frame_index = -1) # loading last frame
print("Frame from trajectory loaded!")


# reference PSD from PoreBlazer
psd_PoreBlazer = np.loadtxt(psd_file)

# PSD (geometric definition, as PoreBlazer)
psd, _, _, _, _ = pp.compute_psd(s_loaded, 
                                 N_iter = 1000,
                                 r_C = 1.71, 
                                 r_N = 1.55)
print("PSD computed!")

plt.plot(psd_PoreBlazer[:,0], psd_PoreBlazer[:,1], label = 'PoreBlazer')
plt.plot(psd[:,0], psd[:,1], label = 'This work')
plt.xlabel('Pore diameter (A)')
plt.ylabel('PSD')
plt.legend()
plt.show()