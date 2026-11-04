# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Mean pore-network connectivity (links per pore) of an imported structure.

Usage: python use_cases/compute_connectivity.py
"""

from pathlib import Path

from nanocarbon import CarbonStructure
from nanocarbon import post_process as pp
from nanocarbon import io_functions as io

data_dir = Path(__file__).parent / "data"
traj_file = data_dir / "harmonic_trajectory.lammpstrj"

# the last frame of the trajectory
dict_lammps = io.import_from_LAMMPS(traj_file)
s_loaded = CarbonStructure.from_LAMMPS_file(dict_lammps,
                                            frame_index=-1)
print('Frame from trajectory loaded!')

r_C = 1.71   # carbon van der Waals radius (Å)
r_N = 1.55   # probe radius (Å)

k_avg = pp.compute_connectivity(s_loaded, r_C=r_C, r_N=r_N)
print(f'Mean pore connectivity (links per pore): {k_avg:.3f}')
