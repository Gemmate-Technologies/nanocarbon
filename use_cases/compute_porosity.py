# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Porosity of an imported structure: the void fraction a nitrogen probe center reaches.

Usage: python use_cases/compute_porosity.py
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
                                            frame_index = -1)
print("Frame from trajectory loaded!")

r_C = 1.71 #van der Waals radius of carbon
r_N = 1.55 #van der Waals radius of nitrogen

# probe-center porosity; r_probe=0 would give the geometric porosity
porosity = pp.compute_porosity(s_loaded,
                               N_iter=1000000,
                               r_atom = r_C,
                               r_probe=r_N)

print(f'Structure porosity:{porosity}')