# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Load frames of a LAMMPS trajectory as CarbonStructures (bonds inferred by distance).

Usage: python use_cases/load_from_lammps.py
"""

from pathlib import Path
from nanocarbon import CarbonStructure
from nanocarbon import io_functions as io

data_dir = Path(__file__).parent / "data"
traj_file = data_dir / "harmonic_trajectory.lammpstrj"

dict_lammps = io.import_from_LAMMPS(traj_file)

# last frame
s_loaded_last = CarbonStructure.from_LAMMPS_file(dict_lammps, 
                                                frame_index=-1)
# first frame
s_loaded_first = CarbonStructure.from_LAMMPS_file(dict_lammps, 
                                                frame_index=0)

print(f'Number of atoms:{s_loaded_last.atoms_number}')
print(f'Coordinate of the first 5 atoms: {s_loaded_last.coordinates[:5]}')
print(f'First 5 atoms bonds: {s_loaded_last.bonds[:5]}')

print(f'Number of atoms:{s_loaded_first.atoms_number}')
print(f'Coordinate of the first 5 atoms: {s_loaded_first.coordinates[:5]}')
print(f'First 5 atoms bonds: {s_loaded_first.bonds[:5]}')