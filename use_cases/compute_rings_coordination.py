# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Ring-size and coordination histograms of an imported structure.

Usage: python use_cases/compute_rings_coordination.py
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

# Ring statistics.
ring_hist, _ = pp.count_rings(s_loaded)
print('\nRing size distribution:')
for size in sorted(ring_hist):
    print(f'  {size}-rings: {ring_hist[size]}')

# Coordination histogram.
coord_hist = pp.count_coordination_number(s_loaded)
print('\nCoordination distribution (% atoms):')
for k, pct in coord_hist.items():
    print(f'  {k}: {pct:.1f} %')
