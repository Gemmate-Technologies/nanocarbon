# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Accessible surface area and volumes of an imported structure (compute_AVA).

Usage: python use_cases/compute_AVA.py
"""

import time
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

start = time.time()
# accessible surface area, probe-center and probe-occupiable volumes
r_C = 1.71 #van der Waals radius of Carbon (structure)
r_N = 1.55 #van der Waals radius of Nitrogen (probe)
voxel_len = 0.5

# int(), as compute_AVA: a remainder thinner than a voxel is not sampled
nx = int(s_loaded.Box[0] / voxel_len)
ny = int(s_loaded.Box[1] / voxel_len)
nz = int(s_loaded.Box[2] / voxel_len)
n_vox = nx * ny * nz
print(f"Voxel grid: {nx}x{ny}x{nz} = {n_vox:,} voxels — dense bool grid: {n_vox / 1024**2:.1f} MB")

ava = pp.compute_AVA(s_loaded, voxel_len, r_C, r_N)
asa, av_center, av_occupiable = ava.asa_smooth_A2, ava.av_center_A3, ava.av_occupiable_A3
print(f'Accessible surface area (marching cubes, smooth): {asa} A^2')
print(f'Center-accessible volume: {av_center} A^3')
print(f'Probe-occupiable volume: {av_occupiable} A^3')
end = time.time()
print(end-start)