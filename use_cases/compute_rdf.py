# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Radial distribution function of an imported structure, plotted against OVITO's for the
same frame.

Usage: python use_cases/compute_rdf.py
"""

from pathlib import Path
import numpy as np

from nanocarbon import CarbonStructure
from nanocarbon import post_process as pp
from nanocarbon import io_functions as io

import matplotlib.pyplot as plt

data_dir = Path(__file__).parent / "data"
rdf_file = data_dir / "tersoff_edip_rdf_ovito.txt"
traj_file = data_dir / "tersoff_edip_trajectory.lammpstrj"

# reference RDF from OVITO: two columns, r [A] and g(r)
rdf_ovito = np.loadtxt(rdf_file)

# the last frame of the trajectory
dict_lammps = io.import_from_LAMMPS(traj_file)
s_loaded = CarbonStructure.from_LAMMPS_file(dict_lammps, 
                                            frame_index = -1) # loading last frame
print("Frame from trajectory loaded!")

# RDF
bins, _, rdf = pp.radial_distribution_function(s_loaded,
                                               size_bin = 0.5)
print("RDF computed!")


# comparison
plt.figure(figsize=(8, 5))
plt.plot(rdf_ovito[:, 0], rdf_ovito[:, 1], label = 'OVITO')
plt.plot(bins, rdf, label = 'nanocarbon')
plt.title('Radial Distribution Function')
plt.xlabel('r (A)')
plt.ylabel('g(r)')
plt.xlim([0, 10])
plt.grid(True)
plt.legend()
plt.show()