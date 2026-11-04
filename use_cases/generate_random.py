# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
One carbon structure from the random generator, written for a LAMMPS quench.

Random structures mean nothing structurally until an MD anneal, so this writes only the
LAMMPS data file, in the bond-free 'atomic' style.

Usage: python use_cases/generate_random.py
"""

from pathlib import Path

from nanocarbon import PoreStructure, CarbonStructure, RandomGenerator

BOX = [30.0, 30.0, 30.0]   # Å
PHI = 0.2                  # target pore volume fraction
PORE_MU = 8.0              # mean pore radius (Å)
PORE_SIGMA = 1.0
MAX_RHO = 1.0              # target carbon density (g/cm³)
MIN_DIST = 1.5             # minimum inter-atom distance (Å)
PORE_STANDOFF = 0.75       # extra exclusion margin around each pore (Å)

p = PoreStructure.generate(Box=BOX, phi=PHI, dx=0.5,
                           mu=PORE_MU, sigma=PORE_SIGMA)
gen = RandomGenerator(pores_structure=p, max_rho=MAX_RHO,
                      min_dist=MIN_DIST, pore_standoff=PORE_STANDOFF)
s = CarbonStructure.generate(generator=gen, target_density_deviation=0.05)

print(f'Atoms          : {s.atoms_number}')
print(f'Carbon density : {s.density:.4f} g/cm³')
print(f'Pore porosity  : {p.porosity:.4f}')

# Bond-free `atomic`-style LAMMPS data file (random structures have no bonds).
out_dir = Path(__file__).parent / 'output' / 'random'
out_dir.mkdir(parents=True, exist_ok=True)
s.print_LAMMPS(name=str(out_dir / 'random_structure'))
print('LAMMPS data file written: random_structure.data')
