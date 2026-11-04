# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Generate one structure with custom potential parameters.

Same shape as generate_default.py: build a GenerationConfig, run run_generation,
write the results and the config. Here the harmonic carbon-carbon potential and the
Lennard-Jones pore wall are set through atoms_potential_params and pores_potential_params;
everything else keeps its default.

Usage: python use_cases/generate_custom_potentials.py
"""

import json
import os
from pathlib import Path

from nanocarbon import GenerationConfig, run_generation


# Configuration

cfg = GenerationConfig(
    box_size=50.0,
    phi=0.2,
    pore_mu=10.0,
    pore_sigma=1.0,
    pore_dx=0.5,
    max_rho=0.9,
    generator_type='ring',

    # Atom-atom potential: harmonic (default). Override a few knobs.
    atoms_potential_type='harmonic',
    atoms_potential_params={
        'k_dihedral': 5,
        'nonBondedAttraction': True,
        'nonBondedAttraction_A': 100.0,
        'shortRangeRepulsion': True,
        'shortRangeRepulsion_A': 100.0,
        'shortRangeRepulsion_B': 0.5,
    },

    # Atom-pore potential: lj is the only one (and the default); pin its
    # parameters here rather than taking the class defaults.
    pores_potential_type='lj',
    pores_potential_params={'A': 10, 'B': 1.0, 'e1': 12, 'e2': 6},

    out_dir=str(Path(__file__).parent / 'output' / 'custom_potentials'),
)


# Generation

result, structure, pores = run_generation(cfg)

print(f"Status           : {result.status}")
print(f"Pore porosity    : {result.achieved_pore_porosity:.4f}  (target {cfg.phi})")
print(f"Carbon density   : {result.achieved_density:.4f} g/cm³")
print(f"Computed porosity: {result.computed_porosity:.4f}")
print(f"Accessible volume: {result.accessible_volume:.1f} Å³")
print(f"Wall time        : {result.wall_time_s:.1f} s")

os.makedirs(cfg.out_dir, exist_ok=True)
with open(os.path.join(cfg.out_dir, 'result.json'), 'w') as f:
    json.dump(result.to_scalar_dict(), f, indent=2)
cfg.to_json(os.path.join(cfg.out_dir, 'config.json'))
if structure is not None and pores is not None:
    structure.print_LAMMPS(name=os.path.join(cfg.out_dir,'custom_potential_structure'))
    with open(os.path.join(cfg.out_dir, 'pores.json'), 'w') as f:
        json.dump(pores.to_dict(), f, indent=2)

print(f"\nResults written to {cfg.out_dir}/")
