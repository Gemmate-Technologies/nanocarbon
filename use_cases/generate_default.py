# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Generate one structure with the default potentials.

Builds a GenerationConfig, runs the whole pipeline with run_generation and writes the
scalar results with a copy of the config.

Usage: python use_cases/generate_default.py
"""

import json
import os
from pathlib import Path

from nanocarbon import GenerationConfig, run_generation


# Configuration

cfg = GenerationConfig(
    box_size=50.0,          # Å, cubic box
    phi=0.2,                # target pore volume fraction
    pore_mu=10.0,           # mean pore radius (Å), Gaussian PSD
    pore_sigma=1.0,         # std dev of pore radius (Å)
    pore_dx=0.5,            # voxel spacing for pore occupancy tracking (0.5 Å resolves sub-nm pores; keeps the grid memory-light)
    max_rho=1.2,            # target carbon density (g/cm³)
    generator_type='ring',  # ring-by-ring growth strategy
    out_dir=str(Path(__file__).parent / 'output' / 'default'),
)


# Generation

result, _, _ = run_generation(cfg)

print(f"Status          : {result.status}")
print(f"Pore porosity   : {result.achieved_pore_porosity:.4f}  (target {cfg.phi})")
print(f"Carbon density  : {result.achieved_density:.4f} g/cm³")
print(f"Computed porosity: {result.computed_porosity:.4f}")
print(f"Accessible volume: {result.accessible_volume:.1f} Å³")
print(f"Wall time       : {result.wall_time_s:.1f} s")

# the scalar results, with a copy of the config
os.makedirs(cfg.out_dir, exist_ok=True)
with open(os.path.join(cfg.out_dir, 'result.json'), 'w') as f:
    json.dump(result.to_scalar_dict(), f, indent=2)
cfg.to_json(os.path.join(cfg.out_dir, 'config.json'))

print(f"\nResults written to {cfg.out_dir}/")
