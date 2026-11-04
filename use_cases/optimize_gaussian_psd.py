# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
PSO calibration demo on a 30 A box with a Gaussian PSD.

Searches the lj pore-wall parameters A (epsilon) and B (sigma) to match a target density and
porosity. With 4 particles x 1 replicate x 4 evaluations (initial swarm + 3 iterations) =
16 generations, it runs in minutes on a laptop.

Usage: python use_cases/optimize_gaussian_psd.py
   or: python -m nanocarbon optimize use_cases/optimize_gaussian_psd.py
"""
from pathlib import Path
from nanocarbon import GenerationConfig
from nanocarbon.objective import Anchors
from nanocarbon.calibrate import run_calibration

BASE_CFG = GenerationConfig(
    box_size=30.0,
    phi=0.25,
    pore_mu=5.0,
    pore_sigma=1.0,
    pore_dx=0.5,
    max_rho=0.9,
    max_pore_attempts=10,
    max_carbon_attempts=2,
    pp_N_iter_porosity=5_000,
    pp_N_iter_psd=200,
    generator_type='ring',
    pores_potential_type='lj',
)

# Loose tolerances: any run within 90% of every target gets a finite cost.
ANCHORS = Anchors(
    density=0.9,
    porosity=0.25,
    anchor_max_dev=0.9,
)

# 2D search over the lj pore wall: A is epsilon (well depth), B is sigma
# (the length scale that sets the carbon-pore standoff). Above B ~ 2.8 A the
# wall leaves no room to seed carbon: those candidates only cost time.
PARAM_BOUNDS = {
    ('pores_potential_params', 'A'): (1.0, 200.0),
    ('pores_potential_params', 'B'): (0.5, 2.5),
}

SURROGATE_PARAMS = {'warmup': 8, # finite evaluations before the first surrogate fit
                    'degree': 2, # polynomial degree
                    'alpha': 1e-3, # regularization strength
                    'l1':0.5, 
                    }

PSO_CONFIG = {
    'swarm_size': 4,
    'n_iter': 3,
    'w': 0.7,
    'c1': 1.5,
    'c2': 1.5,
    'v_max_frac': 0.2,
}
N_REPLICATES = 1
N_REQUIRED = 1
# next to this file whatever the working directory, so both launch lines write in the same
# place; run_calibration creates the folder
OUT_ROOT = str(Path(__file__).parent / 'output' / 'calibration_gaussian')
N_WORKERS = 4

if __name__ == '__main__':
    result = run_calibration(
        base_cfg=BASE_CFG,
        anchors=ANCHORS,
        param_bounds=PARAM_BOUNDS,
        pso_config=PSO_CONFIG,
        n_replicates=N_REPLICATES,
        n_required=N_REQUIRED,
        out_root=OUT_ROOT,
        surrogate_params= SURROGATE_PARAMS,
        n_workers=N_WORKERS,
    )
    print(f'\nBest params found:')
    for (field, key), (lo, hi) in PARAM_BOUNDS.items():
        val = result['gbest_params'].get(field, {}).get(key, float('nan'))
        print(f'  {field}[{key}] = {val:.4f}  (search range [{lo}, {hi}])')
    print(f'\nBest cost: {result["gbest_cost"]:.4f}')
    print(f'\nCost convergence (gbest per iteration):')
    for k, entry in enumerate(result['history']):
        print(f'  iter {k:2d}: {entry["gbest_cost"]:.4f}')
