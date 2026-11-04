# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Small batch with Gaussian PSDs; needs no input data.

Runs in a few minutes on a laptop through GenerationConfig -> run_batch -> BatchResults.
Rerunning resumes where the previous run stopped. The variables mirror
utilities/batch_config.py, and run_batch is what the `generate` verb calls.

Usage: python use_cases/batch_gaussian_psd.py
"""
from pathlib import Path

from nanocarbon import GenerationConfig, run_batch

OUT_ROOT = str(Path(__file__).parent / 'output' / 'batch_gaussian')
N_PER_CONDITION = 2
N_WORKERS = 4

# Cartesian sweep axes. Dict-valued entries group co-varying parameters;
# scalar entries are applied directly as GenerationConfig overrides.
CONDITIONS = {
    'box_size': [20.0, 30.0],
    'psd': [
        {'psd_label': 'small_pores', 'pore_mu': 6.0,  'pore_sigma': 1.5},
        {'psd_label': 'large_pores', 'pore_mu': 14.0, 'pore_sigma': 2.0},
    ],
    'phi': [0.2, 0.3],
}

BASE_CFG = GenerationConfig(
    max_rho=1.0,
    pore_dx=0.5,
    pp_N_iter_porosity=20_000, pp_N_iter_psd=1_000,
    phi_tolerance=0.20, max_pore_attempts=5,
)

if __name__ == '__main__':  # required for multiprocessing on Windows
    batch = run_batch(
        conditions=CONDITIONS,
        n_per_condition=N_PER_CONDITION,
        out_root=OUT_ROOT,
        base_cfg=BASE_CFG,
        n_workers=N_WORKERS,
    )
    assert batch is not None, "Small batch run failed, no results returned."

    ring_pre = batch.filter(generator_type='ring')
    print(f'[done] {len(ring_pre.ok_runs)} ok, '
          f'{len(ring_pre.failed_runs)} failed under {OUT_ROOT}')
