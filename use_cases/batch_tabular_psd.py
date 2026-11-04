# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Small batch with tabulated reference PSDs.

Uses use_cases/data/cdc_subnm_psd.npy and cmk3_meso_psd.npy, digitized volume PSDs
(utilities/digitise_psd.py). The variables mirror utilities/batch_config.py, and run_batch
is what the `generate` verb calls.

Usage: python use_cases/batch_tabular_psd.py
"""
import os
from pathlib import Path

from nanocarbon import GenerationConfig, run_batch

OUT_ROOT = str(Path(__file__).parent / 'output' / 'batch_tabular')
N_PER_CONDITION = 2
N_WORKERS = 6

CDC_PSD = str(Path(__file__).parent / 'data' / 'cdc_subnm_psd.npy')
CMK3_PSD = str(Path(__file__).parent / 'data' / 'cmk3_meso_psd.npy')
for path in (CDC_PSD, CMK3_PSD):
    assert os.path.exists(path), (
        f'Required PSD file not found: {path}. Digitise with '
        'utilities/digitise_psd.py.'
    )

CONDITIONS = {
    'box_size': [20.0, 30.0],
    'psd': [
        {'psd_label': 'cdc_subnm', 'psd_file': CDC_PSD},
        {'psd_label': 'cmk3_meso', 'psd_file': CMK3_PSD},
    ],
    'max_rho': [0.8, 1.3],
}

BASE_CFG = GenerationConfig(
    phi=0.2,
    psd_kind='volume',   # the 'psd' axis files are differential-volume PSDs
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

    assert batch is not None, "Batch generation failed." 

    ring_pre = batch.filter(generator_type='ring')
    print(f'[done] {len(ring_pre.ok_runs)} ok, '
          f'{len(ring_pre.failed_runs)} failed under {OUT_ROOT}')
