# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Batch config template.

Usage: python -m nanocarbon generate utilities/batch_config.py      (from the repository root)
   or: sbatch utilities/run_parallel.slurm                          (on a SLURM cluster)

Generates every combination of CONDITIONS, N_PER_CONDITION replicas each, in parallel, and
writes one results_{run_id}.json (+ .npy arrays, LAMMPS data and pore files) per run under
OUT_ROOT. Rerunning the same file skips the runs already on disk. This example sweeps box
size, pore fraction and the two pore-wall parameters against the CDC sub-nm PSD shipped in
use_cases/data.
"""
from nanocarbon import GenerationConfig

N_PER_CONDITION = 3     # replicates per condition, for error bars
N_WORKERS = 16          # None -> os.cpu_count(); keep peak memory per worker x N_WORKERS within the node
OUT_ROOT = 'batch_output'

CONDITIONS = {
    'box_size': [50.0, 100.0],
    'psd': [
        {
            'psd_label': 'cdc_subnm',
            'psd_file': 'use_cases/data/cdc_subnm_psd.npy',
            'max_rho': 0.912,
        },
    ],
    'phi': [0.35, 0.45],
    # A dict whose values are lists expands into every combination of those values.
    'pores_potential_params': {
        'A': [1.0, 5.0, 50.0],
        'B': [1.0, 1.5],
        'e1': 12,
        'e2': 6,
    },
}

BASE_CFG = GenerationConfig(
    # A tabular PSD (the 'psd' axis above) is read as differential volume.
    psd_kind='volume',
    atoms_potential_type='harmonic',
    atoms_potential_params={
        'k_dihedral': 5,
        'nonBondedAttraction': True,
        'nonBondedAttraction_A': 100.0,
        'shortRangeRepulsion': False,
    },
    # Tersoff instead: atoms_potential_type='tersoff' and drop atoms_potential_params.
    # The pore-wall A then reads in eV rather than kcal/mol.
    lj_exclusion_bondnum=15,
    pores_potential_type='lj',
    target_density_deviation=0.05,
    pp_N_iter_porosity=50_000,
    pp_N_iter_psd=1_000,
    # ASA/AV is voxel-based. pp_voxel_len controls resolution and run time:
    # 0.5 A is a good balance; drop to 0.3 for a finer pass on selected structures.
    pp_voxel_len=0.5,
    pp_rdf_bin=0.25,
    r_C=1.71, r_N=1.55,
    phi_tolerance=0.1, max_pore_attempts=5, max_carbon_attempts=10,
)
