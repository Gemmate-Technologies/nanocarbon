# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Calibration config template.

Usage: python -m nanocarbon optimize utilities/calibrate_config.py   (from the repository root)

Inverse design of the generation parameters against a target pore-size distribution: the
surrogate-assisted PSO searches five knobs so that the generated structures reproduce the
target density and PSD. The method settings (swarm, replicas, surrogate) are those we use
for full calibrations; the target is the CDC sub-nm PSD shipped in use_cases/data.

The LJ atom pair is NOT a knob: sigma and eps stay at literature carbon values, because a
free sigma lets the optimizer fake porosity as slits between sheets instead of placing pores.
Every target value must be measured with the SAME post-process settings as BASE_CFG
(pp_voxel_len, r_C, r_N): otherwise the cost scores a settings mismatch, not a structure.
"""
from nanocarbon import GenerationConfig
from nanocarbon.objective import Anchors

PSD_REF = 'use_cases/data/cdc_subnm_psd.npy'     # differential volume PSD, Nx2: diameter [A], dV/dD

BASE_CFG = GenerationConfig(
    # key structural parameters
    box_size=100.0,
    psd_file=PSD_REF,
    psd_label='cdc_subnm',
    psd_kind='volume',      # differential-volume PSD -> divide by D^3 at sampling
    generator_type='ring',
    max_rho=0.912,
    atoms_potential_type='harmonic',
    lj_exclusion_bondnum=15,
    pores_potential_type='lj',
    pore_dx=0.5,
    force_defect_pairs=True,
    # atoms potential -- LJ pinned at literature carbon values, deliberately not a knob
    atoms_potential_params={
        'k_dihedral': 5,
        'nonBondedAttraction': True,
        'nonBondedAttraction_A': 100.0,
        'shortRangeRepulsion': False,
        'LJ_sigma': 3.407,
        'LJ_eps': 0.0684,
    },
    E_max=1e3,
    # post-process parameters -- the target values must be measured with these
    pp_N_iter_porosity=100_000,
    pp_N_iter_psd=2_500,
    pp_voxel_len=0.5,
    pp_rdf_bin=0.25,
    r_C=1.71,
    r_N=1.55,
    pp_seed=42,
    # loose generation tolerances: a tight one turns an imperfect candidate into a rejected one
    target_density_deviation=0.90,
    phi_tolerance=0.90,
    max_pore_attempts=5,
    max_carbon_attempts=5,
    # local RDF sampling, used only when an rdf_file anchor is set below
    rdf_local_n_samples=20,
    rdf_local_sphere_radius=7.0,
)

ANCHORS = Anchors(
    density=0.912,
    psd_file=PSD_REF,
    # Optional anchors, each with a target measured at the BASE_CFG post-process settings:
    # asa_m2_g=...,          accessible surface area of a reference structure
    # psd_frac_below=...,    diameter [A] isolating the smallest mode; target read from psd_file
    # rdf_file=..., rdf_rmax=7.0, rdf_smear=0.085,   reference g(r), Nx2: r [A], g
    weights={'rho': 1.0, 'psd': 1.0},
    anchor_max_dev=0.9,
)

# D = len(PARAM_BOUNDS): add or remove entries to change the search space.
PARAM_BOUNDS = {
    ('atoms_potential_params', 'k_dihedral'): (0.01, 25.0),
    ('pores_potential_params', 'A'): (0.01, 200.0),
    ('pores_potential_params', 'B'): (0.3, 2.5),     # below 0.3 the wall stops binding
    'phi': (0.01, 0.50),
    'merge_coeff': (0.3, 0.8),
}

PSO_CONFIG = {
    'swarm_size': 16,
    'n_iter': 25,
    'w': 0.7,
    'c1': 1.5,
    'c2': 1.5,
    'v_max_frac': 0.2,
}

SURROGATE_PARAMS = {'warmup': 64,          # successful runs needed before the surrogate starts
                    'degree': 2,           # polynomial parameter expansion
                    'alpha': 1e-3,         # regularization strength
                    'max_iter': 100_000,   # ElasticNet iterations; raise this, not alpha, to converge
                    'l1': 0.5,             # L1/L2 ratio
                    'n_split': 5,          # cross-validation folds (diagnostic only)
                    'variance_eps': 1e-3,  # floor for the variance-based sample weights
                    'n_starts': 15,        # restarts when minimizing the surrogate
                    'optim_tol': 1e-5,
                    }

N_REPLICATES = 7            # generations per candidate, scored by their mean
N_REQUIRED = 5              # successful replicates a candidate needs to enter the swarm
OUT_ROOT = 'calibration_output'
N_WORKERS = 16              # keep peak memory per worker x N_WORKERS within the node
