# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
One generation run, start to finish, with no disk I/O.

run_generation packs the pores (with a loose porosity pre-screen), grows the carbon and
post-processes the result into a RunResult. All writing, resuming and parallelism live in
batch.py, so the same function serves tests, notebooks and the cluster batch.
"""

from __future__ import annotations

import time

import numpy as np

from . import post_process as pp
from .config import GenerationConfig
from .generators import RandomGenerator, RingGenerator
from .potential_classes import (
    HarmonicPotential,
    LJPoresPotential,
    TersoffPotential,
)
from .results import RunResult
from .structures import CarbonStructure, PoreStructure


_ATOMS_POTENTIAL_MAP = {
    'harmonic': HarmonicPotential,
    'tersoff': TersoffPotential,
}
_PORES_POTENTIAL_MAP = {
    'lj': LJPoresPotential,
}


def _build_potentials(cfg: GenerationConfig):
    """
    The atoms and pore potentials selected in cfg, with its parameter dicts applied.

    Parameters
    ----------
    cfg : GenerationConfig
        The run configuration.

    Returns
    -------
    tuple
        (atoms potential, pore potential); (None, None) for harmonic + lj with no parameter
        dicts, so RingGenerator builds its own defaults, exactly as when built directly.

    Raises
    ------
    ValueError
        On an unknown potential type.
    """
    using_defaults = (
        cfg.atoms_potential_type == 'harmonic'
        and cfg.pores_potential_type == 'lj'
        and cfg.atoms_potential_params is None
        and cfg.pores_potential_params is None
    )
    if using_defaults:
        return None, None

    atoms_cls = _ATOMS_POTENTIAL_MAP.get(cfg.atoms_potential_type)
    if atoms_cls is None:
        raise ValueError(
            f"Unknown atoms_potential_type {cfg.atoms_potential_type!r}; "
            f"expected one of {sorted(_ATOMS_POTENTIAL_MAP)}."
        )
    pores_cls = _PORES_POTENTIAL_MAP.get(cfg.pores_potential_type)
    if pores_cls is None:
        raise ValueError(
            f"Unknown pores_potential_type {cfg.pores_potential_type!r}; "
            f"expected one of {sorted(_PORES_POTENTIAL_MAP)}."
        )

    atoms_pot = atoms_cls(Box=cfg.box)
    if cfg.atoms_potential_params is not None:
        atoms_pot.set_potential_parameters(**cfg.atoms_potential_params)

    pores_pot = pores_cls(Box=cfg.box)
    if cfg.pores_potential_params is not None:
        pores_pot.set_potential_parameters(**cfg.pores_potential_params)

    return atoms_pot, pores_pot


def _build_pore_kwargs(cfg: GenerationConfig) -> dict:
    """
    Keyword arguments of PoreStructure.generate for the PSD in cfg: the tabulated psd_file
    (with psd_kind) when set, otherwise the Gaussian (pore_mu, pore_sigma).

    Parameters
    ----------
    cfg : GenerationConfig
        The run configuration.

    Returns
    -------
    dict
        {'psd', 'psd_kind'} or {'mu', 'sigma'}.
    """
    if cfg.psd_file is not None:
        # np.load reads a .npy with shape (N, 2): column 0 = diameter [Å],
        # column 1 = density (PDF).  utilities/digitise_psd.py
        # produces these files from a WebPlotDigitizer CSV export.
        psd_table = np.load(cfg.psd_file)
        return dict(psd=psd_table, psd_kind=cfg.psd_kind)
    return dict(mu=cfg.pore_mu, sigma=cfg.pore_sigma)

def _generate_pores_with_prescreen(cfg: GenerationConfig) -> tuple[PoreStructure, dict]:
    """
    Pack pores until the porosity is within phi_tolerance of phi, at most max_pore_attempts
    times, keeping the best packing if none is.

    The tolerance is loose on purpose: growth consumes part of the pore volume, so the
    porosity of the result drifts from the pore-only value anyway, and a tight pre-screen
    would only rerun without matching the target. It filters failed packings.

    Parameters
    ----------
    cfg : GenerationConfig
        The run configuration.

    Returns
    -------
    pore_struct : PoreStructure
        The accepted (or best) packing.
    history : dict
        'attempts_used' (int), 'phi_history' (list of float, the porosity of each attempt),
        'accepted_within_tolerance' (bool).

    Raises
    ------
    ValueError
        If max_pore_attempts < 1.
    """
    if cfg.max_pore_attempts < 1:
        raise ValueError(f"max_pore_attempts must be >= 1, got {cfg.max_pore_attempts}.")
    pore_kwargs = _build_pore_kwargs(cfg)

    phi_history: list[float] = []
    best_pore: PoreStructure | None = None
    best_score: float = float('inf')

    for attempt in range(cfg.max_pore_attempts):
        pore_struct = PoreStructure.generate(
            Box=cfg.box,
            phi=cfg.phi,
            dx=cfg.pore_dx,
            **pore_kwargs,
        )
        phi_err = abs(pore_struct.porosity - cfg.phi) / cfg.phi
        phi_history.append(float(pore_struct.porosity))

        within = phi_err < cfg.phi_tolerance
        if within:
            return pore_struct, {
                'attempts_used': attempt + 1,
                'phi_history': phi_history,
                'accepted_within_tolerance': True,
            }

        # Track the best-so-far so we can fall back to it.
        if phi_err < best_score:
            best_score = phi_err
            best_pore = pore_struct

    if best_pore is None:  # every attempt gave a NaN porosity error
        raise RuntimeError(f"No pore attempt produced a finite porosity error: {phi_history}.")
    return best_pore, {
        'attempts_used': cfg.max_pore_attempts,
        'phi_history': phi_history,
        'accepted_within_tolerance': False,
    }


def _build_generator(cfg: GenerationConfig, pore_struct: PoreStructure):
    """
    The carbon generator selected by cfg.generator_type, built on the pores.

    Parameters
    ----------
    cfg : GenerationConfig
        The run configuration.
    pore_struct : PoreStructure
        The pores to grow around.

    Returns
    -------
    RingGenerator or RandomGenerator

    Raises
    ------
    ValueError
        On an unknown generator_type.
    """
    if cfg.generator_type == 'ring':
        atoms_pot, pores_pot = _build_potentials(cfg)
        ring_kwargs = {}
        if cfg.ring_atoms_per_ring is not None and cfg.ring_probabilities is not None:
            ring_kwargs['atoms_per_ring'] = cfg.ring_atoms_per_ring
            ring_kwargs['probabilities'] = cfg.ring_probabilities
        ring_kwargs['LJ_exclusion_bondnum'] = cfg.lj_exclusion_bondnum
        ring_kwargs['merge_coeff'] = cfg.merge_coeff
        ring_kwargs['reject_coeff'] = cfg.reject_coeff
        ring_kwargs['bond_dist'] = cfg.bond_dist
        ring_kwargs['E_max'] = cfg.E_max
        ring_kwargs['force_defect_pairs'] = cfg.force_defect_pairs
        return RingGenerator(
            pores_structure=pore_struct,
            max_rho=cfg.max_rho,
            atoms_potential=atoms_pot,
            pores_potential=pores_pot,
            **ring_kwargs,
        )
    if cfg.generator_type == 'random':
        return RandomGenerator(
            pores_structure=pore_struct,
            max_rho=cfg.max_rho,
            min_dist=cfg.random_min_dist,
            pore_standoff=cfg.random_pore_standoff,
        )
    raise ValueError(
        f"Unknown generator_type {cfg.generator_type!r}; "
        f"expected 'ring' or 'random'."
    )


def run_generation(config: GenerationConfig) -> tuple[RunResult, CarbonStructure | None, PoreStructure | None]:
    """
    Generate one carbon structure and post-process it.

    Pores (pre-screened), carbon growth (CarbonStructure.generate), then porosity, PSD,
    accessible surface and volumes, RDF (and local RDFs when enabled), ring and coordination
    histograms, and connectivity. Random-generator runs skip the post-processing: random
    structures mean nothing structurally before an MD anneal, so their result carries the
    generation fields only (post-processing scalars NaN, arrays None);
    utilities/analyze_anneal.py measures them after the anneal.

    Parameters
    ----------
    config : GenerationConfig
        The run configuration.

    Returns
    -------
    result : RunResult
        status 'ok', with every field the run computed.
    carbon_struct : CarbonStructure
        The structure.
    pore_struct : PoreStructure
        Its pores.

    Raises
    ------
    Exception
        Any error (an invalid config, a seed ring that cannot be placed) propagates;
        batch._generate_and_analyze_one records it as status='generation_error'.
    """
    cfg = config

    # Wall-time spans the whole pipeline (pre-screen + carbon + post-proc).
    t_start = time.perf_counter()

    # pores, with the cheap pre-screen
    pore_struct, prescreen = _generate_pores_with_prescreen(cfg)


    # carbon: CarbonStructure.generate retries up to cfg.max_carbon_attempts for
    # the density target and returns the best attempt. Replicates at the batch
    # level capture the structural variance.
    generator = _build_generator(cfg, pore_struct)
    carbon_struct = CarbonStructure.generate(
        generator=generator,
        target_density_deviation=cfg.target_density_deviation,
        max_carbon_attempts=cfg.max_carbon_attempts,
        bond_dist=cfg.bond_dist,
        bond_tol=cfg.bond_tol,
    )

    # random structures mean nothing structurally before an MD anneal: no
    # post-processing (utilities/analyze_anneal.py measures them afterwards)
    if cfg.generator_type == 'random':
        return (
            RunResult(
            run_id=cfg.run_id,
            config=cfg.to_dict(),
            box_size=float(cfg.box_size),
            psd_label=cfg.psd_label,
            generator_type=cfg.generator_type,
            status='ok',
            achieved_pore_porosity=float(pore_struct.porosity),
            n_pores=int(pore_struct.coordinates.shape[0]),
            atoms_number=int(carbon_struct.atoms_number),
            achieved_density=float(carbon_struct.density),
            pore_attempts_used=int(prescreen['attempts_used']),
            pore_attempt_phi_history=prescreen['phi_history'],
            accepted_within_tolerance=bool(prescreen['accepted_within_tolerance']),
            wall_time_s=float(time.perf_counter() - t_start),
        ),
        carbon_struct, # successful carbon structure
        pore_struct, # successful pores structure
        )

    # post-processing of the generated structure

    # porosity: fraction of the box a probe center (radius r_N) can reach
    porosity = pp.compute_porosity(
        carbon_struct,
        N_iter=cfg.pp_N_iter_porosity,
        r_atom=cfg.r_C,
        r_probe=cfg.r_N,
        voxel_len=cfg.pp_voxel_len,
        seed=cfg.pp_seed,
    )

    # pore-size distribution
    # Returns (psd, psd_cum, psd_cum_raw, R_count, coord_count).
    # psd is a 2D array [diameter, PDF]; psd_cum its cumulative form.
    psd, psd_cum, _, _, _ = pp.compute_psd(
        carbon_struct,
        N_iter=cfg.pp_N_iter_psd,
        voxel_len=cfg.pp_voxel_len,
        r_C=cfg.r_C,
        r_N=cfg.r_N,
        seed=cfg.pp_seed,
        workers=cfg.pp_workers,
    )

    # accessible surface area and volumes
    # Non-PBC accessibility (flood-fill from box faces): marching-cubes smooth
    # surface, center-accessible volume, and probe-occupiable volume (EDT).
    _ava = pp.compute_AVA(
        carbon_struct,
        voxel_len=cfg.pp_voxel_len,
        r_atom=cfg.r_C,
        r_probe=cfg.r_N,
        workers=cfg.pp_workers,
    )
    vol_tot, surf_tot, occ_vol = _ava.av_center_A3, _ava.asa_smooth_A2, _ava.av_occupiable_A3

    # radial distribution function
    bins, _, rdf = pp.radial_distribution_function(
        carbon_struct,
        size_bin=cfg.pp_rdf_bin,
    )
    rdf_array = np.column_stack([bins, rdf])

    # Local sample-wise RDFs for the worst-quartile objective mode. None when
    # rdf_local_n_samples is unset -> the objective falls back to the global RDF.
    rdf_local = None
    if cfg.rdf_local_n_samples:
        rdf_local = pp._local_rdf_stack(
            carbon_struct,
            size_bin=cfg.pp_rdf_bin,
            n_samples=cfg.rdf_local_n_samples,
            sphere_radius=cfg.rdf_local_sphere_radius,
            seed=cfg.pp_seed,
        )

    # ring-size histogram {size: count}
    ring_hist, _ = pp.count_rings(carbon_struct)

    # coordination histogram {'0 bonds': %, ..., '4 bonds': %, '>4 bonds': %}
    coord_hist = pp.count_coordination_number(carbon_struct)

    # pore connectivity
    # Mean per-pore connectivity from a Voronoi-with-bottleneck analysis.
    connectivity = pp.compute_connectivity(
        carbon_struct,
        r_C=cfg.r_C,
        r_N=cfg.r_N,
    )

    return (
        RunResult(
        run_id=cfg.run_id,
        config=cfg.to_dict(),
        box_size=float(cfg.box_size),
        psd_label=cfg.psd_label,
        generator_type=cfg.generator_type,
        status='ok',
        # Generation outcomes
        achieved_pore_porosity=float(pore_struct.porosity),
        n_pores=int(pore_struct.coordinates.shape[0]),
        atoms_number=int(carbon_struct.atoms_number),
        achieved_density=float(carbon_struct.density),
        # Pre-screen provenance (rich enough to debug bad batches).
        pore_attempts_used=int(prescreen['attempts_used']),
        pore_attempt_phi_history=prescreen['phi_history'],
        accepted_within_tolerance=bool(prescreen['accepted_within_tolerance']),
        # post-processing
        computed_porosity=float(porosity),
        accessible_volume=float(vol_tot),
        occupiable_volume=float(occ_vol),
        accessible_surface=float(surf_tot),
        pore_connectivity=float(connectivity),
        ring_histogram={str(k): int(v) for k, v in ring_hist.items()},
        coordination_histogram=coord_hist,
        # Arrays
        psd=psd,
        psd_cum=psd_cum,
        rdf=rdf_array,
        rdf_local=rdf_local,
        # Timing
        wall_time_s=float(time.perf_counter() - t_start),
        ),
        carbon_struct, # successful carbon structure
        pore_struct, # successful pores structure
    )
