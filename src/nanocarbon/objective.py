# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Calibration objective: how far one run is from the experimental anchors.

Anchors holds the targets; anchor_cost turns a RunResult into a weighted sum of relative
errors (plus the PSD and RDF distances), infinite when any gated anchor is off by
anchor_max_dev or more, so an off-target candidate cannot steer the swarm.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .results import RunResult
from scipy.integrate import cumulative_trapezoid

@dataclass
class Anchors:
    """
    Calibration targets; an anchor left None is not scored.

    Two porosity conventions exist and are not interchangeable: porosity scores the run's
    computed_porosity (probe-center fraction), while pore_volume_cm3_g scores its
    occupiable_volume (probe-occupiable, PoreBlazer's V_PO, the pore volume adsorption
    measures). The zero-probe geometric porosity is 1 - rho / rho_skeleton by construction, so it
    adds nothing the density does not: prefer pore_volume_cm3_g.

    Parameters
    ----------
    density : float, optional
        Target density, g/cm^3 (label 'rho').
    porosity : float, optional
        Target probe-center porosity (label 'phi').
    asa_m2_g : float, optional
        Target accessible surface area, m^2/g (label 'ASA').
    pore_volume_cm3_g : float, optional
        Target probe-occupiable pore volume, cm^3/g (label 'V_PO').
    psd_file : str, optional
        Target differential PSD, a (K, 2) .npy of diameter (Angstrom) and value (label 'psd').
    psd_frac_below : float, optional
        Diameter cutoff, Angstrom: scores the fraction of pore volume below it, the target
        fraction read from psd_file (label 'psd_low'). Never gates.
    rdf_file : str, optional
        Target g(r), a (n, 2) .npy of r (Angstrom) and g (label 'rdf').
    rdf_rmax : float, optional
        Largest r compared, Angstrom. Default 7.
    rdf_smear : float, optional
        Gaussian width applied to the run's g(r) before comparison, Angstrom (see _smear).
        None (default): off.
    weights : dict, optional
        {label: weight}; a label not listed weighs 1.
    anchor_max_dev : float, optional
        Gate on the relative deviation of the gated anchors. Default 0.9.
    """
    density: float | None = None
    porosity: float | None = None
    asa_m2_g: float | None = None
    pore_volume_cm3_g: float | None = None
    psd_file: str | None = None
    # Diameter (A) below which pore volume is 'micropore'; the target fraction is
    # derived from psd_file, so only the cutoff is specified here.
    psd_frac_below: float | None = None
    rdf_file: str | None = None
    rdf_rmax: float = 7.0
    # Gaussian width (A) applied to the run's g(r) before comparison. None = off.
    rdf_smear: float | None = None
    weights: dict = field(default_factory=lambda: {'rho': 1.0, 'phi': 1.0, 'ASA': 1.0, 'psd': 1.0, 'rdf': 1.0})
    anchor_max_dev: float = 0.9


def _smear(rdf: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-broaden g(r) to mimic thermal motion.

    A constructed structure is rigid at 0 K, so its g(r) is a comb of sharp peaks, while a
    reference measured at finite temperature has each pair distance spread over its RMS
    relative displacement (about 0.085 A for amorphous carbon at room temperature, from the
    measured first-peak FWHM of about 0.2 A). The width is set by physics: do not tune it to
    fit.

    Parameters
    ----------
    rdf : (n, 2) ndarray
        [r (Angstrom), g(r)].
    sigma : float
        Gaussian width, Angstrom.

    Returns
    -------
    (n, 2) ndarray
        [r, broadened g(r)].
    """
    r, g = rdf[:, 0], rdf[:, 1]
    kernel = np.exp(-0.5 * ((r[:, None] - r[None, :]) / sigma) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)  # row-normalized: leaves g -> 1 tails exactly flat
    return np.column_stack([r, kernel @ g])


def _frac_below(psd: np.ndarray, d_star: float) -> float:
    """
    Fraction of the pore volume in pores narrower than d_star.

    Parameters
    ----------
    psd : (K, 2) ndarray
        Differential PSD: diameter (Angstrom), -dF/dd.
    d_star : float
        Diameter cutoff, Angstrom.

    Returns
    -------
    float
        The fraction; nan if the PSD integrates to zero or less.
    """
    d, f = psd[:, 0], np.clip(psd[:, 1].astype(float), 0.0, None)
    total = np.trapezoid(f, d)
    if total <= 0:
        return float('nan')

    grid = np.union1d(d, [d_star])          # insert the cutoff so the split is exact
    f_grid = np.interp(grid, d, f)
    below = grid <= d_star
    return float(np.trapezoid(f_grid[below], grid[below]) / total)


def psd_emd(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    """
    Wasserstein-1 (earth mover's) distance between two differential PSDs, in radius.

    Each PSD is normalized to a probability distribution over the radius (diameter / 2); the
    distance is the integral of |CDF_a - CDF_b| (trapezoid rule, on the union of the two
    grids).

    Parameters
    ----------
    psd_a, psd_b : (K, 2) ndarray
        Diameter (Angstrom), distribution value.

    Returns
    -------
    float
        Distance, Angstrom.
    """
    r_a, d_a = psd_a[:, 0] / 2.0, psd_a[:, 1].astype(float)
    r_b, d_b = psd_b[:, 0] / 2.0, psd_b[:, 1].astype(float)

    r_all = np.union1d(r_a, r_b) # common grid, for non-uniform spacing
    cdf_a = cumulative_trapezoid(d_a, r_a, initial=0.0) # building the CDF
    cdf_a /= cdf_a[-1]
    cdf_b = cumulative_trapezoid(d_b, r_b, initial=0.0)
    cdf_b /= cdf_b[-1]

    # both CDFs on the common grid
    cdf_a = np.interp(r_all, r_a, cdf_a, left=0.0, right=1.0)
    cdf_b = np.interp(r_all, r_b, cdf_b, left=0.0, right=1.0)
    return float(np.trapezoid(np.abs(cdf_a - cdf_b), r_all))

def rdf_distance(rdf_a: np.ndarray, rdf_b: np.ndarray, r_max: float = 7.0,
                 smear: float | None = None) -> float:
    """
    Mean absolute difference between two RDFs, averaged over r.

    g(r) is not a probability distribution, so the RDFs are compared as curves, not with an
    EMD. The comparison runs over the r range both cover, up to r_max.

    Parameters
    ----------
    rdf_a : (n, 2) ndarray
        [r (Angstrom), g(r)] of the run.
    rdf_b : (m, 2) ndarray
        [r, g(r)] of the reference.
    r_max : float, optional
        Largest r compared, Angstrom. Default 7.
    smear : float, optional
        Gaussian width applied to rdf_a only, Angstrom: the 0 K construct lacks the thermal
        broadening the reference already has. None (default): off.

    Returns
    -------
    float
        The r-averaged |g_a - g_b|.
    """
    if smear is not None:
        rdf_a = _smear(rdf_a, smear)

    r_a, g_a = rdf_a[:, 0], rdf_a[:, 1]
    r_b, g_b = rdf_b[:, 0], rdf_b[:, 1]

    r_all = np.union1d(r_a, r_b)

    # compare only where both curves exist: no extrapolation
    idx = np.logical_and(
        r_all >= np.max([np.min(r_a), np.min(r_b), 0.0]),
        r_all <= np.min([np.max(r_a), np.max(r_b), r_max])
    )
    r_all = r_all[idx]

    g_a = np.interp(r_all, r_a, g_a)
    g_b = np.interp(r_all, r_b, g_b)
    integral_average = np.trapezoid(np.abs(g_a - g_b), r_all) / (r_all[-1] - r_all[0])
    return float(integral_average)

def anchor_cost(run: RunResult, anchors: Anchors, return_components: bool = False) -> float | tuple[float, dict]:
    """
    Weighted cost of one run against the anchors.

    Each set anchor adds its weight times its error: the relative error for density,
    porosity, ASA and V_PO; the PSD EMD divided by the target's mean radius; the volume-fraction
    difference below psd_frac_below; the RDF distance. With rdf_local in the run, the RDF term
    is the mean of the worst quarter of the per-sphere distances, so the least-ordered pockets
    drive it instead of the box average. ASA converts from Angstrom^2 to m^2/g as
    accessible_surface * 1e4 / (achieved_density * box_size^3); V_PO converts to cm^3/g the
    same way.

    The total is +inf when the run failed, when a relative anchor deviates by anchor_max_dev
    or more, or when the PSD or RDF needed is missing or invalid; psd_frac_below never gates.
    The components are kept even when the total is inf, so the anchor that gated stays
    visible in the log.

    Parameters
    ----------
    run : RunResult
        The run.
    anchors : Anchors
        The targets.
    return_components : bool, optional
        Also return the per-anchor terms. Default False.

    Returns
    -------
    float, or (float, dict)
        The total cost, and with return_components {label: weighted term}.

    Raises
    ------
    ValueError
        If psd_frac_below is set without psd_file.
    """
    if run.status != 'ok':
        return (float('inf'), {}) if return_components else float('inf')

    w = anchors.weights
    max_dev = anchors.anchor_max_dev
    components: dict[str, float] = {}
    gated = False  # any active anchor over max_dev (or invalid) -> total is inf, but each component stays visible

    if anchors.density is not None:
        rel = abs(run.achieved_density - anchors.density) / anchors.density
        components['rho'] = w.get('rho', 1.0) * rel
        gated = gated or rel >= max_dev

    if anchors.porosity is not None:
        rel = abs(run.computed_porosity - anchors.porosity) / anchors.porosity
        components['phi'] = w.get('phi', 1.0) * rel
        gated = gated or rel >= max_dev

    if anchors.asa_m2_g is not None:
        box_size = run.config['box_size']
        asa = run.accessible_surface * 1e4 / (run.achieved_density * box_size ** 3)
        rel = abs(asa - anchors.asa_m2_g) / anchors.asa_m2_g
        components['ASA'] = w.get('ASA', 1.0) * rel
        gated = gated or rel >= max_dev

    if anchors.pore_volume_cm3_g is not None:
        # occupiable_volume (A^3) -> cm3/g: V*1e-24 / (rho * box^3 * 1e-24).
        box_size = run.config['box_size']
        v_po = run.occupiable_volume / (run.achieved_density * box_size ** 3)
        rel = abs(v_po - anchors.pore_volume_cm3_g) / anchors.pore_volume_cm3_g
        components['V_PO'] = w.get('V_PO', 1.0) * rel
        gated = gated or rel >= max_dev

    if anchors.psd_frac_below is not None:
        # The EMD is a CDF integral, so a too-small mode cancels against a too-fat
        # tail and it reports a match the PSD shape does not deserve. This term is
        # the micropore volume fraction, which does not cancel.
        if anchors.psd_file is None:
            raise ValueError('psd_frac_below needs psd_file: its target fraction is read from it')
        if run.psd is None:
            components['psd_low'] = float('inf')
            gated = True
        else:
            f_run = _frac_below(run.psd, anchors.psd_frac_below)
            f_ref = _frac_below(np.load(anchors.psd_file), anchors.psd_frac_below)
            components['psd_low'] = w.get('psd_low', 1.0) * abs(f_run - f_ref)
            gated = gated or not np.isfinite(components['psd_low'])

    if anchors.psd_file is not None:
        # invalid PSD (missing, or non-positive integral) -> inf component.
        if run.psd is None or np.trapezoid(run.psd[:, 1], run.psd[:, 0] / 2.0) <= 0:
            components['psd'] = float('inf')
            gated = True
        else:
            target_psd = np.load(anchors.psd_file)
            emd = psd_emd(run.psd, target_psd)
            # e_r_star is a density-weighted mean radius of the target PSD; it rescales
            # the emd (in A) to a unitless distance.
            r_star = target_psd[:, 0] / 2.0
            d_star = target_psd[:, 1].astype(float)
            e_r_star = float(np.trapezoid(d_star * r_star, r_star) / np.trapezoid(d_star, r_star))
            components['psd'] = w.get('psd', 1.0) * (emd / e_r_star if e_r_star > 0 else float('inf'))
            gated = gated or not np.isfinite(components['psd'])

    if anchors.rdf_file is not None:
        # Load the target lazily, only when an RDF exists to compare against:
        # a None RDF must gate to inf without needing a valid target file.
        if run.rdf_local is not None:
            # Local sample-wise mode (run enabled rdf_local_n_samples). Score the
            # worst quartile of per-sphere distances to the bulk reference g(r):
            # a sphere matching bulk order scores low, a disordered pocket high,
            # so the least-organized regions drive the cost instead of the
            # box-average (which averages them away). At 20 samples k == 5.
            target_rdf = np.load(anchors.rdf_file)
            d = np.array([rdf_distance(c, target_rdf, r_max=anchors.rdf_rmax, smear=anchors.rdf_smear)
                          for c in run.rdf_local])
            k = max(1, round(0.25 * d.size))
            components['rdf'] = w.get('rdf', 1.0) * float(np.mean(np.sort(d)[-k:]))
        elif run.rdf is not None:
            target_rdf = np.load(anchors.rdf_file)
            components['rdf'] = w.get('rdf', 1.0) * rdf_distance(run.rdf, target_rdf,
                                                                 r_max=anchors.rdf_rmax,
                                                                 smear=anchors.rdf_smear)
        else:  # neither RDF available -> invalid.
            components['rdf'] = float('inf')
            gated = True

    total = float('inf') if gated else float(sum(components.values()))
    return (total, components) if return_components else total
