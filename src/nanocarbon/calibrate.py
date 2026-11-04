# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
PSO calibration of generation parameters against experimental anchors.

run_calibration runs the loop: every iteration evaluates the swarm (plus, once warm, the
surrogate's proposal) as one batch with replicas, scores each candidate with anchor_cost,
updates the swarm and checkpoints it, so a stopped calibration resumes. main is the
`python -m nanocarbon optimize <config_file>` entry point.
"""
from __future__ import annotations
import importlib.util
import os
import json
from typing import Any, cast

import numpy as np

from .config import GenerationConfig
from .batch import run_batch, _build_configs
from .objective import Anchors, anchor_cost
from .pso import ParticleSwarm
from .results import BatchResults
from .surrogate import fit_surrogate, propose


def _vec_to_condition(
    vec: np.ndarray,
    param_bounds: dict,
    base_cfg: GenerationConfig,
    particle_label: str,
) -> dict:
    """
    The CONDITIONS entry of one particle: its position written into config fields.

    Parameters
    ----------
    vec : (D,) ndarray
        Particle position, in the order of param_bounds.
    param_bounds : dict
        {field: (lo, hi)} or {(field, key): (lo, hi)}; a tuple key sets one entry of a dict
        field (e.g. a potential parameter), merged into base_cfg's dict so the keys not swept
        keep their base values.
    base_cfg : GenerationConfig
        The base configuration.
    particle_label : str
        Label of the particle, used in its directory name.

    Returns
    -------
    dict
        {'particle_label': ..., field: value, ...}.
    """


    cond: dict = {'particle_label': particle_label}

    for i, key in enumerate(param_bounds.keys()):
        if isinstance(key, str):
            cond[key] = float(vec[i])
        elif isinstance(key, tuple):
            field_name, param_key = key
            # first key of this field: start from a copy of base_cfg's dict (empty if None),
            # so base_cfg itself is never mutated
            if cond.get(field_name, None) is None:
                cond[field_name] = dict(getattr(base_cfg, field_name) or {}) 
            cond[field_name][param_key] = float(vec[i])

    return cond

def _cost_by_particle(batch_runs: BatchResults, 
                      stems: list[str], 
                      anchors: Anchors,
                      n_required: int,
                      ) -> tuple[np.ndarray, list]:
    """
    Cost of each particle: the mean over its replicas with a finite cost.

    Runs are grouped by config hash (the replicas of one particle share it). Inf replicas are
    left out of the mean, so one failed replica does not sink a good candidate; a particle with
    fewer than n_required finite replicas costs inf. Prints the mean per-anchor split over
    all replicas, inf ones included, which names the anchor that blocks a blind swarm.

    Parameters
    ----------
    batch_runs : BatchResults
        The iteration's batch.
    stems : list of str
        Config hash of each particle, in particle order.
    anchors : Anchors
        The targets.
    n_required : int
        Finite replicas needed for a finite cost.

    Returns
    -------
    costs : (P,) ndarray
        Particle costs.
    extra : list of dict or None
        Per particle: 'anchors' (mean term per label), 'cost_var', 'n_survivors'; None for an
        inf particle.
    """
    P = len(stems)
    groups = {}
    for r in batch_runs.runs:
        if r.status != 'ok':
            continue
        groups.setdefault(r.config_hash, []).append(r) # groups = {hash1: [r0, r1, ]; hash2: [r0, r1] }

    costs = np.empty(P)
    extra: list[dict[str, Any] | None] = []
    all_comps: list[dict] = []  # every replica's per-anchor split, for the iteration cost-split log
    for p in range(P):
        pairs = [anchor_cost(r, anchors, return_components = True) for r in groups.get(stems[p], [])]
        pairs = cast(list[tuple[float, dict]], pairs)
        all_comps.extend(comp for _, comp in pairs if comp)  # comp is populated even for gated (inf) runs
        surv = [(tot_cost, comp) for tot_cost, comp in pairs if np.isfinite(tot_cost)] #list with replicas with non-inf cost
        if len(surv)< n_required:
            costs[p] = np.inf
            extra.append(None)
            continue
        totals  = [t for t, _ in surv]
        comps   = [c for _, c in surv]
        costs[p]  = float(np.mean(totals))

        # per-particle data for the log and the surrogate
        extra.append({
            'anchors': {k: float(np.mean([c[k] for c in comps])) for k in comps[0].keys()},
            'cost_var': float(np.var(totals)),
            'n_survivors': len(surv)
        })

    # mean contribution per anchor over ALL replicas (incl. gated/inf) -> shows which
    # anchor drives the cost even when every particle is inf.
    if all_comps:
        split = {k: float(np.mean([c[k] for c in all_comps])) for k in all_comps[0].keys()}
        print('[PSO] cost split (mean/anchor over %d replicas): ' % len(all_comps)
              + '  '.join('%s=%.3g' % (k, v) for k, v in split.items()))

    return costs, extra

def _evaluate_iter(
    X: np.ndarray,
    param_bounds: dict,
    base_cfg: GenerationConfig,
    anchors: Anchors,
    n_replicates: int,
    n_required: int,
    iter_out: str,
    n_workers: int | None,
) -> tuple[np.ndarray, list]:
    """
    Evaluate a set of positions: one batch of P x n_replicates runs, scored per particle.

    Replicas are matched to their particle through the config hash, not a serial number, so
    the assignment survives reruns and resumes.

    Parameters
    ----------
    X : (P, D) ndarray
        Positions to evaluate.
    param_bounds : dict
        As in run_calibration.
    base_cfg : GenerationConfig
        The base configuration.
    anchors : Anchors
        The targets.
    n_replicates : int
        Runs per particle.
    n_required : int
        Finite replicas needed for a finite cost.
    iter_out : str
        Output directory of this iteration.
    n_workers : int or None
        Worker processes of the batch.

    Returns
    -------
    costs : (P,) ndarray
        Particle costs (all inf if the batch could not be consolidated).
    extra : list
        Per-particle data (_cost_by_particle).
    """

    P = X.shape[0]
    particle_conditions = [
        _vec_to_condition(X[p], param_bounds, base_cfg, f'p{p:03d}')
        for p in range(P)
    ]

    # 'particle' is a placeholder axis: each entry's keys must be GenerationConfig
    # fields, others are dropped.
    batch = run_batch(
        conditions={'particle': particle_conditions},
        n_per_condition=n_replicates,
        out_root=iter_out,
        base_cfg=base_cfg,
        n_workers=n_workers,
    )

    if batch is None:
        return np.full(P, np.inf), [None]*P

    # the config hash of each particle
    probe_cfgs = _build_configs({'particle': particle_conditions}, 1, iter_out, base_cfg)
    stems = [c.config_hash for c in probe_cfgs] # len P, in particle order

    cost, extra = _cost_by_particle(batch_runs= batch, stems= stems, anchors= anchors, n_required = n_required)
    return cost, extra

def _surrogate_candidate(history: list[dict],
                        anchor_labels: list[str],
                        bounds: list[tuple],
                        surrogate_params: dict[str, Any] | None,
                        warmup: int,
                        ) -> np.ndarray | None:
    """
    The surrogate's proposal for this iteration.

    Parameters
    ----------
    history : list of dict
        ParticleSwarm.history.
    anchor_labels : list of str
        Active anchor labels (_anchor_labels).
    bounds : list of (float, float)
        Search bounds.
    surrogate_params : dict or None
        Keys among warmup, degree (at least 2), alpha, l1, max_iter, n_split, variance_eps,
        n_starts, optim_tol; None turns the surrogate off.
    warmup : int
        Finite evaluations needed before the first fit.

    Returns
    -------
    (D,) ndarray or None
        The proposal; None when the surrogate is off, not warm yet, or found nothing.

    Raises
    ------
    KeyError
        On an unknown surrogate_params key.
    """
    if surrogate_params is None:
        return None
    args = set(["warmup", "degree", "alpha", "l1", "max_iter", "n_split", "variance_eps", "n_starts", "optim_tol"])
    if any([k not in args for k in surrogate_params]):
        raise KeyError(f"Surrogate parameter not admissible. Valid entries: {args}")

    params, cost, _, cost_var = training_matrix(history, anchor_labels)

    if len(cost) < warmup:
        return None

    model, _ = fit_surrogate(parameters=params,
                              cost = cost,
                              cost_var = cost_var,
                              degree = max(surrogate_params.get('degree', 2), 2),
                              alpha = surrogate_params.get('alpha', 1e-3),
                              l1 = surrogate_params.get('l1', 0.5),
                              max_iter = surrogate_params.get('max_iter', 5000),
                              n_split = surrogate_params.get('n_split', 5),
                              eps = surrogate_params.get('variance_eps', 1e-3)
                              )

    candidate = propose(model = model, 
                        bounds = bounds,
                        n_starts = surrogate_params.get('n_starts', 20),
                        tol=surrogate_params.get('optim_tol', 1e-5),
                        )
    return candidate

def _anchor_labels(anchors: Anchors) -> list:
    """
    Labels of the anchors set in anchors, in a fixed order (density 'rho', porosity 'phi',
    asa_m2_g 'ASA', pore_volume_cm3_g 'V_PO', psd_frac_below 'psd_low', psd_file 'psd',
    rdf_file 'rdf'): the keys of the cost split and the columns of the surrogate's breakdown.

    Parameters
    ----------
    anchors : Anchors
        The targets.

    Returns
    -------
    list of str
    """
    _ANCHOR_LABELS = {
        'density':  'rho',
        'porosity': 'phi',
        'asa_m2_g': 'ASA',
        'pore_volume_cm3_g': 'V_PO',
        'psd_frac_below': 'psd_low',
        'psd_file': 'psd',
        'rdf_file': 'rdf',
    }

    labels_list = [_ANCHOR_LABELS[field] for field in _ANCHOR_LABELS if getattr(anchors, field) is not None ]

    return labels_list

def _gbest_hash(gbest_vec: np.ndarray, param_bounds: dict,
                base_cfg: GenerationConfig, out_root: str) -> str:
    """
    Config hash of the gbest position: the stem of its carbon_{hash}_r* files.

    The hash ignores out_dir and run_id, so it finds the best structure without tracking which
    iteration produced it.

    Parameters
    ----------
    gbest_vec : (D,) ndarray
        The gbest position.
    param_bounds : dict
        As in run_calibration.
    base_cfg : GenerationConfig
        The base configuration.
    out_root : str
        Calibration output directory.

    Returns
    -------
    str
    """
    cond = _vec_to_condition(gbest_vec, param_bounds, base_cfg, 'gbest')
    return _build_configs({'particle': [cond]}, 1, out_root, base_cfg)[0].config_hash

def _log_row(rec: dict, i: int, gbest_hash: str | None = None) -> dict:
    """
    One line of the iteration log.

    anchor_std is the spread of an anchor's term across the swarm; replica_std the
    run-to-run noise of one candidate. An anchor with anchor_std below replica_std cannot steer
    the search (a constant offset, not a gradient): read the two together.

    Parameters
    ----------
    rec : dict
        A ParticleSwarm.history entry.
    i : int
        Iteration number.
    gbest_hash : str, optional
        Config hash of the gbest (_gbest_hash).

    Returns
    -------
    dict
        iter, gbest_cost, gbest_hash, cost mean / median / std over the finite costs,
        inf_fraction, cum_evals, and anchor_mean / anchor_std / replica_std when any particle
        survived.
    """
    costs  = np.asarray(rec['costs'])
    finite = costs[np.isfinite(costs)]
    row = {
        'iter': i,
        'gbest_cost': rec['gbest_cost'],
        'gbest_hash': gbest_hash,
        'cost_mean': float(np.mean(finite)) if finite.size else float('inf'),
        'cost_median': float(np.median(finite)) if finite.size else float('inf'),
        'cost_std': float(np.std(finite)) if finite.size else 0.0,
        'inf_fraction': float(np.mean(~np.isfinite(costs))), # fraction of infinite costs: feasibility of selected params this iteration
        'cum_evals': (i + 1) * costs.size,
    }

    # Per-anchor diagnostics. anchor_std is the spread of an anchor's contribution
    # ACROSS the swarm; replica_std is the run-to-run noise of the same candidate.
    # An anchor with anchor_std < replica_std cannot steer the search - it is a
    # constant offset, not a gradient - so read the two together.
    survivors = [e for e in (rec.get('extra') or []) if e is not None]
    if survivors:
        keys = survivors[0]['anchors'].keys()
        row['anchor_mean'] = {k: float(np.mean([e['anchors'][k] for e in survivors])) for k in keys}
        row['anchor_std']  = {k: float(np.std([e['anchors'][k] for e in survivors])) for k in keys}
        row['replica_std'] = float(np.mean([np.sqrt(e['cost_var']) for e in survivors]))

    return row

def _append_log_line(row: dict, path: str) -> None:
    """
    Append one JSON line to the iteration log (readable, and parseable back).

    Parameters
    ----------
    row : dict
        The line (_log_row).
    path : str
        The log file (pso_log.jsonl).
    """
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')

def training_matrix(history: list[dict], anchor_labels: list[str]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Training data of the surrogate, from the swarm history.

    The history is unfolded: every finite candidate of every iteration is one sample, weighted
    alike regardless of its iteration.

    Parameters
    ----------
    history : list of dict
        ParticleSwarm.history.
    anchor_labels : list of str
        Order of the breakdown columns.

    Returns
    -------
    params : (M, D) ndarray
        Positions.
    cost : (M,) ndarray
        Costs.
    breakdown : (M, A) ndarray
        Per-anchor terms, columns in anchor_labels order.
    cost_var : (M,) ndarray
        Replica variance of each cost.
    """
    # unfold (n_iter, P) into n_iter * P samples
    params = [] 
    cost = []
    breakdown = []
    cost_var = []

    for rec in history:
        X = rec['X']
        costs = rec['costs']
        extra = rec['extra'] # extra is a list of dicts
        if extra is None: # no per-particle data stored
            continue
        for p in range(len(X)):
            e = extra[p] # 'e' is a dict or None if a run failed
            if e is None:   # failed/inf candidate -> no usable cost
                continue
            params.append(X[p]) 
            cost.append(costs[p])
            # breakdown columns in anchor_labels order
            breakdown.append([e['anchors'][k] for k in anchor_labels])
            cost_var.append(e['cost_var'])

    return (np.asarray(params), np.asarray(cost),
            np.asarray(breakdown), np.asarray(cost_var))

def run_calibration(
    base_cfg: GenerationConfig,
    anchors: Anchors,
    param_bounds: dict,
    pso_config: dict,
    n_replicates: int,
    n_required: int,
    out_root: str,
    surrogate_params: dict[str, Any] | None = None,
    n_workers: int | None = None,
) -> dict:
    """
    Run the PSO calibration and return the best parameters found.

    Iteration 0 evaluates the initial swarm; each following iteration steps the swarm, adds
    the surrogate's proposal once warm, evaluates everything as one batch under
    out_root/iter_{k}, offers the proposal to gbest (consider) before updating the swarm, then
    checkpoints pso_state.json and appends to pso_log.jsonl. An existing pso_state.json is
    resumed (and must match param_bounds and the anchors): clear out_root for a fresh search.

    Parameters
    ----------
    base_cfg : GenerationConfig
        Values of every field not swept.
    anchors : Anchors
        The targets.
    param_bounds : dict
        {field: (lo, hi)} for a config field, {(field, key): (lo, hi)} for one key of a dict
        field (e.g. ('atoms_potential_params', 'k_dihedral')). The order sets the dimensions.
    pso_config : dict
        swarm_size, n_iter (default 30), w, c1, c2, v_max_frac.
    n_replicates : int
        Runs per candidate.
    n_required : int
        Finite replicas needed for a finite cost, 1 <= n_required <= n_replicates.
    out_root : str
        Calibration output directory.
    surrogate_params : dict, optional
        See _surrogate_candidate; None (default) runs plain PSO.
    n_workers : int, optional
        Worker processes of each batch.

    Returns
    -------
    dict
        'gbest_cost', 'gbest_params' (shaped like param_bounds, tuple keys nested), 'history'.

    Raises
    ------
    ValueError
        On a param_bounds field that is not a GenerationConfig field, an n_required out of
        range, or a checkpoint that does not match the current setup.
    """

    cfg_fields = set(GenerationConfig.__dataclass_fields__)

    for key in param_bounds.keys():
        if isinstance(key, tuple):
            field = key[0]
        else:
            field = key

        if field not in cfg_fields:
            raise ValueError(f"ERROR, entry '{key}' is not a valid GenerationConfig field.")

    if not (1 <= n_required <= n_replicates):
        raise ValueError("Required successful runs must be [1, n_replicates] to grant PSO operation.")

    bounds = list(param_bounds.values())
    n_iter = pso_config.get('n_iter', 30)
    swarm_kwargs = {k: v for k, v in pso_config.items() if k != 'n_iter'}   

    os.makedirs(out_root, exist_ok=True)
    state_path = os.path.join(out_root, 'pso_state.json')
    log_path = os.path.join(out_root, 'pso_log.jsonl')

    param_labels = [str(k) for k in param_bounds] # dimension order = param_bounds order
    anchor_labels = _anchor_labels(anchors)
    meta = {'param_labels': param_labels, 'anchor_labels': anchor_labels}

    if os.path.exists(state_path): # resume
        swarm = ParticleSwarm.restore(bounds=bounds, path= state_path, **swarm_kwargs)

        if not (swarm.meta.get('param_labels') == param_labels):
            raise ValueError(f"Expecting PARAM_BOUNDS in this order: {swarm.meta.get('param_labels')}, current={param_labels}")
        if not (set(swarm.meta.get('anchor_labels', [])) == set(anchor_labels)):
            raise ValueError(f"Expecting these Anchors fields: {swarm.meta.get('anchor_labels', [])}")

        n_iter_start = len(swarm.history)
        print(f'[PSO] restarted at iter {n_iter_start}, gbest_cost={swarm.gbest_cost:.6f}')
    else: # new run: evaluate the initial swarm as iteration 0
        swarm = ParticleSwarm(bounds=bounds, **swarm_kwargs)
        swarm.meta = meta
        n_iter_start = 1

        costs, extra = _evaluate_iter(swarm.X, 
                            param_bounds, 
                            base_cfg, 
                            anchors,
                            n_replicates, 
                            n_required, 
                            os.path.join(out_root, 'iter_0'), 
                            n_workers)
        swarm.update(costs, extra)
        swarm.checkpoint(state_path)
        _append_log_line(row = _log_row(rec = swarm.history[-1],
                                  i = len(swarm.history)-1,
                                  gbest_hash = _gbest_hash(swarm.gbest, param_bounds, base_cfg, out_root),
                                  ),
                         path=log_path,
                        )
        print(f'[PSO] iter 0 gbest_cost={swarm.gbest_cost:.6f}')

    for k in range(n_iter_start, n_iter + 1):
        swarm.step()

        candidate = _surrogate_candidate(history= swarm.history,
                                         anchor_labels=anchor_labels,
                                         bounds=bounds,
                                         surrogate_params = surrogate_params,
                                         warmup = surrogate_params.get("warmup",swarm.P) if surrogate_params is not None else swarm.P)
        if candidate is None:
            X_eval = swarm.X
        else:
            X_eval = np.vstack([swarm.X, candidate])

        costs, extra = _evaluate_iter(X_eval, 
                                      param_bounds, 
                                      base_cfg, 
                                      anchors,
                                      n_replicates, 
                                      n_required,
                                      os.path.join(out_root, f'iter_{k}'), 
                                      n_workers)
        if candidate is not None:
            swarm.consider(candidate, costs[-1])

        swarm.update(costs[:swarm.P], extra[:swarm.P])
        swarm.checkpoint(state_path)
        _append_log_line(row = _log_row(rec = swarm.history[-1],
                                  i = len(swarm.history)-1,
                                  gbest_hash = _gbest_hash(swarm.gbest, param_bounds, base_cfg, out_root),
                                  ),
                         path=log_path,
                        )
        print(f'[PSO] iter {k}  gbest_cost={swarm.gbest_cost:.6f}')

    gbest_params: dict = {}
    for i, key in enumerate(param_bounds.keys()):
        if isinstance(key, str):
            gbest_params[key] = float(swarm.gbest[i])
        elif isinstance(key, tuple):
            field_name, param_key = key
            gbest_params.setdefault(field_name, {})[param_key] = float(swarm.gbest[i])

    return {
        'gbest_cost': swarm.gbest_cost,
        'gbest_params': gbest_params,
        'history': swarm.history,
    }


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point of `python -m nanocarbon optimize <config_file>`.

    The config file is a Python file defining BASE_CFG, ANCHORS, PARAM_BOUNDS, PSO_CONFIG,
    N_REPLICATES, N_REQUIRED and OUT_ROOT (optionally SURROGATE_PARAMS, N_WORKERS);
    utilities/calibrate_config.py is the template.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments after the verb; None reads sys.argv.

    Raises
    ------
    ValueError
        If the file cannot be loaded or lacks a required name.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='nanocarbon optimize',
        description='Run PSO calibration from a Python config file.',
    )
    parser.add_argument('config', help='Path to a Python calibrate config file.')
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location('_calibrate_config', args.config)
    if not (spec is not None and spec.loader is not None):
        raise ValueError(f'Could not load config file: {args.config}')
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    required = ('BASE_CFG', 'ANCHORS', 'PARAM_BOUNDS', 'PSO_CONFIG', 'N_REPLICATES', 'OUT_ROOT', 'N_REQUIRED')
    missing = [a for a in required if not hasattr(cfg_mod, a)]
    if not (not missing):
        raise ValueError(f'Config file missing required attributes: {missing}')

    result = run_calibration(
        base_cfg=cfg_mod.BASE_CFG,
        anchors=cfg_mod.ANCHORS,
        param_bounds=cfg_mod.PARAM_BOUNDS,
        pso_config=cfg_mod.PSO_CONFIG,
        n_replicates=cfg_mod.N_REPLICATES,
        n_required=cfg_mod.N_REQUIRED,
        out_root=cfg_mod.OUT_ROOT,
        surrogate_params=getattr(cfg_mod, 'SURROGATE_PARAMS', None),
        n_workers=getattr(cfg_mod, 'N_WORKERS', None),
    )

    print(f'\nCalibration complete. Best cost: {result["gbest_cost"]:.6f}')
    print(f'Best params: {result["gbest_params"]}')
