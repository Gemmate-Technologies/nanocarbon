# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Parallel batch generation: a grid of configurations, one process per structure.

run_batch expands a CONDITIONS grid over a base GenerationConfig, runs each configuration
in a process pool and writes per-run files that make a resubmission resume where it
stopped; main is the `python -m nanocarbon generate <config_file>` entry point. The worker
_generate_and_analyze_one must stay at module level: the pool pickles it by import path.
"""

import json
import os
import time
import warnings
from hashlib import sha1
from typing import Any

from itertools import product
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import traceback

import numpy as np
from .config import GenerationConfig
from .io_functions import LJ_WALL_MINIMUM, WALL_UNIT_TO_EV
from .results import BatchResults, RunResult
from .pipeline import run_generation

def _hash_config(cfg: GenerationConfig) -> str:
    """
    Content hash of a configuration: the identity behind run_id.

    The hash covers every field except run_id and out_dir, so an unchanged configuration keeps
    its id when the grid is edited or reordered and its finished runs are skipped on resume.
    Defaults are included: adding a field or changing a default changes the id of every
    configuration relying on it.

    Parameters
    ----------
    cfg : GenerationConfig
        The configuration.

    Returns
    -------
    str
        First 8 hex digits of the SHA-1 of the sorted-key JSON of the fields.
    """

    d = cfg.to_dict()
    d_filtered = {k: v for k, v in d.items() if k not in ["run_id", "out_dir"]} # identity: the simulation parameters only, not the id or the output path
    d_str = json.dumps(d_filtered, sort_keys = True) # sorted keys: the same config always serializes the same
    d_str = d_str.encode()
    return sha1(d_str).hexdigest()[:8]


def _wall_params(cfg: GenerationConfig) -> tuple[list[float], str]:
    """
    LAMMPS wall parameters of the pore file, taken from the generation pore potential.

    nanocarbon's lj pore potential and LAMMPS' lj126 wall/region are the same function of the
    same variable (distance to the pore surface), so epsilon = A and sigma = B make the anneal
    wall the generation wall. The cutoff has no generation counterpart (the generation potential
    is untruncated): it is the wall's own minimum (io_functions.LJ_WALL_MINIMUM), so the wall
    only repels (WCA). epsilon is converted to eV for a `units metal` deck: A carries the atoms
    potential's energy unit (io_functions.WALL_UNIT_TO_EV).

    Parameters
    ----------
    cfg : GenerationConfig
        The run configuration.

    Returns
    -------
    LJ_vars : list of float
        [epsilon (eV), sigma (Angstrom), cutoff (Angstrom)].
    LJ_type : str
        '126' or '93'.

    Raises
    ------
    ValueError
        If atoms_potential_type has no entry in WALL_UNIT_TO_EV.

    Warns
    -----
    UserWarning
        For a pore potential or exponents with no LAMMPS twin; an lj126 stand-in is written
        instead of blocking the run.
    """
    params = cfg.pores_potential_params or {}

    if cfg.pores_potential_type != 'lj':
        warnings.warn(
            f"pores_potential_type={cfg.pores_potential_type!r} has no wall/region counterpart: "
            "its parameters are written as an lj126 epsilon/sigma, so the anneal wall is only an "
            "approximation of the wall the structure was grown against. Prefer 'lj'."
        )

    # LAMMPS offers lj126 and lj93 only; map the exponents onto whichever fits.
    e1, e2 = params.get('e1', 12), params.get('e2', 6)
    LJ_type = {(12, 6): '126', (9, 3): '93'}.get((e1, e2), '')
    if not LJ_type:
        warnings.warn(
            f"lj pore exponents ({e1}, {e2}) match neither lj126 nor lj93; writing an lj126 wall, "
            "which is not the potential the structure was grown against."
        )
        LJ_type = '126'

    A = params.get('A', 1e-1)   # LJPoresPotential.set_potential_parameters defaults
    B = params.get('B', 1.0)
    if cfg.atoms_potential_type not in WALL_UNIT_TO_EV:
        raise ValueError(f"No energy unit known for atoms_potential_type={cfg.atoms_potential_type!r}; "
                         f"add it to io_functions.WALL_UNIT_TO_EV.")
    epsilon = A * WALL_UNIT_TO_EV[cfg.atoms_potential_type]
    return [epsilon, B, LJ_WALL_MINIMUM[LJ_type] * B], LJ_type


def _generate_and_analyze_one(cfg: GenerationConfig) -> dict:
    """
    Worker: generate and post-process one structure, and write its files.

    A run whose results_{run_id}.json exists is not redone: the stored dict is returned. The
    global numpy RNG is reseeded from the OS in every worker, so replicas differ; a single run
    cannot be regenerated from its configuration. Any error, in the generation or while
    writing the files, is recorded as status='generation_error' instead of stopping the batch
    (a run that fails while writing may leave some of its files behind). In cfg.out_dir it
    writes config_{run_id}.json and results_{run_id}.json, and for a successful run the .npy
    arrays, carbon_{run_id}.data / .json and pores_{run_id}.pores_settings / .json.

    Parameters
    ----------
    cfg : GenerationConfig
        The run configuration, run_id and out_dir set.

    Returns
    -------
    dict
        The scalar results (RunResult.to_scalar_dict).
    """
    results_path = os.path.join(cfg.out_dir, f'results_{cfg.run_id}.json')
    if os.path.exists(results_path):
        print(f'[run {cfg.run_id}] already done — loading from disk', flush=True)
        with open(results_path) as f:
            return json.load(f)

    # Forked workers (Linux) inherit the parent's RNG state and would produce identical
    # replicas: reseed from the OS in every worker. Trade-off: a single run cannot be
    # regenerated from its configuration.
    np.random.seed() # KEEP.

    os.makedirs(cfg.out_dir, exist_ok=True)
    print(
        f'[run {cfg.run_id}] starting',
        flush=True,
    )
    cfg.to_json(os.path.join(cfg.out_dir, f'config_{cfg.run_id}.json'))

    try:
        result, carbon_struct, pore_struct = run_generation(cfg)

        tol_tag = ('within tolerance' if result.accepted_within_tolerance
                   else 'best-so-far, tolerance NOT met')
        print(
            f'[run {cfg.run_id}] pores done — attempts={result.pore_attempts_used}, '
            f'achieved porosity={result.achieved_pore_porosity:.3f} ({tol_tag})',
            flush=True,
        )
        print(
            f'[run {cfg.run_id}] carbon done — {result.atoms_number} atoms, '
            f'density={result.achieved_density:.3f} g/cm³',
            flush=True,
        )

        # Sidecar arrays - needed by BatchResults.from_run_dir. Skipped
        # when run_generation returns None for them (random-generator runs:
        # no post-processing has been computed yet — see pipeline.py).
        for name, array in (
            ('psd',       result.psd),
            ('psd_cum',   result.psd_cum),
            ('rdf',       result.rdf),
            ('rdf_local', result.rdf_local),
        ):
            if array is not None:
                np.save(os.path.join(cfg.out_dir, f'{name}_{cfg.run_id}.npy'), array)

        # LAMMPS input and JSON of the structure and its pores
        if not (carbon_struct is not None and pore_struct is not None):
            raise RuntimeError("Missing structures despite run is set as 'ok!', check log.")
        carbon_struct.print_LAMMPS(name = os.path.join(cfg.out_dir, f'carbon_{cfg.run_id}'))
        LJ_vars, LJ_type = _wall_params(cfg)
        pore_struct.print_LAMMPS(LJ_vars = LJ_vars, LJ_type = LJ_type,
                                 name = os.path.join(cfg.out_dir, f'pores_{cfg.run_id}'))
        with open(os.path.join(cfg.out_dir, f'carbon_{cfg.run_id}.json'), 'w') as f:
            json.dump(carbon_struct.to_dict(), f, indent=2)
        with open(os.path.join(cfg.out_dir, f'pores_{cfg.run_id}.json'), 'w') as f:
            json.dump(pore_struct.to_dict(), f, indent=2)
    except Exception as e:
        result = RunResult(
            run_id=cfg.run_id,
            config=cfg.to_dict(),
            box_size=float(cfg.box_size),
            psd_label=cfg.psd_label,
            generator_type=cfg.generator_type,
            status='generation_error', # != ok
            failure_reason=f'{e!r}\n{traceback.format_exc()}', # log failure reason
        )
        print(f'[run {cfg.run_id}] FAILED ({result.status}) — {result.failure_reason.splitlines()[0]}; logging and moving on.', flush=True)

    scalar = result.to_scalar_dict()
    with open(results_path, 'w') as f:
        json.dump(scalar, f, indent=2)
    return scalar


def _expand_axis(axis_name: str, val) -> list:
    """
    Normalize one CONDITIONS axis into the list of its values.

    - a list passes through; a scalar becomes a one-element list;
    - a dict with only scalar values becomes a one-element list;
    - a dict with some list values expands into the Cartesian product of those lists, the
      scalar keys copied into every entry, with an '{axis_name}_label' naming each entry;
    - a tuple or set raises.

    When axis_name is a GenerationConfig field (e.g. 'pores_potential_params'), each entry is
    wrapped as {axis_name: entry} and sets the whole field; otherwise axis_name is a
    placeholder and the entry's keys must be GenerationConfig fields (others are dropped later
    by _build_configs).

    Parameters
    ----------
    axis_name : str
        The CONDITIONS key.
    val : any
        Its value.

    Returns
    -------
    list

    Raises
    ------
    TypeError
        On a tuple or set.
    """
    if isinstance(val, list):
        return val

    if not isinstance(val, dict):
        if isinstance(val, tuple) or isinstance(val, set):
            raise TypeError(f"Field '{axis_name}' type error: tuples and set containers not allowed")
        return [val]


    is_field = axis_name in GenerationConfig.__dataclass_fields__

    swept = {k: v for k, v in val.items() if isinstance(v, list)}
    if not swept:
        if is_field:
            return [{axis_name: val}]
        return [val]

    fixed = {k: v for k, v in val.items() if not isinstance(v, list)}
    expanded = []
    for combo in product(*swept.values()):
        entry = {**fixed}
        for k, v in zip(swept.keys(), combo):
            entry[k] = v
        label_parts = [f'{k}{v:g}' if isinstance(v, float) else f'{k}{v}'
                       for k, v in zip(swept.keys(), combo)]
        if is_field:
            expanded.append({axis_name: entry,
                             f'{axis_name}_label': '_'.join(label_parts)})
        else:
            entry[f'{axis_name}_label'] = '_'.join(label_parts)
            expanded.append(entry)
    return expanded


def _build_configs(
    conditions: dict[str, Any] ,
    n_per_condition: int,
    out_root: str,
    base_cfg: GenerationConfig,
) -> list[GenerationConfig]:
    """
    Expand a CONDITIONS grid into one GenerationConfig per structure.

    Every combination of the axes (see _expand_axis) overrides base_cfg through
    dataclasses.replace and gets its own directory under out_root, named from the axis values
    or labels; it is then replicated n_per_condition times with run_id "{hash}_r{replica}".
    Override keys that are not GenerationConfig fields are dropped with a printed note (a
    typo'd axis is otherwise a silent no-op).

    Parameters
    ----------
    conditions : dict
        {axis: list of values, scalar, or dict}; see _expand_axis.
    n_per_condition : int
        Replicas per combination.
    out_root : str
        Batch output directory.
    base_cfg : GenerationConfig
        Defaults for every field the grid does not set.

    Returns
    -------
    list of GenerationConfig
    """
    conditions = {k: _expand_axis(k, v) for k, v in conditions.items()}
    cfg_fields = set(GenerationConfig.__dataclass_fields__)
    axis_names = list(conditions.keys())

    configs = []

    for combo in product(*conditions.values()):
        # Accumulate all field overrides and build a human-readable dir name.
        overrides: dict = {}
        label_parts: list[str] = []

        for axis, val in zip(axis_names, combo):
            if isinstance(val, dict):
                # Grouped overrides: unpack all keys as config fields.
                # Use the {axis}_label entry for the dir name if present.
                overrides.update(val)
                label_parts.append(str(val.get(f'{axis}_label', axis)))
            elif isinstance(val, float):
                overrides[axis] = val
                label_parts.append(f'{axis}{val:g}')
            else:
                overrides[axis] = val
                label_parts.append(f'{axis}{val}')

        condition_dir = os.path.join(out_root, '_'.join(label_parts))

        # dataclasses.replace accepts only recognized fields; silently drop
        # any extra keys carried inside dict values (e.g. metadata labels).
        valid_overrides = {k: v for k, v in overrides.items() if k in cfg_fields}
        excluded_labels = [k for k in overrides.keys() if k not in cfg_fields and isinstance(k, str) and '_label' in k]
        list_neglected = [k for k in overrides.keys() if k not in cfg_fields]
        list_neglected = [k for k in list_neglected if k not in excluded_labels]

        if list_neglected:
            print(f"The following fields are not recognised as valid entries for dataclass 'GenerationConfig': {list_neglected}")

        # Clone base_cfg and apply the per-condition overrides. Every other field flows through unchanged.
        cfg_condition = dataclasses.replace(base_cfg, out_dir = condition_dir, **valid_overrides)
        h = _hash_config(cfg_condition)

        for r in range(n_per_condition):
            # change the run_id
            cfg = dataclasses.replace(cfg_condition, run_id = f"{h}_r{r}") # hash_replicaNumber
            configs.append(cfg)

    return configs

def run_batch(conditions: dict[str, list],
              n_per_condition: int,
              out_root: str,
              base_cfg: GenerationConfig,
              n_workers: int | None = None,
              ) -> BatchResults | None:
    """
    Run a grid of configurations in parallel and consolidate the results.

    The grid (_build_configs) runs in a process pool, one structure per worker process, which
    bounds the memory a worker can accumulate. A worker killed by the OS (e.g. out of memory)
    stops the batch with BrokenProcessPool instead of hanging; finished runs are on disk, so a
    resubmission resumes. Writes manifest.json (the expected run ids), summary.json,
    failures.json when any run failed, and the snapshot (BatchResults.consolidate).

    Parameters
    ----------
    conditions : dict
        The CONDITIONS grid; see _build_configs.
    n_per_condition : int
        Replicas per combination.
    out_root : str
        Batch output directory.
    base_cfg : GenerationConfig
        Defaults for every field the grid does not set.
    n_workers : int, optional
        Worker processes. Default os.cpu_count(); keep the peak memory per worker times
        n_workers within the node.

    Returns
    -------
    BatchResults or None
        The consolidated batch; None if writing the snapshot failed (the per-run files stay
        the record).
    """

    if n_workers is None:
        n_workers = os.cpu_count()

    configs = _build_configs(
        conditions = conditions,
        n_per_condition = n_per_condition,
        out_root = out_root,
        base_cfg = base_cfg,
    )

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'manifest.json'), 'w') as f: # write out the list of runs we plan to do 
        ids_dict = {'expected_ids': [cfg.run_id for cfg in configs]}
        json.dump(ids_dict, f, indent=2)

    axis_summary = ' x '.join(f'{len(v)} {ax}' for ax, v in conditions.items())
    print(
        f'Launching {len(configs)} tasks '
        f'({axis_summary} x {n_per_condition} replicates) '
        f'across {n_workers} workers.'
    )

    t0 = time.perf_counter() 
    # One task per worker process (max_tasks_per_child=1) bounds per-worker memory growth.
    # A worker killed by the OS (e.g. out of memory) makes the executor raise
    # BrokenProcessPool instead of waiting forever; finished runs are on disk, so a
    # resubmission resumes from where the batch stopped.
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=1) as pool:
        results = list(pool.map(_generate_and_analyze_one, configs))
    elapsed = time.perf_counter() - t0
    print(f'\nAll {len(results)} tasks finished in {elapsed/60:.1f} min.')
    print(f'Results saved under {os.path.abspath(out_root)}/')

    # Per-run results_{id}.json files are the source of truth. We also
    # write a compact failures.json so a partial-failure batch is
    # immediately visible in the SLURM log, and consolidate the per-run
    # files into a single batch_results.json + batch_arrays.npz snapshot.

    failures = [r for r in results if r.get('status', 'ok') != 'ok']
    print(f'  successes : {len(results) - len(failures)} / {len(results)}')
    print(f'  failures  : {len(failures)}')

    os.makedirs(out_root, exist_ok=True)
    if failures:
        fail_path = os.path.join(out_root, 'failures.json')
        compact = list(failures)
        with open(fail_path, 'w') as f:
            json.dump(compact, f, indent=2)
        print(f'  failure log -> {fail_path}')
        for r in failures:
            print(f'    [FAILED] run_id={r.get("run_id")}: '
                  f'{r.get("failure_reason", "unknown")}')

    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Failure here is non-fatal: the per-run files remain the record.
    try:
        batch = BatchResults.consolidate(out_root)
        print(f'Batch snapshot: {len(batch.runs)} runs in {out_root}/batch_results.json')
        return batch
    except Exception as exc: 
        print(f'warning: could not consolidate BatchResults snapshot: {exc!r}')
        return None


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point of `python -m nanocarbon generate <config_file>`.

    The config file is a Python file defining CONDITIONS, N_PER_CONDITION, OUT_ROOT and
    BASE_CFG (and optionally N_WORKERS); utilities/batch_config.py is the template.

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
    import importlib.util


    parser = argparse.ArgumentParser(prog='nanocarbon generate',
        description='Run a parallel batch from a Python config file.')
    parser.add_argument(
        'config',
        help='Path to a Python config file defining CONDITIONS, '
             'N_PER_CONDITION, OUT_ROOT, BASE_CFG (and optionally N_WORKERS).',
    )
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location('_batch_config', args.config)
    if not (spec is not None and spec.loader is not None):
        raise ValueError(f'Could not load config file: {args.config}')
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    missing = [a for a in ('CONDITIONS', 'N_PER_CONDITION', 'OUT_ROOT', 'BASE_CFG')
               if not hasattr(cfg, a)]
    if not (not missing):
        raise ValueError(f'Config file missing required attributes: {missing}')

    run_batch(
        conditions=cfg.CONDITIONS,
        n_per_condition=cfg.N_PER_CONDITION,
        out_root=cfg.OUT_ROOT,
        base_cfg=cfg.BASE_CFG,
        n_workers=getattr(cfg, 'N_WORKERS', None),
    )
