# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Result containers: RunResult (one run) and BatchResults (a batch directory).

RunResult's scalar fields are the results_{run_id}.json a batch run writes; its array
fields come from the .npy sidecars. BatchResults loads a batch from its per-run files, or
from a consolidated snapshot (batch_results.json + batch_arrays.npz).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# Field names that hold ndarray data and live in the .npz sidecar
# (as opposed to the JSON snapshot). Defined once so to_files / from_files
# in BatchResults never disagree about the split.
_ARRAY_FIELDS = ('psd', 'psd_cum', 'rdf', 'rdf_local')


@dataclass
class RunResult:
    """
    One generated structure: its configuration, outcome and post-processing results.

    Parameters
    ----------
    run_id : str
        Run identity "{config hash}_r{replica}".
    config : dict
        The GenerationConfig, as a dict.
    box_size : float
        Box edge, Angstrom (copied from config, for grouping).
    psd_label : str
        PSD condition label (copied from config).
    generator_type : str
        "ring" or "random" (copied from config).
    status : str
        'ok', or 'generation_error' for a run that raised.
    failure_reason : str
        The error and traceback of a failed run; empty when ok.
    achieved_pore_porosity : float
        Porosity of the pore packing alone (voxel grid).
    n_pores : int
        Number of pores.
    atoms_number : int
        Number of carbon atoms.
    achieved_density : float
        Carbon density, g/cm^3.
    pore_attempts_used : int
        Pore packings tried by the pre-screen.
    pore_attempt_phi_history : list of float
        Porosity of each packing tried.
    accepted_within_tolerance : bool
        Whether the kept packing met phi_tolerance.
    computed_porosity : float
        compute_porosity of the structure (probe-center fraction).
    accessible_volume : float
        Probe-center accessible volume, Angstrom^3 (compute_AVA).
    occupiable_volume : float
        Probe-occupiable volume, Angstrom^3 (compute_AVA).
    accessible_surface : float
        Accessible surface area, Angstrom^2 (compute_AVA).
    pore_connectivity : float
        k_avg (compute_connectivity).
    ring_histogram : dict
        {ring size (str): count}.
    coordination_histogram : dict
        {'0 bonds': %, ..., '>4 bonds': %}.
    wall_time_s : float
        Run time, seconds.
    psd, psd_cum : (M, 2) ndarray or None
        Differential and cumulative PSD (compute_psd).
    rdf : (n_bins, 2) ndarray or None
        [r, g(r)].
    rdf_local : (S, n_bins, 2) ndarray or None
        Per-sphere RDFs, when the run enabled rdf_local_n_samples.
    """

    # bookkeeping and provenance
    run_id: str = ''
    config: dict = field(default_factory=dict)

    # condition labels, copied from config for grouping
    box_size: float = 0.0
    psd_label: str = ''
    generator_type: str = 'ring'

    # outcome
    status: str = 'ok'

    failure_reason: str = ''

    # generation
    achieved_pore_porosity: float = float('nan')
    n_pores: int = 0
    atoms_number: int = 0
    achieved_density: float = float('nan')

    # pre-screen
    pore_attempts_used: int = 0
    pore_attempt_phi_history: list = field(default_factory=list)
    accepted_within_tolerance: bool = False

    # post-processing scalars
    computed_porosity: float = float('nan')
    accessible_volume: float = float('nan')
    occupiable_volume: float = float('nan')
    accessible_surface: float = float('nan')
    pore_connectivity: float = float('nan')
    ring_histogram: dict = field(default_factory=dict)
    coordination_histogram: dict = field(default_factory=dict)

    # timing
    wall_time_s: float = float('nan')

    # arrays (sidecar files); None when absent
    psd: np.ndarray | None = None
    psd_cum: np.ndarray | None = None
    rdf: np.ndarray | None = None
    rdf_local: np.ndarray | None = None

    @classmethod
    def from_results_dict(cls, d: dict) -> 'RunResult':
        """
        RunResult from the dict stored in results_{run_id}.json.

        Keys missing from older files take the defaults, and unknown keys are ignored, so older
        and newer batches load alike.

        Parameters
        ----------
        d : dict
            The stored dict.

        Returns
        -------
        RunResult
        """
        # Pull out only the keys the dataclass recognizes; ignore extras
        # so future schema additions in the worker don't break loading.
        recognised = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in recognised}
        return cls(**kwargs)

    @classmethod
    def from_run_files(
        cls,
        results_json_path: str,
        psd_npy: str | None = None,
        psd_cum_npy: str | None = None,
        rdf_npy: str | None = None,
        rdf_local_npy: str | None = None,
    ) -> 'RunResult':
        """
        RunResult from a results JSON plus its .npy sidecars (absent files are skipped).

        Parameters
        ----------
        results_json_path : str
            Path of results_{run_id}.json.
        psd_npy, psd_cum_npy, rdf_npy, rdf_local_npy : str, optional
            Paths of the sidecar arrays.

        Returns
        -------
        RunResult
        """
        with open(results_json_path) as f:
            d = json.load(f)
        run = cls.from_results_dict(d)
        if psd_npy and os.path.exists(psd_npy):
            run.psd = np.load(psd_npy)
        if psd_cum_npy and os.path.exists(psd_cum_npy):
            run.psd_cum = np.load(psd_cum_npy)
        if rdf_npy and os.path.exists(rdf_npy):
            run.rdf = np.load(rdf_npy)
        if rdf_local_npy and os.path.exists(rdf_local_npy):
            run.rdf_local = np.load(rdf_local_npy)
        return run

    def to_scalar_dict(self) -> dict:
        """
        The scalar fields as a dict (arrays excluded: they go to the .npz sidecar).

        Returns
        -------
        dict
        """
        d = asdict(self)
        for k in _ARRAY_FIELDS:
            d.pop(k, None)
        return d

    def array_payload(self) -> dict[str, np.ndarray]:
        """
        The array fields keyed "{run_id}_{name}", so many runs can share one .npz.

        Returns
        -------
        dict of str to ndarray
            Only the arrays that are not None.
        """
        out: dict[str, np.ndarray] = {}
        for k in _ARRAY_FIELDS:
            arr = getattr(self, k)
            if arr is not None:
                out[f'{self.run_id}_{k}'] = np.asarray(arr)
        return out

    @property
    def config_hash(self) -> str:
        """The hash part of run_id, shared by the replicas of one configuration."""
        return self.run_id.split('_r')[0]

@dataclass
class BatchResults:
    """
    The RunResults of one batch output directory.

    Parameters
    ----------
    out_root : str
        The batch output directory.
    runs : list of RunResult
        The runs.
    """

    out_root: str
    runs: list[RunResult] = field(default_factory=list)

    @classmethod
    def from_run_dir(cls, out_root: str) -> 'BatchResults':
        """
        Load a batch from its per-run files: results_*.json in the subdirectories of out_root,
        with their .npy sidecars.

        Only the run ids listed in out_root/manifest.json (written by the latest run_batch) are
        loaded; others, left by an edited grid, are named in a warning and skipped. Without a
        manifest every run is loaded.

        Parameters
        ----------
        out_root : str
            The batch output directory.

        Returns
        -------
        BatchResults
            Runs sorted by run_id; empty if out_root does not exist.
        """
        runs: list[RunResult] = []
        if not os.path.isdir(out_root):
            return cls(out_root=out_root, runs=runs)

        # the run ids the latest run_batch expects (manifest.json)
        expected = None
        json_path = os.path.join(out_root, 'manifest.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                payload = json.load(f)
            expected = set( payload['expected_ids'] )

        if expected is None:
            print("Manifest file not found, unable to check expected run ids. Loading all configurations found")

        # Worker writes one results_{run_id}.json per condition subdirectory.
        # The glob is name-agnostic so any subdir naming convention works.
        pattern = os.path.join(out_root, '*', 'results_*.json')
        orphans = [] # list of ids stale from previous configurations, not needed for the current investigated parameters config
        for results_path in sorted(glob.glob(pattern)):
            condition_dir = os.path.dirname(results_path)
            run_id_str = os.path.basename(results_path)
            # results_{run_id}.json -> {run_id}, an hash_replica str
            run_id = run_id_str[len('results_'):-len('.json')]

            if expected is not None and run_id not in expected:
                orphans.append(run_id) # record which run_id should not be loaded but deleted instead
                continue

            run = RunResult.from_run_files(
                results_path,
                psd_npy=os.path.join(condition_dir, f'psd_{run_id}.npy'),
                psd_cum_npy=os.path.join(condition_dir, f'psd_cum_{run_id}.npy'),
                rdf_npy=os.path.join(condition_dir, f'rdf_{run_id}.npy'),
                rdf_local_npy=os.path.join(condition_dir, f'rdf_local_{run_id}.npy'),
            )
            runs.append(run)

        # if we have stale runs, let's see them
        if orphans:
            print(f"WARNING: Stale run ids not considered in " 
                  f"most CONDITIONS configuration: {orphans}. "
                  "The listed runs are not loaded")

        # Sort by run_id so the order matches the batch's
        # original scheduling. Helpful for reading the JSON snapshot.
        runs.sort(key=lambda r: r.run_id)
        return cls(out_root=out_root, runs=runs)

    @classmethod
    def from_files(cls, out_dir: str) -> 'BatchResults':
        """
        Load a batch from its snapshot (batch_results.json + batch_arrays.npz), or from the
        per-run files (from_run_dir) when there is no snapshot.

        Parameters
        ----------
        out_dir : str
            The batch output directory.

        Returns
        -------
        BatchResults
        """
        json_path = os.path.join(out_dir, 'batch_results.json')
        npz_path = os.path.join(out_dir, 'batch_arrays.npz')

        if not os.path.exists(json_path):
            # No snapshot — rebuild from per-run files.
            return cls.from_run_dir(out_dir)

        with open(json_path) as f:
            payload = json.load(f)

        runs = [RunResult.from_results_dict(d) for d in payload['runs']]

        # Reattach arrays if the .npz exists.  Keys are
        # f"{run_id}_{field}" over _ARRAY_FIELDS (see RunResult.array_payload).
        if os.path.exists(npz_path):
            with np.load(npz_path) as npz:
                for run in runs:
                    for k in _ARRAY_FIELDS:
                        key = f'{run.run_id}_{k}'
                        if key in npz.files:
                            setattr(run, k, npz[key])

        return cls(out_root=payload.get('out_root', out_dir), runs=runs)

    @classmethod
    def consolidate(cls, out_root: str) -> 'BatchResults':
        """
        Load a batch from its per-run files and write its snapshot; run_batch calls it at the end.

        Parameters
        ----------
        out_root : str
            The batch output directory.

        Returns
        -------
        BatchResults
        """
        batch = cls.from_run_dir(out_root)
        batch.to_files(out_root)
        return batch

    def to_files(self, out_dir: str) -> None:
        """
        Write the snapshot: batch_results.json (scalars) and batch_arrays.npz (arrays).

        Each file is written to a temporary name and then renamed, so an interrupted write leaves
        the previous snapshot intact.

        Parameters
        ----------
        out_dir : str
            Output directory (created if missing).
        """
        os.makedirs(out_dir, exist_ok=True)

        # scalars
        json_path = os.path.join(out_dir, 'batch_results.json')
        json_tmp = json_path + '.tmp'
        with open(json_tmp, 'w') as f:
            json.dump(
                {
                    'out_root': self.out_root,
                    'n_runs': len(self.runs),
                    'runs': [r.to_scalar_dict() for r in self.runs],
                },
                f,
                indent=2,
            )
        os.replace(json_tmp, json_path)

        # arrays; skip the file entirely if no run carries arrays
        # (e.g. a partially-loaded BatchResults).
        payload: dict[str, Any] = {}
        for run in self.runs:
            payload.update(run.array_payload())
        if payload:
            npz_path = os.path.join(out_dir, 'batch_arrays.npz')
            npz_tmp = npz_path + '.tmp'
            np.savez_compressed(npz_tmp, **payload)
            # numpy adds .npz suffix if missing; force the actual path.
            if not os.path.exists(npz_tmp) and os.path.exists(npz_tmp + '.npz'):
                os.replace(npz_tmp + '.npz', npz_path)
            else:
                os.replace(npz_tmp, npz_path)

    def filter(self, **kwargs) -> 'BatchResults':
        """
        The runs whose attributes match every keyword, e.g. filter(generator_type='ring').

        Parameters
        ----------
        **kwargs
            Attribute name -> required value.

        Returns
        -------
        BatchResults
        """
        def matches(run: RunResult) -> bool:
            """
            Whether a run matches every keyword of filter.

            Parameters
            ----------
            run : RunResult
                The run.

            Returns
            -------
            bool
            """
            return all(getattr(run, k, None) == v for k, v in kwargs.items())

        return BatchResults(
            out_root=self.out_root,
            runs=[r for r in self.runs if matches(r)],
        )

    @property
    def ok_runs(self) -> list[RunResult]:
        """Runs with status 'ok'."""
        return [r for r in self.runs if r.status == 'ok']

    @property
    def failed_runs(self) -> list[RunResult]:
        """Runs with any other status (recorded failures)."""
        return [r for r in self.runs if r.status != 'ok']

    def __repr__(self) -> str:
        """One-line summary: run count, failures, generators, PSD labels, box sizes."""
        n = len(self.runs)
        if n == 0:
            return f'BatchResults(out_root={self.out_root!r}, n_runs=0)'
        gens = sorted({r.generator_type for r in self.runs})
        psds = sorted({r.psd_label for r in self.runs})
        boxes = sorted({r.box_size for r in self.runs})
        n_fail = len(self.failed_runs)
        fail_part = f', n_failed={n_fail}' if n_fail else ''
        return (
            f'BatchResults(out_root={self.out_root!r}, n_runs={n}{fail_part}, '
            f'generators={gens}, psds={psds}, box_sizes={boxes})'
        )

def load_psd(path: str) -> np.ndarray:
    """
    Load a tabulated PSD and check its shape.

    Parameters
    ----------
    path : str
        A .npy file of shape (K, 2): diameter (Angstrom), value.

    Returns
    -------
    (K, 2) ndarray

    Raises
    ------
    ValueError
        If the array is not (K, 2).
    """
    arr = np.load(path)
    if not (arr.ndim == 2 and arr.shape[1] == 2):
        raise ValueError(f'Expected Nx2 array in {path!r}, got {arr.shape!r}')
    return arr