# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Structural analysis of an anneal trajectory, written to one CSV.

Usage: python utilities/analyze_anneal.py traj.lammpstrj -o results.csv
       python utilities/analyze_anneal.py traj.lammpstrj -o results.csv --full-every 20

Two tiers, because the two questions have different costs and cadences.

CHEAP tier, every frame: coordination and bond turnover. They read only the bond graph and
answer the topological question, has bond rearrangement stopped, which is what anneal
convergence means. This gives the sp2(t) curve.

EXPENSIVE tier, first and last frame only (or every --full-every frames): PSD, porosity,
accessible surface and volumes, RDF, rings and pore connectivity k_avg. These build voxel
grids, pair counts or a Voronoi tessellation and take minutes to hours; they answer a
different question, did the constructed geometry survive, which needs only the endpoints.

Meant for a cluster: one trajectory per call, no plotting, no display.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from nanocarbon import CarbonStructure
from nanocarbon.io_functions import read_LAMMPS_traj
from nanocarbon.post_process import (
    count_coordination_number,
    count_rings,
    compute_psd,
    compute_porosity,
    compute_AVA,
    compute_connectivity,
    radial_distribution_function,
)

# Post-process parameters. These MUST match the ones of the values you compare against
# (e.g. the post-process settings of the calibration config), or a comparison measures the
# parameter mismatch instead of the structural difference.
R_C = 1.71
R_N = 1.55
VOXEL_LEN = 0.5
RDF_BIN = 0.25
PSD_N_ITER = 2_500
POROSITY_N_ITER = 100_000
PP_SEED = 42
BOND_DIST, BOND_TOL = 1.42, 1.2   # -> 1.704 A cutoff
RING_MAX = 12

def edge_set(s: CarbonStructure):
    """
    The bonds of a structure as a set of (i, j) pairs with i < j.

    Parameters
    ----------
    s : CarbonStructure
        The structure.

    Returns
    -------
    set of tuple
    """
    return {(min(i, j), max(i, j)) for i, nb in s.adjacency.items() for j in nb}


def cheap_row(s: CarbonStructure) -> dict:
    """
    Bond-graph observables of one frame (seconds per frame).

    Parameters
    ----------
    s : CarbonStructure
        The frame.

    Returns
    -------
    dict
        n_atoms, n_bonds, coord_0 ... coord_4 and coord_over4 (percent of atoms).
    """
    coord = count_coordination_number(s)
    row = {
        'n_atoms': s.atoms_number,
        'n_bonds': int(sum(len(v) for v in s.adjacency.values()) // 2),
    }
    for i in range(5):
        row[f'coord_{i}'] = coord[f'{i} bonds']
    row['coord_over4'] = coord['>4 bonds']

    return row


def full_row(s: CarbonStructure, out_dir: Path, tag: str) -> dict:
    """
    Voxel-grid, pair-count and graph observables of one frame (minutes to hours per frame).

    Also saves the PSD and RDF curves as psd_{tag}.npy and rdf_{tag}.npy in out_dir.

    Parameters
    ----------
    s : CarbonStructure
        The frame.
    out_dir : Path
        Where the curves go.
    tag : str
        Frame tag of the file names.

    Returns
    -------
    dict
        porosity_geom, asa_A2, asa_m2_g, av_centre_A3, av_occupiable_A3, av_occupiable_frac,
        k_avg, psd_mode_A, psd_mean_A, ring_3 ... ring_12, ring_total, ring_defect_ratio.
    """
    psd, psd_cum, _, _, _ = compute_psd(
        s, N_iter=PSD_N_ITER, bin_size=RDF_BIN, voxel_len=VOXEL_LEN,
        r_C=R_C, r_N=R_N, seed=PP_SEED,
    )
    porosity = compute_porosity(
        s, N_iter=POROSITY_N_ITER, r_atom=R_C, r_probe=R_N,
        voxel_len=VOXEL_LEN, seed=PP_SEED,
    )
    ava = compute_AVA(s, VOXEL_LEN, R_C, R_N)
    asa, av_centre, av_occupiable = ava.asa_smooth_A2, ava.av_center_A3, ava.av_occupiable_A3
    # pore-pore connectivity: no external tool measures it, so the anneal series is
    # the only place it can be reported. Voronoi on a buffer-padded tile, so it scales
    # with atom count -- expensive tier only.
    k_avg = compute_connectivity(s, r_C=R_C, r_N=R_N)
    r, _, g = radial_distribution_function(s, size_bin=RDF_BIN)

    ring_counts, _ = count_rings(s, max_ring_size=RING_MAX)

    # keep the curves: the scalars below are only summaries of them
    np.save(out_dir / f'psd_{tag}.npy', psd)
    np.save(out_dir / f'rdf_{tag}.npy', np.column_stack([r, g]))

    d, f = psd[:, 0], np.clip(psd[:, 1].astype(float), 0.0, None)
    area = np.trapezoid(f, d)
    volume_A3 = float(np.prod(s.Box))

    rings_dict={}
    total = sum(ring_counts.values())
    for n in range(3, RING_MAX + 1):
        rings_dict[f'ring_{n}'] = ring_counts.get(n, 0)
    # (5+7)/6: defect rings per hexagon
    hexes = ring_counts.get(6, None) 
    rings_dict['ring_total'] = total
    rings_dict['ring_defect_ratio'] = (
        (ring_counts.get(5, 0) + ring_counts.get(7,0)) / hexes if hexes else float('nan')
    )

    return {
        'porosity_geom': porosity,   # probe-center porosity (r_probe = R_N), despite the name
        'asa_A2': asa,
        'asa_m2_g': asa / (s.atoms_number * 12.011 / 6.02214076e23) * 1e-20,
        'av_centre_A3': av_centre,
        'av_occupiable_A3': av_occupiable,
        'av_occupiable_frac': av_occupiable / volume_A3,
        'k_avg': k_avg,
        'psd_mode_A': float(d[np.argmax(f)]) if area > 0 else float('nan'),
        'psd_mean_A': float(np.trapezoid(f * d, d) / area) if area > 0 else float('nan'),
        **rings_dict
    }

def main() -> None:
    """
    Read the trajectory, analyze every frame (expensive tier on the selected ones) and
    rewrite the CSV after each frame.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trajectory', type=Path, help='.lammpstrj file')
    ap.add_argument('-o', '--out', type=Path, default=None, help='output CSV')
    ap.add_argument('--full-every', type=int, default=0,
                    help='also run the expensive tier every N frames (0 = first+last only)')
    ap.add_argument('--atom-type', type=int, default=1, help='carbon atom type in the dump')
    ap.add_argument('--max-frames', type=int, default=None, help='stop after N frames')
    args = ap.parse_args()

    assert args.trajectory.exists(), f'no such trajectory: {args.trajectory}'
    out_csv = args.out or args.trajectory.with_suffix('.csv')
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'reading {args.trajectory} ...', flush=True)
    traj = read_LAMMPS_traj(args.trajectory, max_frame_index=args.max_frames)
    frames = sorted(traj.keys())
    print(f'{len(frames)} frames', flush=True)

    last = frames[-1]
    rows, fieldnames = [], None
    prev_edges = None

    for k, idx in enumerate(frames):
        t0 = time.perf_counter()
        s = CarbonStructure.from_LAMMPS_file(
            traj, frame_index=idx, atom_type_hash=args.atom_type,
            bond_dist=BOND_DIST, bond_tol=BOND_TOL,
        )
        edges = edge_set(s)
        row: dict = {'frame': k, 'timestep': int(traj[idx]['timestep'])} # the LAMMPS step of the frame
        row.update(cheap_row(s))
        row['bond_turnover'] = len(edges ^ prev_edges) / len(edges) if prev_edges else 0.0
        prev_edges = edges

        # expensive tier: first and last frame, and every --full-every frames
        do_full = (idx in (frames[0], last)) or (
            args.full_every and k % args.full_every == 0
        )
        if do_full:
            print(f'  frame {k}: full post-process ...', flush=True)
            row.update(full_row(s, out_dir, tag=f'f{k:04d}'))

        row['seconds'] = round(time.perf_counter() - t0, 1)
        rows.append(row)

        # rewrite the whole CSV each frame: a killed job still leaves usable output
        fieldnames = list(dict.fromkeys(
            [key for r in rows for key in r]
        ))
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, restval='')
            w.writeheader()
            w.writerows(rows)

        print(f'  frame {k}/{len(frames)-1} '
              f'3-coord {row["coord_3"]:.1f}%  ({row["seconds"]} s)', flush=True)

    print(f'\nwrote {out_csv} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
