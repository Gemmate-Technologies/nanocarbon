# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Convert the generation JSONs of a folder into LAMMPS input for a reactive anneal.

Usage: python utilities/convert_to_atomic.py <folder> [-o outdir] [--wall-units auto|kcal/mol|eV]

A batch writes 'full'-style .data files (with a Bonds section); reactive decks (EDIP,
Tersoff) need the 'atomic' style: they rebuild the bonds from the geometry, and a stale bond
list forces a wide comm cutoff or aborts once atoms move. Each carbon_*.json is reloaded and
written as an atomic .data with the same stem, and each pores_*.json as a .pores_settings
with the WCA cutoff and epsilon in eV. Output goes to <folder>/atomic/ unless -o is given,
so the original files are not overwritten.

The pore-wall depth A carries the energy unit of the atoms potential it was summed with
(kcal/mol for harmonic, eV for tersoff): --wall-units auto (default) reads the potential
from each carbon json; kcal/mol or eV forces the conversion for a json that does not record
it.
"""
import argparse
import json
from pathlib import Path

from nanocarbon import CarbonStructure
from nanocarbon.structures import PoreStructure
from nanocarbon.io_functions import print_LAMMPS_structure, LJ_WALL_MINIMUM, WALL_UNIT_TO_EV

# The pore-wall A carries the atoms potential's energy unit (kcal/mol for harmonic, eV for
# tersoff); the library holds the one conversion map. --wall-units picks the branch by hand.
KCAL_PER_MOL_TO_EV = WALL_UNIT_TO_EV['harmonic']

def main() -> None:
    """Convert every carbon_*.json and pores_*.json of the folder given on the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', type=Path, help='folder of carbon_*.json structures')
    ap.add_argument('-o', '--outdir', type=Path, default=None,
                    help='output folder (default: <folder>/atomic)')
    ap.add_argument('--wall-units', choices=['auto', 'kcal/mol', 'eV'], default='auto',
                    help="energy unit of the generation pore-wall A. 'auto' (default) reads "
                         "the atoms potential from each carbon json: harmonic->kcal/mol, "
                         "tersoff->eV. Pass kcal/mol or eV to force the branch.")
    args = ap.parse_args()

    outdir = args.outdir or args.folder / 'atomic'
    outdir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(args.folder.glob('carbon_*.json'))
    assert jsons, f'no carbon_*.json files in {args.folder}'

    for jp in jsons:
        s = CarbonStructure.from_dict(json.loads(jp.read_text()))
        name = str(outdir / jp.stem)          # print_LAMMPS_structure appends '.data'
        print_LAMMPS_structure(Box=s.Box.tolist(), coord=s.coordinates[:, :3],
                               bonds=None, write_bonds=False, name=name)
        print(f'{jp.name} -> {name}.data  ({s.coordinates.shape[0]} atoms, atomic)')

    # Pores: re-emit each .pores_settings with the WCA cutoff and epsilon in eV. PoreStructure
    # stores only geometry, so the generation (A, B) come off the paired carbon json -- both
    # must be in the folder. The cutoff mirrors batch._wall_params (WCA); we also convert eps.
    pore_jsons = sorted(args.folder.glob('pores_*.json'))
    assert pore_jsons, f'no pores_*.json files in {args.folder}'

    for pp in pore_jsons:
        run_id = pp.stem[len('pores_'):]
        cp = args.folder / f'carbon_{run_id}.json'
        assert cp.exists(), f'{pp.name} has no paired carbon_{run_id}.json (needed for A, B)'

        cj = json.loads(cp.read_text())
        pot = cj.get('pores_potential')
        assert pot, f'{cp.name} carries no pores_potential (A, B); regenerate it or pass -o'
        p = pot['dict_params']                       # {A, B, e1, e2}
        e1, e2 = p.get('e1', 12), p.get('e2', 6)
        LJ_type = {(12, 6): '126', (9, 3): '93'}[(e1, e2)]

        # epsilon -> eV on the atoms potential's energy scale; cutoff at the wall's own minimum
        # (WCA). LJ_WALL_MINIMUM is the same map batch._wall_params uses, so the 9-3 wall gets
        # (2/5)^(1/6) rather than the 12-6 value.
        if args.wall_units == 'eV':
            eps_to_ev = 1.0
        elif args.wall_units == 'kcal/mol':
            eps_to_ev = KCAL_PER_MOL_TO_EV
        else:                                        # auto: the unit follows the atoms potential
            atoms_type = (cj.get('atoms_potential') or {}).get('pot_type')
            assert atoms_type in WALL_UNIT_TO_EV, (
                f'{cp.name}: atoms_potential={atoms_type!r}, cannot pick wall-eps units '
                f'automatically. Pass --wall-units kcal/mol|eV.')
            eps_to_ev = WALL_UNIT_TO_EV[atoms_type]
        LJ_vars = [p['A'] * eps_to_ev, p['B'], LJ_WALL_MINIMUM[LJ_type] * p['B']]

        pores = PoreStructure.from_dict(json.loads(pp.read_text()))
        name = str(outdir / pp.stem)                 # print_LAMMPS_pores appends '.pores_settings'
        pores.print_LAMMPS(LJ_vars=LJ_vars, LJ_type=LJ_type, name=name)
        basis = {1.0: 'eV', KCAL_PER_MOL_TO_EV: 'kcal/mol'}.get(eps_to_ev, f'x{eps_to_ev:.4g}')
        print(f'{pp.name} -> {name}.pores_settings  '
              f'(A {p["A"]:.3f} {basis} -> eps {LJ_vars[0]:.3f} eV, '
              f'sigma {LJ_vars[1]:.3f}, cutoff {LJ_vars[2]:.3f} A)')


if __name__ == '__main__':
    main()
