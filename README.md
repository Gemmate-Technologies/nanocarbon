# nanocarbon

**nanocarbon** builds atomistic models of amorphous nanoporous carbon with explicit pores,
and measures their structure. A structure is specified by a pore-size distribution (PSD), a
pore volume fraction and a carbon density: pores are placed first, drawn from the target PSD
up to the target pore fraction; carbon is then grown ring by ring around them up to the
target density, and the structure is written out for a molecular-dynamics anneal (LAMMPS),
with the pores kept as walls. These properties are therefore *inputs* of the model, not
something read off a melt-quench afterwards. The same post-processing measures any
structure, constructed or imported from an MD trajectory: pore-size distribution, porosity,
accessible surface and volume, radial distribution function, ring statistics, coordination
and pore connectivity. A particle-swarm calibration searches the generation parameters that
reproduce experimental targets: density, PSD, porosity, accessible surface area, pore volume
and radial distribution function.

A manuscript describing the method, *Explicit pore pre-conditioning and inverse design of
amorphous nanoporous carbon*, is in preparation (November 2026).

## Installation

```bash
pip install nanocarbon
```

Python 3.11 or later. The library depends on `numpy`, `scipy`, `scikit-learn` and
`scikit-image` only. For the examples and notebooks:

```bash
git clone https://github.com/Gemmate-Technologies/nanocarbon.git
cd nanocarbon
uv sync              # library + matplotlib, pandas and Jupyter in .venv
```

## Quick start

```python
from nanocarbon import GenerationConfig, run_generation, CarbonStructure
from nanocarbon import post_process as pp, io_functions as io

# Generate one structure and post-process it
cfg = GenerationConfig(box_size=30.0, phi=0.2, pore_mu=5.0, pore_sigma=1.0, max_rho=0.9)
result, structure, pores = run_generation(cfg)
print(result.atoms_number, round(result.achieved_density, 2), result.ring_histogram)
structure.print_LAMMPS("my_structure")          # my_structure.data

# Analyze a frame of an MD trajectory
traj = io.import_from_LAMMPS("use_cases/data/harmonic_trajectory.lammpstrj")
s = CarbonStructure.from_LAMMPS_file(traj, frame_index=-1)
print(pp.compute_porosity(s), pp.count_coordination_number(s)["3 bonds"])
```

Run it from a clone of the repository (the trajectory ships in `use_cases/data/`); the
generation takes about a minute and a half on a laptop. Every function and class carries a
NumPy-style docstring with shapes and units; lengths are in Angstrom throughout.
`use_cases/` holds one runnable example per feature (generation, import, each
post-processing function, a batch, a calibration), listed under
[Utilities and examples](#utilities-and-examples).

## Generation

**Pores.** `PoreStructure.generate` packs spheres at random positions up to a target pore
fraction `phi`, with radii from a Gaussian or a tabulated PSD (`.npy`, diameter in Angstrom
and value). A tabulated PSD is read as a volume distribution dV/dD by default
(`psd_kind="volume"`), the convention of adsorption measurements; pass `psd_kind="count"`
for a number distribution.

**Carbon.** `RingGenerator` grows carbon ring by ring from existing bonds: each candidate
ring is oriented by a one-dimensional energy search and accepted if it clears an energy cap
(`E_max`), the pore exclusion and a merge/reject distance window, so growing fronts stitch
into planes. Ring sizes are drawn from a configurable distribution, with 5-7 defect pairs.
`RandomGenerator` places single atoms at random under a minimum distance, as an
unstructured baseline. `CarbonStructure.generate` reruns a generator until the target
density is met and stores the bonds found by distance (pairs closer than 1.704 Angstrom).

**Potentials.** The energy that gates each ring sums an atom-atom and an atom-pore term:

| Class | Interaction |
|---|---|
| `HarmonicPotential` | Harmonic bonds, angles and dihedrals, plus 12-6 LJ and growth aids (kcal/mol) |
| `TersoffPotential` | Tersoff bond order, used as a placement criterion (eV) |
| `LJPoresPotential` | 12-6 LJ wall on the distance to each pore surface (unit of the atoms potential) |

`GenerationConfig` holds every parameter of a run (its docstring documents each field), and
`run_generation(cfg)` runs pores, carbon and post-processing in one call, with no file
I/O. The object-level path (generators and potentials built by hand) is shown in
`use_cases/generate_*.py`.

**To LAMMPS.** `CarbonStructure.print_LAMMPS` writes a `.data` file; `PoreStructure.print_LAMMPS`
writes each pore as a spherical `wall/region` LJ wall, cut at its minimum so it only repels
(WCA). Pass the growth's `A` (converted to the deck's energy unit,
`io_functions.WALL_UNIT_TO_EV`) and `B` as `LJ_vars` to get the wall the structure was grown
against; `run_batch` does this for you. For a reactive anneal (EDIP, Tersoff),
`utilities/convert_to_atomic.py` rewrites stored structures in the bond-free `atomic` style
with the wall depth in eV, and `utilities/anneal.in` is a starting deck.

## Analysis

| Function | Measures | Method |
|---|---|---|
| `compute_psd` | Pore-size distribution | Largest empty sphere containing each point ([Gelb & Gubbins 1999](https://doi.org/10.1021/la9808418); [Bhattacharya & Gubbins 2006](https://doi.org/10.1021/la052651k)), Monte-Carlo on a voxel grid |
| `compute_porosity` | Void fraction | Monte-Carlo; probe-center fraction by default, geometric with `r_probe=0` |
| `compute_AVA` | Accessible surface area, probe-center and probe-occupiable volumes | Voxel grid, face-connected flood fill from the box faces, marching cubes |
| `radial_distribution_function` | g(r) | Exact pair counts with a periodic KD-tree |
| `count_rings` | Ring-size histogram | Primitive rings, those with no shortcut path between two of their atoms ([Franzblau 1991](https://doi.org/10.1103/PhysRevB.44.4925)) |
| `count_coordination_number` | Bonds per atom | Stored bond graph |
| `compute_connectivity` | Mean links per pore | Voronoi pore network with a probe bottleneck test |

Void measures and pair counts are periodic. Accessibility (`compute_AVA`) is not: the box
faces stand for the outside, so a pore enclosed inside the box does not count as accessible.
Each channel except the connectivity was checked against an independent tool on QMD
reference frames: PSD, surface and volumes against PoreBlazer v4.0
([Sarkisov et al. 2020](https://doi.org/10.1021/acs.chemmater.0c03575)), g(r) against OVITO
([Stukowski 2010](https://doi.org/10.1088/0965-0393/18/1/015012)), coordination against
MDAnalysis ([Michaud-Agrawal et al. 2011](https://doi.org/10.1002/jcc.21787)) and rings
against R.I.N.G.S. ([Le Roux & Jund 2010](https://doi.org/10.1016/j.commatsci.2010.04.023)).
`use_cases/compute_psd.py` and `compute_rdf.py` plot the results next to PoreBlazer and OVITO
references for the same frame.

## Batches and calibration

Two command-line verbs read a Python config file:

```bash
python -m nanocarbon generate utilities/batch_config.py      # a grid of runs, in parallel
python -m nanocarbon optimize utilities/calibrate_config.py  # PSO calibration
```

`generate` expands a `CONDITIONS` grid over a base `GenerationConfig` and runs one structure
per process. Each run writes its configuration, results, arrays, LAMMPS files and JSON
structures; a resubmitted batch skips the runs already on disk, and a run that fails is
recorded with `status='generation_error'` instead of stopping the batch. Load a batch with
`BatchResults.from_run_dir(out_root)`. `optimize` searches the parameters in `PARAM_BOUNDS`
so that the generated structures match the targets in `ANCHORS` (density, PSD, porosity,
surface area, pore volume, RDF); a polynomial surrogate can propose one extra candidate per
iteration. The calibration checkpoints every iteration and resumes from `pso_state.json`.

## Utilities and examples

| `utilities/` | |
|---|---|
| `batch_config.py`, `calibrate_config.py` | Config templates for `generate` and `optimize` |
| `run_parallel.slurm` | Runs a batch on one SLURM node |
| `convert_to_atomic.py` | Stored structures to atomic-style LAMMPS input, wall depth in eV |
| `anneal.in` | LAMMPS EDIP anneal deck for the converted structures |
| `analyze_anneal.py` | Anneal trajectory to CSV: coordination every frame, full analysis at the endpoints |
| `digitise_psd.py` | WebPlotDigitizer CSV export to a PSD `.npy` |
| `pso_analysis.ipynb` | Which parameters a calibration constrains, from its search history |

| `use_cases/` | |
|---|---|
| `generate_default.py`, `generate_custom_potentials.py` | One structure, default or custom potentials |
| `generate_random.py` | A random-generator structure for a LAMMPS quench |
| `load_from_lammps.py`, `load_from_json.py`, `write_lammps_data.py` | Structure input and output |
| `compute_psd.py`, `compute_rdf.py`, `compute_AVA.py`, `compute_porosity.py` | One post-processing function each |
| `compute_connectivity.py`, `compute_rings_coordination.py` | Same |
| `batch_gaussian_psd.py`, `batch_tabular_psd.py` | Small batches on a laptop |
| `optimize_gaussian_psd.py` | A calibration in a few minutes |

The examples run from any directory, read `use_cases/data/` and write to `use_cases/output/`.

## Contributing

Bug reports, questions and suggestions are welcome as GitHub issues or by email to the authors.
Pull requests are not accepted.

## Authors

- Matteo Alberghini (matteo.alberghini@gemmate-technologies.com)
- Giacomo Pierotti (giacomo.pierotti@gemmate-technologies.com)

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),
copyright (c) 2026 Gemmate Technologies s.r.l. See
[LICENSE](https://github.com/Gemmate-Technologies/nanocarbon/blob/main/LICENSE). For commercial
use, contact Gemmate Technologies.
