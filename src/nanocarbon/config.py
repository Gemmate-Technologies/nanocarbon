# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
GenerationConfig: every parameter of one generation run and its post-processing.

One dataclass instead of scattered keyword arguments, so a run reads at a glance, is saved
to JSON next to its results (each output carries its own provenance), and travels to a
worker process as a single object.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class GenerationConfig:
    """
    Parameters of one generation run: pores, carbon growth, potentials, post-processing.

    The run identity (batch._hash_config) is a hash of every field except run_id and out_dir,
    defaults included: adding a field, or changing a default, gives every config that relies
    on that default a new run_id, so existing batch and PSO runs stop resuming.

    Parameters
    ----------
    box_size : float
        Edge of the cubic box, Angstrom.
    phi : float
        Pore volume fraction placed before carbon growth, in (0, 1). Growth fills part of the
        pores, so the porosity measured on the result is lower.
    pore_mu, pore_sigma : float
        Mean and standard deviation of the Gaussian pore RADIUS, Angstrom; ignored when
        psd_file is set.
    pore_dx : float
        Voxel edge of the pore-occupancy grid, Angstrom; smaller is more accurate and slower.
    max_rho : float
        Target carbon density, g/cm^3.
    target_density_deviation : float
        Accepted relative error on max_rho (0.05 = 5 %); growth is rerun until it is met or
        max_carbon_attempts runs are done.
    pp_N_iter_porosity : int
        Monte-Carlo samples of compute_porosity.
    pp_N_iter_psd : int
        Monte-Carlo samples of compute_psd.
    pp_voxel_len : float
        Voxel edge of compute_porosity, compute_psd and compute_AVA, Angstrom.
    pp_rdf_bin : float
        Bin width of the RDF, Angstrom.
    rdf_local_n_samples : int or None
        None: the run stores only the whole-structure RDF, and the objective scores it. An
        int: the run also stores that many per-sphere RDFs (post_process._local_rdf_stack),
        and the objective scores the worst quartile of them, so the least-ordered pockets
        drive the cost instead of the box average.
    rdf_local_sphere_radius : float
        Radius of the local-RDF spheres, Angstrom.
    pp_seed : int or None
        Seed of the stochastic post-processing (PSD, porosity, local-RDF spheres): the same
        structure and seed give the same numbers. None draws a fresh seed. Generation itself
        is not seeded.
    pp_workers : int
        KD-tree threads of compute_psd and compute_AVA. 1 is deterministic and avoids
        oversubscription when a batch runs one structure per core; -1 suits one large
        analysis on an idle node, at the cost of bit-reproducibility.
    r_C : float
        Carbon radius of porosity, PSD, AVA and connectivity, Angstrom.
    r_N : float
        Probe (nitrogen) radius of the same, Angstrom.
    generator_type : {"ring", "random"}
        RingGenerator (ring-by-ring growth) or RandomGenerator (random sequential addition,
        the unstructured baseline; its runs skip post-processing, see run_generation).
    random_min_dist, random_pore_standoff : float
        RandomGenerator min_dist and pore_standoff, Angstrom.
    merge_coeff, reject_coeff : float
        RingGenerator merge and reject distances, in units of bond_dist. The defaults here
        (0.4, 1.2) differ from the RingGenerator constructor's (0.3, 0.8) on purpose: keep
        reject_coeff >= bond_tol, or a contact the growth accepted unbonded becomes a bond
        when the structure is re-bonded at the end. A RingGenerator built directly gets the
        looser 0.8.
    bond_dist : float
        C-C bond length of the growth, Angstrom.
    bond_tol : float
        End-of-growth re-bond: pairs closer than bond_dist * bond_tol are bonded.
    E_max : float
        Largest energy accepted for a placed ring, in the atoms potential's unit.
    ring_atoms_per_ring, ring_probabilities : list or None
        Ring sizes and their probabilities (equal lengths, summing to 1). None for both uses
        the RingGenerator defaults, [5, 6, 7, 8] with [0.15, 0.7, 0.10, 0.05].
    force_defect_pairs : bool
        RingGenerator force_defect_pairs (5-7 defect pairs).
    lj_exclusion_bondnum : int
        RingGenerator LJ_exclusion_bondnum: atoms within this many bonds of the anchors leave
        the 12-6 LJ (harmonic) or the whole energy (tersoff). The default here (15) differs
        from the constructor's (5).
    atoms_potential_type : {"harmonic", "tersoff"}
        Atom-atom potential: harmonic bonds, angles and dihedrals with LJ, or Tersoff bond
        order.
    pores_potential_type : {"lj"}
        Atom-pore potential. lj is singular at the pore surface, so carbon never rests on it
        (which a LAMMPS `fix wall/region ... side out` refuses to start from), and LAMMPS'
        lj126 wall is the same function, so the anneal wall reproduces the generation wall.
    atoms_potential_params, pores_potential_params : dict or None
        Keyword arguments of the potential's set_potential_parameters; None keeps the class
        defaults. RandomGenerator ignores the potential fields.
    psd_file : str or None
        Path of a tabulated PSD, a (K, 2) .npy of diameter (Angstrom) and value
        (utilities/digitise_psd.py writes one); when set, pore_mu and pore_sigma are ignored.
    psd_label : str
        Label of the PSD condition, used in output directory names.
    psd_kind : {"volume", "count"}
        Convention of psd_file: "volume" for a volume distribution dV/dD (what adsorption
        and pore-analysis tools report; converted so the placed pores reproduce the volume
        per size), "count" for a number distribution. Ignored for the Gaussian.
    phi_tolerance : float
        Relative tolerance of the pore-only pre-screen on the porosity against phi. Loose on
        purpose: growth moves the porosity anyway, so the pre-screen only filters failed
        packings.
    max_pore_attempts : int
        Pore packings tried by the pre-screen before keeping the best; 1 disables it.
    max_carbon_attempts : int
        Carbon growths tried before keeping the best (CarbonStructure.generate).
    out_dir : str
        Output directory of this run; set by the batch.
    run_id : str
        Run identity "{hash8}_r{replica}"; set by the batch, do not set by hand.
    """

    # box
    box_size: float = 50.0

    # pores
    phi: float = 0.2
    pore_mu: float = 10.0
    pore_sigma: float = 1.0
    pore_dx: float = 0.5

    # carbon
    max_rho: float = 1.2
    target_density_deviation: float = 0.05

    # post-processing
    pp_N_iter_porosity: int = 200_000
    pp_N_iter_psd: int = 5_000
    pp_voxel_len: float = 0.5
    pp_rdf_bin: float = 0.5
    rdf_local_n_samples: int | None = None
    rdf_local_sphere_radius: float = 7.0
    pp_seed: int | None = None
    pp_workers: int = 1
    r_C: float = 1.71
    r_N: float = 1.55

    # generator
    generator_type: str = "ring"
    random_min_dist: float = 1.5
    random_pore_standoff: float = 0.75
    merge_coeff: float = 0.4
    reject_coeff: float = 1.2
    bond_dist: float = 1.42
    bond_tol: float = 1.2
    E_max: float = 1e3

    # ring sizes and exclusions (RingGenerator only)
    ring_atoms_per_ring: list[int] | None = None
    ring_probabilities: list[float] | None = None
    force_defect_pairs: bool = True
    lj_exclusion_bondnum: int = 15

    # potentials (RingGenerator only)
    atoms_potential_type: str = "harmonic"
    pores_potential_type: str = "lj"
    atoms_potential_params: dict | None = None
    pores_potential_params: dict | None = None

    # PSD source
    psd_file: str | None = None
    psd_label: str = "gaussian"
    psd_kind: str = "volume"

    # pre-screen and retries
    phi_tolerance: float = 0.15
    max_pore_attempts: int = 20
    max_carbon_attempts: int = 10

    # bookkeeping (set by the batch)
    out_dir: str = 'parallel_output'
    run_id: str = ''

    @property
    def box(self) -> list[float]:
        """Box edges [L, L, L], Angstrom, as PoreStructure and the generators take them."""
        return [self.box_size, self.box_size, self.box_size]

    @property
    def config_hash(self) -> str:
        """The hash part of run_id, shared by the replicas of one configuration."""
        return self.run_id.split('_r')[0]

    def to_dict(self) -> dict:
        """
        All fields as a plain dict.

        Returns
        -------
        dict
            {field name: value}, JSON-serializable.
        """
        return asdict(self)

    def to_json(self, path: str) -> None:
        """
        Write all fields to a JSON file, for provenance.

        Parameters
        ----------
        path : str
            Output file.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
