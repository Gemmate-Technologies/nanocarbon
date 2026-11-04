# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Generators: pore packing and carbon growth.

PoresGenerator packs spherical pores to a target porosity, with radii drawn from a Gaussian or
a tabulated PSD. CarbonGenerator is the base of the carbon generators (box, target density,
pores, neighbor cells). RingGenerator grows carbon ring by ring around the pores, orienting
each ring by an energy line search gated by E_max; RandomGenerator places single atoms by
random sequential addition, with rollback when jammed.
"""
import numpy as np
from numpy.typing import NDArray

from .potential_classes import HarmonicPotential, TersoffPotential, LJPoresPotential
from .auxiliary_classes import NeighborhoodManager
from . import geometry_functions as gf

class PoresGenerator:
    """
    Random packing of spherical pores up to a target porosity.

    Spheres are dropped at uniform random positions, with radii drawn from a Gaussian or a
    tabulated PSD; a sphere is skipped if it is wider than the box or adds too little new void.
    Occupancy is tracked on a periodic voxel grid.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    phi : float
        Target pore volume fraction, > 0.
    periodic : list of int, optional
        Periodic flags per axis. Default [1, 1, 1].
    dx : float, optional
        Voxel edge of the porosity bookkeeping, Angstrom. Default min(Box) / 100.
    psd : (K, 2) ndarray, optional
        Tabulated PSD: pore diameters (Angstrom) in column 0, distribution values in column 1.
        When given, mu and sigma are ignored.
    mu : float, optional
        Mean pore RADIUS of the Gaussian PSD, Angstrom. Default 10.
    sigma : float, optional
        Standard deviation of the Gaussian radius, Angstrom. Default 1.
    psd_kind : {"volume", "count"}, optional
        Convention of the tabulated PSD. "volume" (default) is a volume distribution dV/dD, as
        adsorption experiments and pore-analysis tools report: it is divided by R^3 before
        sampling, so the placed spheres reproduce the pore volume per size. "count" samples the
        curve directly as a number distribution.
    acceptance_threshold : float, optional
        Fraction of a new sphere's voxels that must be new void for it to be accepted; 0.01
        asks for more than 1 %. Default 0.

    Raises
    ------
    ValueError
        On a non-positive box or phi, bad periodic flags, a negative threshold or an unknown
        psd_kind.
    """
    def __init__(self, 
                 Box: list[float], 
                 phi: float, 
                 periodic: list[int] = [1, 1, 1], 
                 dx: float | None = None, 
                 psd: NDArray | None = None, 
                 mu: float = 10.0, 
                 sigma: float = 1.0,
                 psd_kind: str = "volume",
                 acceptance_threshold: float = 0.0) -> None:
        if not (all(b>0 for b in Box)):
            raise ValueError("Box lengths must be positive")
        if not (0 < phi):
            raise ValueError("configuration porosity must be positive")
        if not (len(periodic) == 3 and all(p in [0, 1] for p in periodic)):
            raise ValueError("x, y, z periodicity must be [0, 1]")
        if not (acceptance_threshold >= 0.0):
            raise ValueError("Acceptance threshold must be non-negative")

        if psd_kind not in ["count", "volume"]:
            raise ValueError(f"invalid PSD normalizatin convention, psd_kind must be 'volume' or 'count', got {psd_kind!r}")

        # results
        self.coord = np.empty((0, 3), dtype=float) 
        self.R = np.array([], dtype=float)
        self.phi_num = None
        self.acceptance_threshold = acceptance_threshold
        self.psd_kind = psd_kind

        # parameters
        self.Box = np.array(Box)
        self.phi = phi
        self.periodic = np.array(periodic)

        # radius sampler: Gaussian or tabulated PSD
        self._setPSD(mu = mu, sigma = sigma, func = psd, psd_kind = self.psd_kind)
        # voxel grid of the porosity bookkeeping
        if dx is None:
            dx = float(np.min(Box)/100)

        self.voxel_volume = dx**3
        self.voxel_centers, _ , _ = gf.build_voxel_centers(self.Box, dx)
        self.occupied = np.zeros(self.voxel_centers.shape[0], dtype=bool)

    def _setPSD(self, 
                mu: float = 10, 
                sigma: float = 1, 
                func: NDArray | None = None, 
                psd_kind: str = "volume") -> None:
        """
        Set up the radius sampler: a Gaussian, or the inverse CDF of a tabulated PSD.

        Parameters
        ----------
        mu : float, optional
            Mean radius of the Gaussian, Angstrom.
        sigma : float, optional
            Standard deviation of the Gaussian, Angstrom.
        func : (K, 2) ndarray, optional
            Tabulated PSD (diameter in Angstrom, value); takes precedence over mu and sigma.
            Diameters must be strictly increasing and values non-negative.
        psd_kind : {"volume", "count"}, optional
            Convention of func; "volume" divides the values by R^3 to get a count density.

        Raises
        ------
        ValueError
            On a bad psd_kind, negative values, non-increasing diameters, or neither a table nor
            (mu, sigma).
        """
        if func is not None:
            R = func[:, 0]/2 # the table holds diameters
            p = func[:, 1]

            if psd_kind == "volume":
                p_count = np.zeros_like(p, dtype=float)
                np.divide(p, R ** 3, out=p_count, where=R > 0)  # R=0 bin holds no pores
                p = p_count
            elif psd_kind != "count":
                raise ValueError(f"psd_kind must be 'volume' or 'count', got {psd_kind!r}")

            if np.any(p < 0):
                raise ValueError("PSD contains negative probabilities")
            if not np.all(np.diff(R) > 0):
                raise ValueError("Radii must be strictly increasing")

            # Build cumulative probability distributions
            cdf = np.cumsum(p * np.gradient(R))
            cdf /= cdf[-1]  # normalize: the CDF ends at 1

            self._psd_R = R 
            self._psd_cdf = cdf
            self._psd_type = "tabulated"
            return

        if mu is None or sigma is None:
            raise ValueError("Provide either psd or (mu, sigma)")

        self._psd_mu = mu
        self._psd_sigma = sigma
        self._psd_type = "gaussian"

    def _getSphere(self) -> float:
        """
        Draw one pore radius from the PSD.

        Returns
        -------
        float
            Radius, Angstrom; Gaussian draws are repeated until positive.
        """
        if self._psd_type == "gaussian":
            R = -1.0
            while R <= 0.0:
                R = np.random.normal(self._psd_mu, self._psd_sigma)
            return R

        elif self._psd_type == "tabulated":
            u = np.random.rand()
            return np.interp(u, self._psd_cdf, self._psd_R)

        else:
            raise RuntimeError("PSD not correctly initialized")

    def _generate(self) -> tuple[NDArray, NDArray, float]:
        """
        Pack spheres until the voxel porosity reaches phi, or 1e6 draws have been rejected.

        Returns
        -------
        coord : (P, 3) ndarray
            Pore centers, Angstrom.
        R : (P, 1) ndarray
            Pore radii, Angstrom.
        phi_num : float
            Porosity reached on the voxel grid.
        """
        max_rejects = 1e6
        rejects = 0

        phi_num = 0.0 
        R = []
        coord = []
        while (phi_num < self.phi) and (rejects < max_rejects):
            R_next = self._getSphere()

            if (np.min(self.Box) < 2*R_next):
                rejects += 1
                continue

            coord_next = np.array([np.random.random()*box for box in self.Box])
            # fraction of the sphere that is new void; rejected if too low
            frac_new = self._add_sphere_voxelized(coord_next, R_next)

            if frac_new <= self.acceptance_threshold:
                rejects += 1 #avoids infinite loop
                continue

            R.append(R_next)
            coord.append(coord_next)

            phi_num = np.sum(self.occupied) * self.voxel_volume / np.prod(self.Box)

        self.phi_num = phi_num
        self.R = np.array(R, dtype = float).reshape(-1,1)
        self.coord = np.array(coord, dtype = float)

        return self.coord, self.R, self.phi_num

    def _voxels_inside_sphere(self, 
                              coord_single: NDArray, 
                              R_single: float) -> NDArray[np.bool]:
        """
        Voxels whose centers lie inside a sphere (minimum image along the periodic axes).

        Parameters
        ----------
        coord_single : (3,) ndarray
            Sphere center, Angstrom.
        R_single : float
            Sphere radius, Angstrom.

        Returns
        -------
        (Nvox,) ndarray of bool
            True for the voxels inside the sphere.
        """

        delta = self.voxel_centers - coord_single
        delta -= self.Box*self.periodic*np.round(delta/self.Box)

        dist = np.linalg.norm(delta, axis=1)

        return dist <= R_single

    def _add_sphere_voxelized(self, 
                              coord_single: NDArray, 
                              R_single: float) -> float:
        """
        Add a sphere to the occupancy grid if enough of it is new void.

        Parameters
        ----------
        coord_single : (3,) ndarray
            Sphere center, Angstrom.
        R_single : float
            Sphere radius, Angstrom.

        Returns
        -------
        float
            Fraction of the sphere's voxels newly occupied; 0 when the sphere is rejected (fraction
            not above acceptance_threshold, or no voxel inside), and the grid is then unchanged.
        """
        mask_voxels_included = self._voxels_inside_sphere(coord_single, R_single)

        # Only count voxels not already filled
        new_voxels = np.logical_and(mask_voxels_included, np.logical_not(self.occupied))
        # a sphere smaller than a voxel may hold no voxel center
        if np.sum(mask_voxels_included) == 0:
            return 0.0

        frac_new = np.sum(new_voxels) / np.sum(mask_voxels_included)

        if frac_new <= self.acceptance_threshold:
            return 0.0

        # Update occupancy
        self.occupied = np.logical_or(mask_voxels_included, self.occupied)
        return frac_new

class CarbonGenerator:
    """
    Base class of the carbon generators: box, target density, pores and neighbor cells.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom; ignored when pores_structure is given. The default [0, 0, 0] is
        invalid on purpose: pass Box or pores_structure.
    max_rho : float
        Target carbon density, g/cm^3; must be > 0 (the default 0 is invalid on purpose).
    periodic : list of int, optional
        Periodic flags per axis; ignored when pores_structure is given. Default [1, 1, 1].
    N_box : int, optional
        Neighbor-grid cells per axis. Default geometry_functions.set_N_box(Box).
    pores_structure : PoreStructure, optional
        Pores to grow around; its box and periodicity replace Box and periodic.

    Raises
    ------
    ValueError
        On a non-positive density or box, or bad periodic flags.
    """
    def __init__(self, 
                 Box: list[float] = [0., 0., 0.], # invalid on purpose: pass Box or pores_structure
                 max_rho: float = 0, # invalid on purpose: must be > 0
                 periodic: list[int] = [1, 1, 1],
                 N_box: int | None = None,
                 pores_structure = None
                 ) -> None:
        if not (0 < max_rho):
            raise ValueError("configuration density must be positive")

        if pores_structure is None:
            if not (all(b>0 for b in Box)):
                raise ValueError("Box lengths must be positive")
            if not (len(periodic) == 3 and all(p in [0, 1] for p in periodic)):
                raise ValueError("x, y, z periodicity must be [0, 1]")

            self.periodic = np.array(periodic)
            self.Box = np.array(Box)
            self._set_pores(None, None)
        else:
            self.Box = pores_structure._Box
            self.periodic = pores_structure._periodic
            self._set_pores(pores_structure._coord, pores_structure._R)

        if N_box is None:
            N_box = gf.set_N_box(self.Box)
        self.N_box = N_box

        self.max_rho = max_rho

        self.max_atoms = int(np.round(self.max_rho*(np.prod(self.Box)*1e-24)*6.02214076e23/12.0107))
        # no potential
        self._atoms_potential_class = None
        self._pores_potential_class = None

        self._resetVariables()

    def _set_pores(self, 
                   pores_coord: NDArray | None , 
                   pores_radius: NDArray | None ) -> None:

        """
        Store the pores (none when both are None) and drop the current pore potential.

        Parameters
        ----------
        pores_coord : (P, 3) ndarray or None
            Pore centers, Angstrom.
        pores_radius : (P,) or (P, 1) ndarray or None
            Pore radii, Angstrom.
        """
        if pores_coord is None:
            self.coord_pores = np.empty((0,3), dtype=float)
        else:
            self.coord_pores = pores_coord

        if pores_radius is None:
            self.R_pores = np.empty(0, dtype=float)
        else:
            self.R_pores = pores_radius

        self._pores_potential_class = None

    def _resetVariables(self) -> None:
        """Clear the atoms and the neighbor cells."""
        self._coord = np.empty((0,3), dtype = float)
        self._neigh_boxes = NeighborhoodManager(Box = list(self.Box), periodic = list(self.periodic))

    def _check_pore_constraint(self, 
                               atoms_coord: NDArray,
                               standoff: float = 0.0) -> bool :
        """
        True if no atom lies inside a pore, or within standoff of its surface.

        Parameters
        ----------
        atoms_coord : (N, 3) or (3,) ndarray
            Atom coordinates, Angstrom.
        standoff : float, optional
            Extra clearance beyond each pore radius, Angstrom. Default 0.

        Returns
        -------
        bool
            True if every atom is admissible.
        """
        if self.coord_pores is None:
            return True

        for atom in np.atleast_2d(atoms_coord):
            distances = gf.distance(
                atom,
                self.coord_pores,
                self.Box,
                self.periodic
            )
            if np.any(distances < self.R_pores.ravel() + standoff):
                return False

        return True

    @property
    def pores_potential_class(self):
        """The atom-pore potential in use (set it with set_pores_potential)."""
        return self._pores_potential_class

    @property
    def coordinates(self):
        """Atom coordinates, (N, 3) ndarray, Angstrom."""
        return self._coord

    @property
    def atoms_number(self):
        """Number of atoms placed."""
        return self._coord.shape[0]

    @property
    def density(self):
        """Carbon density, g/cm^3 (all atoms carbon, 12.0107 g/mol)."""
        return self.atoms_number*12.0107/6.02214076e23/(np.prod(self.Box)*1e-24)

    # no bond graph in the base class; RingGenerator overrides these
    @property
    def adjacency(self):
        """Bond graph {atom: set of bonded atoms}; empty in the base class."""
        return {}

    @property
    def bonds(self):
        """Bonds, (M, 2) int ndarray; empty in the base class."""
        return np.empty((0, 2), dtype=int)

    def _generate(self, *args, **kwargs):
        """
        Build the structure; implemented by each subclass.

        Parameters
        ----------
        *args, **kwargs
            The subclass's arguments.
        """
        raise NotImplementedError("Super-Class does not include a generation method")

class RingGenerator(CarbonGenerator):
    """
    Grows carbon ring by ring from the existing atoms, around the pores.

    Each step picks an unsaturated anchor, builds a ring of sampled size on one of its bonds,
    orients it by a one-dimensional line search on the energy (atoms potential + pore wall), and
    accepts it if it clears E_max, the pore exclusion and the merge/reject distance window. Ring
    topology and the 5-7 defect bookkeeping are stored alongside the coordinates.

    Parameters
    ----------
    Box, max_rho, periodic, N_box, pores_structure
        As in CarbonGenerator.
    atoms_potential : HarmonicPotential or TersoffPotential, optional
        Atom-atom potential, already parameterized. Default a HarmonicPotential with default
        parameters.
    pores_potential : LJPoresPotential, optional
        Atom-pore potential, already parameterized. Default an LJPoresPotential with default
        parameters.
    bond_dist : float, optional
        C-C bond length of the ring templates, Angstrom. Default 1.42.
    merge_coeff : float, optional
        A new atom closer than bond_dist * merge_coeff to an existing one is merged into it:
        this is how fronts stitch. Default 0.3.
    reject_coeff : float, optional
        A ring with an unmerged atom closer than bond_dist * reject_coeff to the structure is
        rejected. Default 0.8.
    E_max : float, optional
        Largest energy accepted for a placed ring, in the atoms potential's unit. Default 1000.
    LJ_exclusion_bondnum : int, optional
        Atoms within this many bonds of either anchor are excluded: from the 12-6 LJ under
        harmonic, from the whole energy under tersoff. Default 5.
    atoms_per_ring : list of int, optional
        Ring sizes that can be drawn. Default [5, 6, 7, 8].
    probabilities : list of float, optional
        Probability of each size, summing to 1. Default [0.15, 0.7, 0.10, 0.05].
    force_defect_pairs : bool, optional
        True: a child of a 5- or 7-ring is a 6-ring or the parent's 12-complement (7 or 5) with
        equal probability, at most one complement per parent, so defects come in 5-7 pairs.
        False: 5/7 rings grow like any other. Default True.

    Raises
    ------
    ValueError
        On a non-positive bond length, or probabilities that do not sum to 1 or do not match
        atoms_per_ring.
    """
    def __init__(self,
                 Box: list[float] = [0., 0., 0.], # invalid on purpose: pass Box or pores_structure
                 max_rho: float = 0,
                 periodic: list[int] = [1, 1, 1],
                 N_box: int | None = None,
                 pores_structure = None,
                 atoms_potential: TersoffPotential | HarmonicPotential | None = None,
                 pores_potential: LJPoresPotential | None = None,
                 bond_dist: float = 1.42,
                 merge_coeff: float = 0.3, 
                 reject_coeff: float = 0.8,
                 E_max:float = 1e3, 
                 LJ_exclusion_bondnum: int = 5,
                 atoms_per_ring: list[int] = [5, 6, 7, 8],
                 probabilities: list[float] = [0.15, 0.7, 0.10, 0.05],
                 force_defect_pairs: bool = True) -> None:

        super().__init__(Box, max_rho, periodic, N_box, pores_structure)
        if not (bond_dist > 0):
            raise ValueError("bond lenght must be positive")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Ring generation probabilities must sum to 1")
        if not (len(probabilities) == len(atoms_per_ring)):
            raise ValueError("Admissible ring sizes and their probabilities must match length")

        self.d = bond_dist
        self._merge = bond_dist * merge_coeff
        self._reject = bond_dist * reject_coeff
        self.E_max = E_max
        self.LJ_exclusion_bondnum = LJ_exclusion_bondnum

        self._atoms_per_ring = atoms_per_ring
        self._probabilities = probabilities
        self._force_defect_pairs = force_defect_pairs

        #Initialize parameters and variables
        self.set_optimization_parameters()
        self.set_atoms_potential(potential_class= atoms_potential)
        self.set_pores_potential(potential_class= pores_potential)

        self._resetVariables()

    def set_atoms_potential(self, 
                            pot_type: str = "harmonic", 
                            potential_class: HarmonicPotential | TersoffPotential | None = None) -> None:
        """
        Set the atom-atom potential; clears the structure built so far.

        Parameters
        ----------
        pot_type : {"harmonic", "tersoff"}, optional
            Potential built with default parameters when potential_class is None. Default
            "harmonic".
        potential_class : HarmonicPotential or TersoffPotential, optional
            An already-parameterized instance, used as is.

        Raises
        ------
        NotImplementedError
            On an unknown pot_type.
        """
        if potential_class is None:
            if pot_type == "tersoff":
                self._atoms_potential_class = TersoffPotential(Box = list(self.Box), periodic = list(self.periodic))
            elif pot_type == "harmonic":
                self._atoms_potential_class = HarmonicPotential(Box = list(self.Box), periodic = list(self.periodic))
            else:
                raise NotImplementedError("Potential class non implemented")
        else:
            self._atoms_potential_class = potential_class
        self._resetVariables()

    def set_pores_potential(self,
                            pot_type: str = "lj",
                            potential_class: LJPoresPotential | None = None) -> None:
        """
        Set the atom-pore potential and hand it the pores; clears the structure built so far.

        Parameters
        ----------
        pot_type : str, optional
            "lj" for the Lennard-Jones wall (12-6 by default), the only one implemented. Its
            repulsion is singular, so carbon never rests on a pore surface, and LAMMPS' lj126
            wall/region is the same potential (see io_functions.print_LAMMPS_pores). To add
            another, subclass PoresPotential and give it a branch here.
        potential_class : PoresPotential subclass, optional
            An already-parameterized instance, used as is instead of building one from pot_type.

        Raises
        ------
        NotImplementedError
            On an unknown pot_type.
        """
        if potential_class is None:
            if pot_type == "lj":
                self._pores_potential_class = LJPoresPotential(Box = list(self.Box), periodic = list(self.periodic))
            else:
                raise NotImplementedError("Potential class non implemented")
        else:
            self._pores_potential_class = potential_class

        self._pores_potential_class.set_coords(pores_coord = self.coord_pores, pores_radius = self.R_pores)
        self._resetVariables()

    def set_optimization_parameters(self, 
                                    max_iter: int = 100, 
                                    grad_step: float = 2.5/360*2*np.pi) -> None:
        """
        Set the orientation line search.

        Parameters
        ----------
        max_iter : int, optional
            Maximum rotation steps per ring. Default 100.
        grad_step : float, optional
            First rotation step, radians; it is halved and reversed after each uphill step.
            Default 2.5 degrees.
        """
        self.max_iter = max_iter
        self.grad_step = grad_step

    def _resetVariables(self) -> None:
        """Clear the structure: atoms, neighbor cells, bond graph and ring registry."""
        self._coord = np.empty((0,3), dtype = float)
        self._neigh_boxes = NeighborhoodManager(Box = list(self.Box), periodic = list(self.periodic))
        self._neigh_boxes.set_M_matrix(self.N_box) 

        self._adjacency = {}

        # rings
        self._rings = {}
        self._atom_to_ring = {}
        self._ring_counter = 0

        #initialize bonds
        self._bonds = np.empty((0, 2), dtype=int)

    @property
    def bonds(self):
        """Bonds, (M, 2) int ndarray, in the order they were made (the growth graph)."""
        return self._bonds

    @property
    def adjacency(self):
        """Bond graph {atom: set of bonded atoms}."""
        return self._adjacency

    #ADJACENCY-RELATED METHODS
    def _update_adjacency(self, vector: NDArray[np.int64]) -> None:
        """
        Add bonds to the bond graph.

        Parameters
        ----------
        vector : (M, 2) array of int
            New bonds, as atom-index pairs.
        """
        for a, b in vector:
            if a not in self._adjacency:
                self._adjacency[a] = set()
            if b not in self._adjacency: 
                self._adjacency[b] = set()
            self._adjacency[a].add(int(b))
            self._adjacency[b].add(int(a))

    def _count_bonds(self, atom: int) -> int:
        """
        Number of bonds of an atom.

        Parameters
        ----------
        atom : int
            Atom index.

        Returns
        -------
        int
        """
        return len(self.adjacency.get(atom, ()))


    #RINGS-RELATED METHODS    
    def _decide_new_ring_size(self, 
                              anchor_atom: int) -> tuple[int, int, bool]:
        """
        Size of the next ring grown from an atom, with the 5-7 defect-pair rule.

        Sizes are drawn from atoms_per_ring and probabilities. With force_defect_pairs, a child of a
        5- or 7-ring is a 6-ring or the parent's 12-complement with equal probability, until the
        parent has one such defect child; after that, a 6-ring.

        Parameters
        ----------
        anchor_atom : int
            Atom the ring grows from (-1 for the seed ring).

        Returns
        -------
        rand_size : int
            Size of the new ring.
        ring_id : int
            Ring the anchor belongs to (-1 if none).
        defect_pair : bool
            True if the new ring is the parent's defect child.
        """
        ring_id = self._atom_to_ring.get(anchor_atom, None)
        rand_size = np.random.choice(self._atoms_per_ring, p = self._probabilities)

        if ring_id is None:
            return rand_size, -1, False  # free growth

        ring = self._rings[ring_id]

        if ring['n_atoms'] in (5, 7) and self._force_defect_pairs:
            if not ring['defect_child']:
                rand_size = np.random.choice([6, 12 - ring['n_atoms']], p = [0.5, 0.5])
                if rand_size == 12 - ring['n_atoms']:
                    return rand_size, ring_id, True
                else:
                    return rand_size, ring_id, False
            else:
                return 6, ring_id, False

        return rand_size, ring_id, False

    def _register_ring(self, 
                       atom_indices: list[int], 
                       n_atoms: int, 
                       parent: int | None = None, 
                       defect_pair: bool = False) -> int:
        """
        Register a newly placed ring in the internal topology structure.

        This method:
        - assigns a unique ring identifier,
        - stores ring metadata (size, atoms, adjacency),
        - updates atom-to-ring mapping,
        - updates ring-ring neighbor relationships,
        - handles defect parent/child bookkeeping.

        Parameters
        ----------
        atom_indices : list[int]
            Global atom indices belonging to the ring. The list may include indices of pre-existing atoms.
        n_atoms : int
            Ring size (number of atoms in the cycle). This is the intended ring size sampled during growth.
        parent : int or None, default=None
            Identifier of the parent ring from which this ring was generated.
            If provided, mutual adjacency between parent and child is recorded.
        defect_pair : bool, default=False
            If True, marks the parent ring as having generated a defect
            (e.g., non-hexagonal 5-7 pair logic).

        Returns
        -------
        ring_id : int
            Unique identifier assigned to the registered ring.

        Notes
        -----
        - The internal structure '_rings' stores:
            * 'n_atoms'      : ring size
            * 'atoms'        : set of atom indices
            * 'neighbors'    : adjacent ring IDs
            * 'defect_child' : flag used in defect bookkeeping
            * 'parent'       : parent ring ID (if any)
        - Atom-to-ring mapping is updated for all atoms in 'atom_indices'.
        - Adjacency is defined at the ring level via shared growth ancestry, not by geometric intersection.
        """
        ring_id = self._ring_counter
        self._ring_counter += 1

        self._rings[ring_id] = {
                'n_atoms': n_atoms,
                'atoms': set(atom_indices),
                'neighbors': set(),     # all adjacent rings
                'defect_child': False,  # only for 5–7 logic
                'parent': parent
        }

        for a in atom_indices:
            self._atom_to_ring[a] = ring_id 

        if parent is not None:
            self._rings[parent]['neighbors'].add(ring_id)
            self._rings[ring_id]['neighbors'].add(parent)

            if defect_pair:
                self._rings[parent]['defect_child'] = True

        return ring_id

    def _build_candidate_ring(self, 
                              idx : int
                              ) -> tuple[NDArray | None, int, tuple[int, int, bool]]:
        """
        Build a candidate ring on a bond of atom idx.

        The second anchor is a random unsaturated neighbor of idx. The ring is laid in the plane of
        idx and its first two neighbors, then placed on the bond idx - second anchor, pointing away
        from the second anchor's other neighbor (geometry_functions.place_ring).

        Parameters
        ----------
        idx : int
            Anchor atom.

        Returns
        -------
        new_ring : (n, 3) ndarray or None
            Ring coordinates, rows 0 and 1 on the two anchors; None if idx has three bonds, no
            unsaturated neighbor, or a second anchor with no other neighbor.
        second_anchor : int
            The second anchor (-1 on failure).
        ring_data : tuple of (int, int, bool)
            Ring size, parent ring id and defect-pair flag (see _decide_new_ring_size).
        """
        indeces = np.array(sorted(self._adjacency.get(idx, set())), dtype=int)
        bond_numbers = np.array([self._count_bonds(atom) for atom in indeces])

        if np.all(bond_numbers>=3) or indeces.size == 3: # no growth from a saturated atom
            return None, -1, (-1, -1, False)

        second_anchor = np.random.choice(indeces[bond_numbers<3]) # any unsaturated neighbor of idx

        second_anchor_bonded = [n for n in self._adjacency.get(second_anchor, set())
                if n != idx and n != second_anchor]

        if len(second_anchor_bonded) == 0: # nothing to orient the ring against
            return None, -1, (-1, -1, False)

        # Create and rotate/align new ring
        n_ring, ring_id, defect_pair = self._decide_new_ring_size(idx)

        new_ring = gf.carbon_rings_template(n_ring, self.d)
        # fold the neighbors onto the anchor: one across a periodic face would otherwise give a
        # vector spanning the box, and a wrong plane
        neighbours = gf.fold(self._coord[idx,:], self._coord[indeces[:2],:], self.Box, self.periodic)
        target_normal = gf.plane_normal(self._coord[idx,:], neighbours[0], neighbours[1])
        new_ring = gf.align_ring_to_normal(new_ring, target_normal)

        new_ring = gf.place_ring(new_ring, 
                                 self._coord[idx], 
                                 self._coord[second_anchor],
                                 self._coord[second_anchor_bonded],
                                 self.Box, 
                                 self.periodic, 
                                 bond_dist = self.d, 
                                 anchor_idx = [0, 1])

        return new_ring, second_anchor, (int(n_ring), ring_id, defect_pair)

    def _build_neigh_lists(self, 
                           idx: int, 
                           second_anchor: int) -> tuple[list[int], list[int], list[int]]:
        """
        Neighbor lists for the energy of a ring grown on the bond idx - second_anchor.

        Parameters
        ----------
        idx, second_anchor : int
            The two anchors.

        Returns
        -------
        listNeigh : list of int
            Atoms in the half-shell cells of idx (complete to about 5 A from it), idx included:
            every atom the energy can see.
        LJ_exclusion : list of int
            The anchors and the atoms within LJ_exclusion_bondnum bonds of either; excluded from the
            12-6 LJ under harmonic, from the whole energy under tersoff.
        attraction_list : list of int
            Atoms of listNeigh with fewer than 3 bonds: 9-3 attraction instead of LJ (harmonic,
            during generation).
        """
        listNeigh = set(self._neigh_boxes.get_local_neighbors(idx)) | {idx}

        # atoms close in the bond graph to either anchor
        LJ_exclusion = set(gf.get_n_level_neighbors(self._adjacency, idx, n_level = self.LJ_exclusion_bondnum)) | {idx}
        LJ_exclusion |= set(gf.get_n_level_neighbors(self._adjacency, second_anchor, n_level = self.LJ_exclusion_bondnum)) | {second_anchor}

        attraction_list = [atom for atom in listNeigh if self._count_bonds(atom) < 3]

        listNeigh = sorted(listNeigh)
        LJ_exclusion = sorted(LJ_exclusion) # Subset of nearby atoms to be excluded from computation of non-bonded interactions
        attraction_list = sorted(attraction_list) # Subset of non-saturated atoms nearby

        return listNeigh, LJ_exclusion, attraction_list

    def _evaluate_generation_energy(self, 
                                    new_idxes: list[int], 
                                    new_ring: NDArray, 
                                    listNeigh: list[int], 
                                    LJ_exclusion: list[int], 
                                    attraction_list: list[int]) -> float:
            """
            Energy of a candidate ring: atoms potential (generation=True) plus pore wall.

            Parameters
            ----------
            new_idxes : list of int
                Indices the new atoms (rows 2 onward of new_ring) will take.
            new_ring : (n, 3) ndarray
                Candidate ring; rows 0 and 1 are the anchors, already in the structure.
            listNeigh, LJ_exclusion, attraction_list : list of int
                Neighbor lists from _build_neigh_lists.

            Returns
            -------
            float
                Energy in the atoms potential's unit (the pore wall is on the same scale).
            """
            prov_coord = np.vstack((self._coord, new_ring[2:])) # coordinates with the new ring
            E = self._atoms_potential_class.compute_energy(
                new_idxes = new_idxes, 
                coord = prov_coord,  
                neighList = listNeigh, 
                neigh_exclusion = LJ_exclusion, 
                attraction_list = attraction_list, 
                generation = True) # rows 0 and 1 of new_ring are the anchors, already in the structure

            if self.coord_pores is not None:
                E_pores, _ = self._pores_potential_class.compute_energy(new_ring[2:])
                E += E_pores

            return E

    def _optimize_ring_orientation(self,
                                   idx: int,
                                   second_anchor: int,
                                   new_ring: NDArray[np.float64],
                                   listNeigh: list[int],
                                   LJ_exclusion: list[int],
                                   attraction_list: list[int]) -> NDArray[np.float64] | None:
        """
        Orient a candidate ring by a line search on its energy, then gate it on E_max.

        The ring rotates about the bond idx - second_anchor: a step that lowers the energy is kept,
        an uphill step halves the step and reverses it. The search stops when the energy changes by
        less than 1 % (or 1e-4), or after max_iter steps (set_optimization_parameters).

        Parameters
        ----------
        idx, second_anchor : int
            The two anchors; the rotation axis is their bond.
        new_ring : (n, 3) ndarray
            Candidate ring from _build_candidate_ring.
        listNeigh, LJ_exclusion, attraction_list : list of int
            Neighbor lists from _build_neigh_lists.

        Returns
        -------
        (n, 3) ndarray or None
            The oriented ring; None if its final energy exceeds E_max (the gate tests the ring that
            would be placed, not the intermediate candidates), or on a NaN or infinite energy.
        """
        new_idxes = np.arange(self.atoms_number, self.atoms_number+new_ring[2:].shape[0]) # indices the new atoms will take

        E = self._evaluate_generation_energy(new_idxes, new_ring, listNeigh, LJ_exclusion, attraction_list) # starting energy
        E = np.sum(E) # some potential classes provide an energy for each atom

        rotation_axis = self._coord[idx,:]-self._coord[second_anchor,:] # this axis does not change during optimization
        rotation_axis /= np.linalg.norm(rotation_axis)

        anchor = 0 # this anchor also is fixed
        d_angle = self.grad_step

        for _ in range(self.max_iter):
            # orientation line-search
            moved_ring = gf.axis_rotation(new_ring, rotation_axis, d_angle, anchor) # rotate the ring by d_angle about the bond

            E_new = self._evaluate_generation_energy(new_idxes, moved_ring, listNeigh, LJ_exclusion, attraction_list) 
            E_new = np.sum(E_new) # in case a potential returns per-atom energies

            if E_new > E:
                d_angle = -d_angle/2.0
                continue

            if np.isnan(E_new) or np.isinf(E_new):
                return None

            if np.allclose(E_new, E, rtol=1e-2, atol=1e-4):
                break

            new_ring = moved_ring.copy()
            E = E_new

        if E > self.E_max:
            return None
        return new_ring

    def _start_carbon(self) -> tuple[NDArray, int]: 
        """
        Seed ring: a ring of sampled size at a random position and orientation.

        A candidate is retried until no atom is inside a pore, every atom is farther than two bond
        lengths from the existing atoms, and its pore energy is within E_max (the seed has no
        parent, so this is its only energy gate). Works on a box that already holds atoms.

        Returns
        -------
        coord : (n, 3) ndarray
            Seed ring coordinates, Angstrom.
        n_ring : int
            Ring size.

        Raises
        ------
        RuntimeError
            After 10 000 failed tries, with the count of each failed check.
        """
        n_ring, _ , _  = self._decide_new_ring_size(-1) 
        tries_cap = 10_000

        tries: dict[str, int] = {'holes': 0, 'coords': 0, 'energy': 0}

        while True:

            if sum(tries.values()) > tries_cap: 
                raise RuntimeError(f"_start_carbon failed seeding after {tries_cap} attempts "
                                   f"(box={self.Box}, n_pores={0 if self.coord_pores is None else len(self.coord_pores)}); "
                                   f"holes/coords/energy tries: {tries.get('holes', 0)}, {tries.get('coords', 0)}, {tries.get('energy', 0)}")

            new_point = [
                np.random.random() * self.Box[0],
                np.random.random() * self.Box[1],
                np.random.random() * self.Box[2]
            ]

            new_ring = gf.carbon_rings_template(n_ring, self.d)
            random_orientation = gf.random_unit_vector()
            new_ring = gf.align_ring_to_normal(new_ring, random_orientation)

            coord = new_ring + new_point #translation

            # Apply PBC or box limits 
            coord = gf.apply_box_constraints(coord, self.Box, self.periodic)

            check_holes = self._check_pore_constraint(coord)

            if not check_holes:
                tries['holes'] += 1
                continue

            check_coords = True
            if np.size(self._coord) != 0:
                check_coords = all(
                    np.all(gf.distance(atom, self._coord, self.Box, self.periodic)>2*self.d)
                    for atom in coord
                )

            if not check_coords:
                tries['coords'] += 1
                continue

            check_energy = True
            if self.coord_pores is not None:
                E_pores, _ = self._pores_potential_class.compute_energy(coord)
                check_energy = E_pores <= self.E_max

            if not check_energy:
                tries['energy'] += 1
                continue

            if check_holes and check_coords and check_energy:
                break

        return coord, n_ring

    def _detect_overlaps(self, 
                         ring: NDArray, 
                         listNeigh: list[int]) -> list[list[int]]:
        """
        New ring atoms to merge into existing atoms.

        Parameters
        ----------
        ring : (n, 3) ndarray
            Candidate ring; rows 0 and 1 (the anchors) are skipped.
        listNeigh : list of int
            Existing atoms to test.

        Returns
        -------
        list of (int, int)
            (ring row, existing atom) for each ring atom closer than bond_dist * merge_coeff to an
            atom of listNeigh; the first such atom in listNeigh order is taken, whatever its degree.
        """
        overlap_pairs = []

        for j, ring_atom in enumerate(ring[2:], start=2):

            distances = gf.distance(
                ring_atom,
                self._coord[listNeigh],
                self.Box,
                self.periodic
            )

            close_mask = np.where(distances < self._merge)[0]

            if close_mask.size > 0:
                existing_atom = listNeigh[close_mask[0]]
                overlap_pairs.append((j, existing_atom))

        return overlap_pairs

    def _add_atoms(self, 
                   idx: int
                   ) -> tuple[NDArray | None, int, list[list[int]], tuple]:
        """
        Try to grow one ring from atom idx: build, orient, check the pores, find the merges.

        Parameters
        ----------
        idx : int
            Anchor atom.

        Returns
        -------
        optimized_ring : (n, 3) ndarray or None
            The ring, wrapped into the box; None if a step failed.
        second_anchor : int
            The second anchor (-1 on failure).
        overlap_pairs : list of (int, int)
            Merges from _detect_overlaps.
        ring_data : tuple
            (size, parent ring id, defect-pair flag); empty on failure.
        """

        # step 1: build and place a candidate ring
        new_ring, second_anchor, ring_data = self._build_candidate_ring(idx)
        if new_ring is None: # saturated anchor
            return None, -1, [], ()

        # step 2: neighbor lists
        listNeigh, LJ_exclusion, attraction_list = self._build_neigh_lists(idx, second_anchor)

        # step 3: orient the ring (one degree of freedom) and gate it on E_max

        optimized_ring = self._optimize_ring_orientation(idx,
                                                         second_anchor,
                                                         new_ring,
                                                         listNeigh,
                                                         LJ_exclusion,
                                                         attraction_list)
        if optimized_ring is None: 
            return None, -1, [], ()

        # no atom inside a pore
        if self._check_pore_constraint(optimized_ring) is False:
            return None, -1, [], ()

        # Apply PBC or box limits after optimization
        optimized_ring = gf.apply_box_constraints(optimized_ring, self.Box, self.periodic)

        # atoms landing on existing ones are merged, not rejected
        overlap_pairs = self._detect_overlaps(optimized_ring, listNeigh)

        return optimized_ring, second_anchor, overlap_pairs, ring_data

    def _generate(self) -> None:
        """
        Grow the structure until the target density, or until every atom has failed as an anchor
        (front death).

        After the seed ring, each step picks an anchor (preferring the live atoms of the active
        ~10 A cell, so one cell fills before growth moves on), grows a ring on it and applies, in
        order: the merges, the reject distance, the over-long-bond guard (1.5 bond_dist) and the
        coordination cap (at most 4 bonds). An anchor that fails is not tried again. Duplicate bonds
        from merges are removed at the end.
        """

        self._resetVariables() 
        FAILED = set()

        # PLACE FIRST RING
        self._initialize_structure()

        # START OF GENERATION LOOP
        prefe = {}
        active_box = -1  # set on the first iteration, which always reseeds
        while (self.atoms_number > len(FAILED)) and (self.density < self.max_rho):

            # BOX-SATURATION GROWTH: fill one ~10 A box before moving on. When the
            # active box has no live anchors left (first iteration, or box fully
            # saturated), reseed with a random anchor and switch to its box.
            reseed = len(set(prefe) - FAILED) == 0
            idx = self._select_growth_index(prefe, FAILED)
            if reseed:
                _, active_box = self._neigh_boxes.box_identify(self._coord[idx])

            # ADD NEW RING
            new_coord, second_anchor, overlap_pairs, ring_data = self._add_atoms(idx)
            if new_coord is None:
                FAILED.add(idx)
                prefe.pop(idx, None)
                continue
            new_coord = new_coord[2:]

            new_coord, new_bonds, new_idxes = self._resolve_overlap(new_coord,
                                                         overlap_pairs,
                                                         idx,
                                                         second_anchor,
                                                         )
            if len(new_idxes) == 0:
                FAILED.add(idx)
                prefe.pop(idx,None)
                continue

            #SAFETY NET ON OVERLAPPING WITH EXISTING STRUCTURE:
            overlaps = self._reject_overlapping_ring(idx, new_coord)
            if overlaps:
                FAILED.add(idx)
                prefe.pop(idx,None)
                continue

            #PHANTOM-BOND GUARD: reject a ring whose merge would store an over-long
            # cross-front bond (> 1.5*bond_dist, the physical C-C ceiling).
            if self._has_overlong_bond(new_bonds, new_coord):
                FAILED.add(idx)
                prefe.pop(idx, None)
                continue

            #OVER-COORDINATION GUARD: a merge lets an existing atom absorb a new ring
            # vertex and inherit its chain bonds (_resolve_overlap); both anchors are guarded 
            # (<3) but the merge target is not. 4 is allowed (tersoff makes sp3); 5 is not.
            gained = {}
            for a, b in new_bonds:
                gained.setdefault(int(a), set()).add(int(b))
                gained.setdefault(int(b), set()).add(int(a))
            if any(len(self._adjacency.get(atom, set()) | nb) > 4
                   for atom, nb in gained.items() if atom < self.atoms_number):
                FAILED.add(idx)
                prefe.pop(idx, None)
                continue
            self._bonds = np.vstack((self._bonds, np.array(new_bonds, dtype=int)))
            self._update_adjacency(np.array(new_bonds, dtype=int))
            self._coord = np.vstack((self._coord, new_coord)) #add coordinates

            #UPDATE RINGS
            #taking the indeces of the ring from new_bonds, which contains all the indexes involved in the new placement, included already-existing atoms.
            self._register_ring([int(new_idx) for new_idx in np.unique(new_bonds)], ring_data[0], parent = ring_data[1], defect_pair = ring_data[2]) 
            #UPDATED BOXING
            self._neigh_boxes.assign_atoms_to_box(new_idxes, self._coord)

            #UPDATE PREFERRED GROWTH IDXES -- box-saturation: prefer every
            # not-yet-failed atom currently in the active box; refresh each
            # iteration so atoms the ring just added to the box join the front.
            box_atoms = np.ravel(self._neigh_boxes.get_atoms_in_MyBoxes(active_box))
            prefe = {int(a): 1 for a in box_atoms if int(a) not in FAILED}

        # dedupe the stored bonds: a merge can emit an edge already in the graph.
        # Adjacency is a set so it is unaffected, but the stored bond array must be
        # duplicate-free. Canonicalize (min,max) and keep the first of each unique edge.
        if len(self._bonds):
            canon = np.sort(self._bonds, axis=1)
            _, keep = np.unique(canon, axis=0, return_index=True)
            self._bonds = self._bonds[np.sort(keep)]

    def _initialize_structure(self) -> None:
        """Place the seed ring, with its bonds, ring record and neighbor cells."""
        # place first ring
        coord, n_ring = self._start_carbon()
        atom_indices = [*range(len(self._coord), len(self._coord)+len(coord))]
        self._coord = np.vstack((self._coord, coord))
        self._register_ring(atom_indices, n_ring, parent = None, defect_pair = False)
        # assign early particles to Boxes
        self._neigh_boxes.assign_atoms_to_box(atom_indices, self._coord)

        # update bonds and adjacency
        new_bonds = [[atom_indices[k], atom_indices[(k+1) % len(atom_indices)]] for k in range(len(atom_indices))]
        self._bonds = np.vstack((self._bonds, np.array(new_bonds)))
        self._update_adjacency(self._bonds)

    def _resolve_overlap(self, 
                         new_coord: NDArray[np.float64], 
                         overlap_pairs: list[list[int]], 
                         idx: int, 
                         second_anchor: int
                         ) -> tuple[NDArray, list[list[int]], list[int]]:
        """
        Number the new atoms, apply the merges and list the bonds of the new ring.

        A merged ring atom is replaced by the existing atom it landed on, which inherits its chain
        bonds; a merge onto an anchor adds no bond from the anchor to itself.

        Parameters
        ----------
        new_coord : (m, 3) ndarray
            New ring atoms (anchors excluded), Angstrom.
        overlap_pairs : list of (int, int)
            (ring row, existing atom) from _detect_overlaps; ring rows count the anchors, so row r
            is new_coord[r - 2].
        idx, second_anchor : int
            The two anchors.

        Returns
        -------
        new_coord : (k, 3) ndarray
            The atoms to append (merged ones removed).
        new_bonds : list of [int, int]
            Bonds of the new ring, chain plus the two anchor bonds, in final indices.
        new_idxes : list of int
            Final indices of the appended atoms; empty if every atom merged.

        Raises
        ------
        RuntimeError
            If a bond points past the atoms that will exist.
        """
        new_idxes = np.arange(self.atoms_number, self.atoms_number+len(new_coord)) # indices of the new atoms before any merge

        # no merge
        if len(overlap_pairs)==0: 
            new_bonds = [[i+self.atoms_number, i+1+self.atoms_number] for i in range(len(new_coord)-1)]
            new_bonds.append([second_anchor, self.atoms_number])
            new_bonds.append([idx, self.atoms_number+len(new_coord)-1])
            return new_coord.copy(), new_bonds, [int(index) for index in new_idxes]

        # merges: each merged ring atom becomes the existing atom it landed on
        # Map local overlapping atoms to existing atoms
        index_map = {}
        for local_idx, existing_atom in overlap_pairs:
            global_new_idx = self.atoms_number + (local_idx - 2)
            index_map[global_new_idx] = existing_atom
        # Remove overlapping coordinates from the new list
        mask_keep = np.ones(len(new_coord), dtype=bool)
        for local_idx, _ in overlap_pairs:
            mask_keep[local_idx - 2] = False
        new_coord = new_coord[mask_keep, :].copy()

        removed = new_idxes[np.logical_not(mask_keep)]

        # chain bonds, renumbered past the removed atoms
        new_bonds = []
        for i in range(len(new_idxes) - 1):
            a = new_idxes[i]
            b = new_idxes[i + 1]
            a = index_map.get(a, a)
            b = index_map.get(b, b)
            # consider removed indeces
            a -= np.sum(a > removed)
            b -= np.sum(b > removed)
            if a != b:
                new_bonds.append([a, b])
        # Bonds connecting to anchors
        new_idxes_updated = new_idxes[mask_keep]
        new_idxes_updated = sorted([i - np.sum(i > removed) for i in new_idxes_updated])

        if len(new_idxes_updated)>0:

            first_atom = new_idxes_updated[0]
            if np.logical_not(mask_keep[0]): # the first ring atom was merged
                first_atom = index_map.get(new_idxes[0], first_atom)
            last_atom = new_idxes_updated[-1]
            if np.logical_not(mask_keep[-1]):
                last_atom = index_map.get(new_idxes[-1], last_atom)
            # a ring atom merged onto an anchor IS that anchor: bonding it to itself would count
            # a phantom third bond and give it a zero-length plane normal when it next anchors
            if last_atom != idx:
                new_bonds.append([idx, last_atom])
            if first_atom != second_anchor:
                new_bonds.append([second_anchor, first_atom])

            if not (np.all( np.unique(new_bonds).ravel() < self.atoms_number + len(new_coord) )):
                raise RuntimeError("indeces not-consistent with atoms placed")

            return new_coord, new_bonds, new_idxes_updated
        else:
            return np.empty((0,3), dtype=float), [], []

    def _select_growth_index(self, 
                             prefe: dict, 
                             FAILED: set) -> int:
        """
        Pick the next anchor: among the preferred atoms (weighted by their values in prefe), or
        uniformly among all live atoms when no preferred one is left.

        Parameters
        ----------
        prefe : dict
            Preferred atoms {atom: weight}.
        FAILED : set of int
            Atoms that have failed as anchors.

        Returns
        -------
        int
            Atom index.
        """
        valid_prefe = set(prefe.keys()) - FAILED

        if len(valid_prefe) == 0:
            opz = list(set(range(self.atoms_number)) - FAILED)
            return np.random.choice(opz)

        prefe_list = list(valid_prefe)
        counts = np.array([prefe.get(i, 1) for i in prefe_list], dtype=float)
        probs = counts / counts.sum()

        return int(np.random.choice(prefe_list, p=probs))

    def _reject_overlapping_ring(self, 
                                 idx: int, 
                                 ring: NDArray) -> bool:
        """
        True if a new ring atom is closer than bond_dist * reject_coeff to an existing atom.

        Parameters
        ----------
        idx : int
            Anchor; the atoms of its half-shell cells, minus its bonded neighbors, are tested.
        ring : (m, 3) ndarray
            New ring atoms (after the merges), Angstrom.

        Returns
        -------
        bool
        """
        listNeigh = list(set(self._neigh_boxes.get_local_neighbors(idx)) - set(gf.get_n_level_neighbors(self._adjacency, idx, n_level=1)))
        distances = np.array([gf.distance(ring_atom, self._coord[listNeigh, :], self.Box, self.periodic) for ring_atom in ring]).flatten()
        if np.any(distances < self._reject):
            return True
        return False

    def _has_overlong_bond(self,
                           new_bonds: list,
                           new_coord: NDArray) -> bool:
        """
        True if a bond the placement would store is longer than 1.5 bond_dist: an over-long weld
        between fronts (a phantom bond).

        Parameters
        ----------
        new_bonds : list of [int, int]
            Bonds from _resolve_overlap; indices from atoms_number on point into new_coord.
        new_coord : (k, 3) ndarray
            The atoms to append, Angstrom.

        Returns
        -------
        bool
        """
        cap = 1.5 * self.d
        n0 = self.atoms_number
        for a, b in new_bonds:
            a, b = int(a), int(b)
            pa = self._coord[a] if a < n0 else new_coord[a - n0]
            pb = self._coord[b] if b < n0 else new_coord[b - n0]
            if gf.distance(pa, pb[None, :], self.Box, self.periodic)[0] > cap:
                return True
        return False


class RandomGenerator(CarbonGenerator):
    """
    Disordered carbon by random sequential addition of single atoms.

    Each atom must keep min_dist from the others and pore_standoff from the pore surfaces. When
    max_retries draws fail in a row, the atom closest to the last failed draw is removed
    (rollback) and placement resumes, until the target density or max_rollbacks is reached.

    Parameters
    ----------
    Box, max_rho, periodic, N_box, pores_structure
        As in CarbonGenerator.
    min_dist : float, optional
        Minimum interatomic distance, Angstrom. Default 1.5.
    pore_standoff : float, optional
        Clearance beyond each pore radius, Angstrom: an atom at r is forbidden if
        dist(r, pore center) < R_pore + pore_standoff. Default 0.75.
    max_retries : int, optional
        Random draws before a rollback. Default 50.
    max_rollbacks : int, optional
        Rollbacks before generation stops, possibly below max_rho. Default 200.

    Raises
    ------
    ValueError
        On a non-positive min_dist or max_retries, or a negative pore_standoff or
        max_rollbacks.
    """
    def __init__(self,
                 Box: list[float] = [0., 0., 0.],   # invalid on purpose: pass Box or pores_structure
                 max_rho: float = 0,
                 periodic: list[int] = [1, 1, 1],
                 N_box: int | None = None,
                 pores_structure = None,
                 min_dist: float = 1.5,
                 pore_standoff: float = 0.75,
                 max_retries: int = 50,
                 max_rollbacks: int = 200,
                 ) -> None:
        super().__init__(Box, max_rho, periodic, N_box, pores_structure)

        if not (min_dist > 0):
            raise ValueError("min_dist must be positive")
        if not (pore_standoff >= 0):
            raise ValueError("pore_standoff must be non-negative")
        if not (max_retries > 0):
            raise ValueError("max_retries must be positive")
        if not (max_rollbacks >= 0):
            raise ValueError("max_rollbacks must be non-negative")

        self.min_dist = min_dist
        self.pore_standoff = pore_standoff
        self.max_retries = max_retries
        self.max_rollbacks = max_rollbacks

        self._resetVariables()

    def _resetVariables(self) -> None:
        """Clear the atoms and the neighbor cells."""
        self._coord = np.empty((0, 3), dtype=float)
        self._neigh_boxes = NeighborhoodManager(Box = list(self.Box), 
                                                periodic = list(self.periodic))
        self._neigh_boxes.set_M_matrix(self.N_box)

    def _check_min_dist(self, candidate: NDArray) -> bool:
        """
        True if a candidate is at least min_dist from every atom in its half-shell cells.

        Parameters
        ----------
        candidate : (3,) ndarray
            Candidate position, Angstrom.

        Returns
        -------
        bool
        """
        if self.atoms_number == 0:
            return True
        neighBox, _ = self._neigh_boxes.box_identify(candidate)
        neigh_atoms = np.unique(np.concatenate(
                    [np.ravel(self._neigh_boxes.get_atoms_in_MyBoxes(b)) for b in neighBox]
        ))
        if neigh_atoms.size == 0:
            return True
        distances = gf.distance(candidate, self._coord[neigh_atoms, :], self.Box, self.periodic)
        return bool(np.all(distances >= self.min_dist))

    def _is_valid_position(self, candidate: NDArray) -> bool:
        """
        True if a candidate satisfies both min_dist and the pore standoff.

        Parameters
        ----------
        candidate : (3,) ndarray
            Candidate position, Angstrom.

        Returns
        -------
        bool
        """
        return self._check_min_dist(candidate) and self._check_pore_constraint(candidate, standoff= self.pore_standoff)

    def _sample_candidate(self) -> NDArray:
        """
        Uniform random position in the box.

        Returns
        -------
        (3,) ndarray
            Position, Angstrom.
        """
        return np.array([np.random.random() * b for b in self.Box])

    # Rollback

    def _select_rollback_target(self, failed_candidate: NDArray) -> int:
        """
        Atom to remove when placement is jammed around a failed candidate.

        The nearest atom within 2 * min_dist of the candidate: removing the nearest blocker frees
        the most room there. With no atom in that sphere (the pores block the spot) a random atom
        is removed instead.

        Parameters
        ----------
        failed_candidate : (3,) ndarray
            Last position that could not be placed, Angstrom.

        Returns
        -------
        int
            Index of the atom to remove.

        Raises
        ------
        RuntimeError
            On an empty structure.
        """
        if self.atoms_number == 0:
            raise RuntimeError("_select_rollback_target called with empty structure")

        distances = gf.distance(failed_candidate, self._coord, self.Box, self.periodic)
        search_radius = 2.0 * self.min_dist
        nearby_mask = distances < search_radius

        if np.any(nearby_mask):
            nearby_indices = np.where(nearby_mask)[0]
            return int(nearby_indices[np.argmin(distances[nearby_mask])])
        else:
            # Remove a random atom to reshuffle
            return int(np.random.randint(0, self.atoms_number))

    def _remove_atom(self, idx: int) -> None:
        """
        Remove an atom and rebuild the neighbor cells from scratch.

        NeighborhoodManager has no removal, and the indices above idx shift; a rebuild is O(N) and
        rollbacks are rare compared with placements.

        Parameters
        ----------
        idx : int
            Atom to remove.
        """
        self._coord = np.delete(self._coord, idx, axis=0)
        # rebuild the neighbor cells after the index shift
        self._neigh_boxes = NeighborhoodManager(Box = list(self.Box), periodic = list(self.periodic))
        self._neigh_boxes.set_M_matrix(self.N_box)

        if self.atoms_number > 0:
            self._neigh_boxes.assign_atoms_to_box(
                list(range(self.atoms_number)), self._coord
            )

    # Core generation

    def _generate(self) -> NDArray:
        """
        Place atoms until max_atoms (the target density), max_rollbacks, or an empty box that
        still rejects every draw.

        Returns
        -------
        (N, 3) ndarray
            Atom coordinates, Angstrom.
        """
        # Always start a new structure, never accumulate
        self._resetVariables()

        rollbacks = 0
        last_failed_candidate = None

        while self.atoms_number < self.max_atoms:
            placed = False

            for _ in range(self.max_retries):
                candidate = self._sample_candidate()
                if self._is_valid_position(candidate):
                    # Accept placement
                    new_idx = self.atoms_number
                    self._coord = np.vstack((self._coord, candidate[np.newaxis, :]))
                    self._neigh_boxes.assign_atoms_to_box([new_idx], self._coord)
                    placed = True
                    last_failed_candidate = None
                    break
                else:
                    last_failed_candidate = candidate

            if placed:
                continue

            # Placement blocked: execute rollback
            if rollbacks >= self.max_rollbacks:
                break  # Give up trying to replace atoms

            if self.atoms_number == 0:
                # Completely empty box and still failing -> pore geometry is the issue; nothing can be done.
                break

            if last_failed_candidate is None:
                # Should be unreachable: the retry loop always sets 'last_failed_candidate' before 'placed' can be False.
                # Guard against future refactors.
                raise RuntimeError(
                    "Rollback triggered but last_failed_candidate is None. "
                    "This is a bug - please report it."
                )

            target_idx = self._select_rollback_target(last_failed_candidate)
            self._remove_atom(target_idx)
            rollbacks += 1

        return self._coord.copy()