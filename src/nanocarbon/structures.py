# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Structure containers: PoreStructure and CarbonStructure.

PoreStructure holds the pores (centers, radii, achieved porosity); CarbonStructure holds the
atoms and their bond graph, and is what every post-processing function takes. Both build
from a generator (generate), round-trip through to_dict / from_dict, and write LAMMPS input.
CarbonStructure.from_LAMMPS_file imports one frame of an MD trajectory. Lengths in Angstrom.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from . import geometry_functions as gf
from .io_functions import print_LAMMPS_structure, print_LAMMPS_pores
from .generators import PoresGenerator, CarbonGenerator

class PoreStructure:
    """
    Spherical pores in a simulation box.

    Pores may overlap: two intersecting spheres share the void they define, and a placement is
    rejected only when it adds too little new void (PoresGenerator.acceptance_threshold).
    Build one with PoreStructure.generate or PoreStructure.from_dict.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis. Default [1, 1, 1].

    Raises
    ------
    ValueError
        On a non-positive box or bad periodic flags.
    """
    def __init__(self, 
                 Box: list[float], 
                 periodic: list[int] = [1, 1, 1]) -> None:
        # check input validity
        if not (all(b > 0 for b in Box)):
            raise ValueError("Box dimensions must be positive.")
        if not (len(periodic) == 3 and all(p in [0, 1] for p in periodic)):
            raise ValueError("Periodic must be [0/1] x 3.")

        # results
        self._coord = np.empty((0, 3), dtype=float) 
        self._R = np.array([], dtype=float)
        self._phi_num = 0
        self._occupied = None

        # parameters
        self._Box = np.array(Box)
        self._periodic = np.array(periodic)
        self._phi = None

    @classmethod
    def generate(cls,
                 Box: list[float],
                 phi: float,
                 periodic: list[int] = [1, 1, 1],
                 dx: float | None = None,
                 psd: NDArray[np.float64] | None = None,
                 mu: float = 10,
                 sigma:float = 1,
                 psd_kind: str = "volume") -> "PoreStructure":
        """
        Pack pores with PoresGenerator up to the target porosity.

        The achieved porosity (porosity) matches phi to the resolution of the voxel grid.

        Parameters
        ----------
        Box : list of float
            Box edges, Angstrom.
        phi : float
            Target pore volume fraction, 0 < phi < 1.
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
            Convention of the tabulated PSD: "volume" (default) for a volume distribution dV/dD,
            "count" for a number distribution (see PoresGenerator).

        Returns
        -------
        PoreStructure

        Raises
        ------
        ValueError
            If phi is not in (0, 1), or on invalid generator inputs.
        """
        if not (0 < phi < 1):
            raise ValueError("Porosity must be between 0 and 1.")

        # instantiate generator
        gen = PoresGenerator(Box, phi, periodic,
                             dx = dx, psd = psd, mu = mu, sigma = sigma,
                             psd_kind = psd_kind)
        # create pores
        coord, R, phi_num = gen._generate()
        # instantiate pores class
        obj = cls(Box, periodic)
        # set class variables
        obj._coord = coord
        obj._R = R
        obj._phi_num = phi_num
        obj._phi = phi
        obj._occupied = gen.occupied

        # return class object
        return obj

    def print_LAMMPS(self,
                     LJ_vars: list = [0.1, 1.0, None],
                     LJ_type: str = '126',
                     name: str = 'pores_coord') -> None:
        """
        Write the pores as LAMMPS spherical regions with a wall/region LJ each.

        Radii are written unchanged; a pore crossing a periodic face is also written as images
        (io_functions.print_LAMMPS_pores).

        Parameters
        ----------
        LJ_vars : list, optional
            [epsilon, sigma, cutoff]. Pass the generation pore potential's A as epsilon and B as
            sigma, so the anneal wall is the generation wall. B is in Angstrom and carries over as
            is; A is in the atoms potential's energy unit and must first be converted to the deck's
            (io_functions.WALL_UNIT_TO_EV). cutoff None cuts the wall at its own minimum (WCA,
            repulsive only). Default [0.1, 1.0, None].
        LJ_type : {'126', '93'}, optional
            Wall shape. Default '126'.
        name : str, optional
            Output path without extension; '.pores_settings' is appended. Default 'pores_coord'.
        """

        print_LAMMPS_pores(self._coord, self._R, self._Box, self._periodic,
                           LJ_vars=LJ_vars, LJ_type = LJ_type, name= name)

    def to_dict(self) -> dict:
        """
        The structure as a JSON-ready dict, for from_dict.

        Returns
        -------
        dict
            Keys 'Box', 'phi', 'phi_num', 'coord', 'R', 'periodic' (lists and floats).
        """
        return {
            "Box": self._Box.tolist(),
            "phi": self._phi,
            "phi_num": self._phi_num,
            "coord": self._coord.tolist(),
            "R": self._R.tolist(),
            "periodic": self._periodic.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PoreStructure":
        """
        Rebuild a structure from the output of to_dict.

        Parameters
        ----------
        data : dict
            As written by to_dict.

        Returns
        -------
        PoreStructure

        Raises
        ------
        ValueError
            On an empty dict.
        """
        if not (data):
            raise ValueError("Empty or invalid data dict object")

        obj = cls(
            Box = data["Box"],
            periodic = data["periodic"],
            )
        obj._coord = np.array(data["coord"], dtype = float)
        obj._R = np.array(data["R"], dtype = float)
        obj._phi_num = data["phi_num"]
        obj._phi = data["phi"]
        return obj

    @property
    def coordinates(self) -> NDArray:
        """Pore centers, (P, 3) ndarray, Angstrom."""
        return self._coord

    @property
    def radii(self) -> NDArray:
        """Pore radii, (P, 1) ndarray, Angstrom."""
        return self._R  

    @property
    def porosity(self) -> float:
        """Pore volume fraction achieved on the voxel grid."""
        return self._phi_num

    @property
    def occupied(self) -> NDArray:
        """
        Voxel occupancy of the packing, (Nvox,) bool ndarray; empty unless built by generate.
        """
        if self._occupied is None:
            return np.empty(0, dtype=bool)
        return self._occupied

class CarbonStructure:
    """
    A carbon structure: atom coordinates and bond graph, the input of post-processing.

    Build one with CarbonStructure.generate (from a carbon generator), from_dict (a saved
    structure) or from_LAMMPS_file (one frame of an MD trajectory). Constructed and imported
    structures store the same graph: bonds strictly shorter than bond_dist * bond_tol
    (1.42 * 1.2 = 1.704 A by default), found by distance.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis. Default [1, 1, 1].

    Raises
    ------
    ValueError
        On a non-positive box or bad periodic flags.
    """
    def __init__(self,
                 Box: list[float],
                 periodic: list[int] = [1, 1, 1]) -> None:
        if not (all(b>0 for b in Box)):
            raise ValueError("Box lengths must be positive")
        if not (len(periodic) == 3 and all(p in [0, 1] for p in periodic)):
            raise ValueError("x, y, z periodicity must be [0, 1]")

        self.periodic = np.array(periodic)
        self.Box = np.array(Box)

        self._resetVariables()
        self._atoms_pot_type = None
        self._pores_pot_type = None
        self._atoms_potential_parameters = None
        self._pores_potential_parameters = None

    def _resetVariables(self):
        """Clear the atoms and the bond graph."""
        self._coord = np.empty((0,3), dtype = float)
        self._adjacency = {}

        #initialize bonds
        self._bonds = np.empty((0, 2), dtype=int)

    @property
    def atoms_number(self):
        """Number of atoms."""
        return self._coord.shape[0]

    @property
    def coordinates(self):
        """Atom coordinates, (N, 3) ndarray, Angstrom."""
        return self._coord

    @property
    def density(self):
        """Density, g/cm^3 (all atoms carbon, 12.0107 g/mol)."""
        return self.atoms_number*12.0107/6.02214076e23/(np.prod(self.Box)*1e-24)

    @property
    def bonds(self):
        """Bonds, (M, 2) int ndarray, each row (i, j) with i < j."""
        return self._bonds

    @property
    def adjacency(self):
        """Bond graph {atom: set of bonded atoms}."""
        return self._adjacency

    def print_LAMMPS(self, name: str ='carbon_structure', write_bonds: bool = True) -> None:
        """
        Write the structure as a LAMMPS data file.

        Parameters
        ----------
        name : str, optional
            Output path without extension; '.data' is appended. Default 'carbon_structure'.
        write_bonds : bool, optional
            True writes the 'full' style with a Bonds section; False writes the 'atomic' style
            without bonds, which a reactive pair style (EDIP, Tersoff) needs, as it builds its own
            bonds from the geometry. Default True.
        """
        print_LAMMPS_structure(
            Box=list(self.Box),
            coord=self._coord,
            bonds=self.bonds,
            write_bonds=write_bonds,
            name=name,
        )

    def to_dict(self) -> dict:
        """
        The structure as a JSON-ready dict, for from_dict.

        Returns
        -------
        dict
            Keys 'Box', 'periodic', 'coord', 'bonds', plus 'atoms_potential' and 'pores_potential'
            ({'pot_type', 'dict_params'}) when the structure was generated.
        """
        dict_out = {
            "Box": self.Box.tolist(),
            "periodic": self.periodic.tolist(),

            "coord": self._coord.tolist(),
            "bonds": self._bonds.tolist(),
            }
        if self._atoms_pot_type is not None:
            dict_out["atoms_potential"] = {"pot_type": self._atoms_pot_type, "dict_params": self._atoms_potential_parameters }
        if self._pores_pot_type is not None:
            dict_out["pores_potential"] = {"pot_type": self._pores_pot_type, "dict_params": self._pores_potential_parameters }
        return dict_out    

    @classmethod
    def from_dict(cls,
                  data: dict) -> "CarbonStructure":
        """
        Rebuild a structure from the output of to_dict, with its stored bonds (no re-bonding).

        Parameters
        ----------
        data : dict
            As written by to_dict. A trajectory dict from io_functions goes to from_LAMMPS_file.

        Returns
        -------
        CarbonStructure

        Raises
        ------
        ValueError
            On an empty dict.
        KeyError
            If 'Box', 'periodic', 'coord' or 'bonds' is missing, naming from_LAMMPS_file.
        """
        if not (data):
            raise ValueError("Empty or invalid data dict object")
        # A LAMMPS-derived dict carries no box or bond graph, so it fails here on a bare
        # KeyError that reads like a corrupt file; name the expected producer instead.
        missing = [k for k in ("Box", "periodic", "coord", "bonds") if k not in data]
        if missing:
            raise KeyError(
                f"from_dict expects a dict produced by CarbonStructure.to_dict; "
                f"missing key(s): {missing}. To load a LAMMPS trajectory dict "
                f"(read_LAMMPS_traj / read_LAMMPS_xyz), use from_LAMMPS_file instead."
            )

        obj = cls(
            Box = data["Box"],
            periodic = data["periodic"]
        )
        obj._coord = np.array(data["coord"], dtype = float)
        obj._bonds = np.array(data["bonds"],dtype= int).reshape(-1, 2)
        obj._adjacency = gf.build_adjacency(obj._bonds)

        # potential types and parameters, when stored
        try:
            obj._atoms_pot_type = data["atoms_potential"].get("pot_type")
            obj._atoms_potential_parameters = data["atoms_potential"].get("dict_params")
        except:
            obj._atoms_pot_type = None
            obj._atoms_potential_parameters = None

        try:
            obj._pores_pot_type = data["pores_potential"].get("pot_type")
            obj._pores_potential_parameters = data["pores_potential"].get("dict_params")
        except:
            obj._pores_pot_type = None
            obj._pores_potential_parameters = None

        return obj

    @classmethod
    def from_LAMMPS_file(cls,
                         dict_lammps: dict,
                         frame_index: int = -1,
                         atom_type_hash: int = 1,
                         box: list[float] | None = None,
                         bond_dist: float = 1.42,
                         bond_tol: float = 1.2) -> "CarbonStructure":
        """
        Structure from one frame of a trajectory read by io_functions, bonded by distance.

        The box comes from the frame bounds (shifted so the origin is 0); a frame without bounds,
        as in every .xyz file, needs box, and its atoms are wrapped into that box. The structure
        is periodic on all axes.

        Parameters
        ----------
        dict_lammps : dict
            Output of io_functions.read_LAMMPS_traj or read_LAMMPS_xyz, {frame: data}.
        frame_index : int, optional
            Frame to load. Default -1; an index that is not a key of dict_lammps loads the last
            frame.
        atom_type_hash : int, optional
            LAMMPS atom type to keep (carbon); other types are dropped. Default 1.
        box : list of float, optional
            Box edges [Lx, Ly, Lz], Angstrom, origin at 0, for a frame without bounds. Never
            estimated from the atom extents, which is wrong under periodic boundaries.
        bond_dist : float, optional
            Reference bond length, Angstrom. Default 1.42.
        bond_tol : float, optional
            Bonds are pairs closer than bond_dist * bond_tol. Default 1.2.

        Returns
        -------
        CarbonStructure

        Raises
        ------
        ValueError
            If the frame has no box bounds and box is None.
        """
        if frame_index not in dict_lammps.keys(): # -1 or an absent key: the last frame
            frame_index = max(list(dict_lammps.keys()))
        frame = dict_lammps[frame_index]

        coord = frame['coord'][:, 1:] # drop the atom id column
        coord = coord[coord[:, 0] == atom_type_hash , 1:] # keep one atom type, then the coordinates only

        bounds = [frame['box_x'], frame['box_y'], frame['box_z']]
        if any(None in b for b in bounds):
            # No box in the file. Guessing it from the atom extents is wrong under periodic
            # boundaries (the two extreme atoms become neighbors), so the caller must give it.
            if box is None:
                raise ValueError("The frame carries no box bounds (e.g. an .xyz file): pass box=[Lx, Ly, Lz].")
            Box = np.array(box, dtype=float)
            # MD writers leave atoms slightly outside the cell; the periodic KD-trees of
            # the post-processing only accept [0, L) (a .lammpstrj is wrapped by its reader)
            coord = np.mod(coord, Box)
        else:
            lo = np.array([b[0] for b in bounds], dtype=float)
            Box = np.array([b[1] for b in bounds], dtype=float) - lo
            coord = coord - lo # shift the origin to 0

        obj = cls(
            Box = list(Box),
            periodic = [1, 1, 1]
        )

        obj._coord = coord
        obj._bonds = obj._set_bonds(d = bond_dist, bond_tol = bond_tol)
        obj._adjacency = gf.build_adjacency(obj._bonds)

        return obj

    def _set_bonds(self,
                   d: float = 1.42,
                   bond_tol: float = 1.2) -> NDArray[np.int64]:
        """
        Bond list by distance: every pair strictly closer than d * bond_tol (periodic KD-tree).

        Parameters
        ----------
        d : float, optional
            Reference bond length, Angstrom. Default 1.42.
        bond_tol : float, optional
            Tolerance factor on d. Default 1.2.

        Returns
        -------
        (M, 2) ndarray of int
            Rows (i, j) with i < j, sorted by i then j; (0, 2) if there are no bonds.
        """
        cutoff = d * bond_tol
        periodic = np.asarray(self.periodic, dtype=bool)
        # a non-periodic axis gets a box three times longer: no image ever comes closer than the atom itself
        boxsize = np.where(periodic, self.Box, 3*self.Box)
        coords = np.where(periodic, np.mod(self._coord, self.Box), self._coord) # boxsize= requires [0, L)

        tree = KDTree(coords, boxsize=boxsize)
        pairs = tree.query_pairs(r=cutoff, output_type='ndarray') # every pair closer than cutoff, i < j

        # query_pairs keeps d == cutoff: drop those, bonding is strictly below the cutoff
        delta = coords[pairs[:, 0]] - coords[pairs[:, 1]]
        delta -= boxsize*np.round(delta/boxsize)
        pairs = pairs[np.linalg.norm(delta, axis=1) < cutoff]

        order = np.lexsort((pairs[:, 1], pairs[:, 0])) # by i, then j
        return pairs[order]

    @classmethod
    def generate(cls,
                 generator: CarbonGenerator,
                 target_density_deviation: float = 0.05,
                 max_carbon_attempts: int = 10,
                 bond_dist: float = 1.42,
                 bond_tol: float = 1.2) -> "CarbonStructure":
        """
        Run a carbon generator until the target density is met, keeping the best attempt.

        The generator is rerun (each time from an empty box) until the relative density error is
        below target_density_deviation, or max_carbon_attempts runs; the attempt with the smallest
        error is returned. Its bonds are then rebuilt by distance (_set_bonds), so a generated
        structure stores the same kind of graph as an imported one, and the potentials' types and
        parameters are stored for to_dict.

        Parameters
        ----------
        generator : CarbonGenerator
            A configured RingGenerator or RandomGenerator.
        target_density_deviation : float, optional
            Accepted relative error on the generator's max_rho. Default 0.05.
        max_carbon_attempts : int, optional
            Maximum number of runs. Default 10.
        bond_dist : float, optional
            Reference bond length of the rebuilt graph, Angstrom. Default 1.42.
        bond_tol : float, optional
            Bonds are pairs closer than bond_dist * bond_tol. Default 1.2.

        Returns
        -------
        CarbonStructure
        """
        import copy

        best_error = float('inf')
        best_coords = None
        best_adjacency = None
        best_bonds = None

        for cycle in range(1, max_carbon_attempts + 1):
            generator._generate()
            density_error = abs(generator.density - generator.max_rho) / generator.max_rho

            if density_error < best_error:
                best_error = density_error
                best_coords = generator.coordinates.copy()
                best_adjacency = copy.deepcopy(generator.adjacency)
                best_bonds = generator.bonds.copy()

            if density_error < target_density_deviation:
                print(f"Density target met at cycle {cycle}.")
                break
            print(f"Cycle {cycle}/{max_carbon_attempts}: density {generator.density:.3f} vs target {generator.max_rho:.2f} (error {100*density_error:.1f} %).")
        else:
            print(f"Max carbon attempts ({max_carbon_attempts}) exhausted. Returning best attempt (error {100*best_error:.1f} %).")

        obj = cls(Box=list(generator.Box),
                  periodic=list(generator.periodic))
        obj._coord = best_coords
        obj._adjacency = best_adjacency
        obj._bonds = best_bonds

        # re-bond by distance: the stored graph is the same kind as an imported structure's
        obj._bonds = obj._set_bonds(d=bond_dist, bond_tol=bond_tol)
        obj._adjacency = gf.build_adjacency(obj._bonds)

        if generator._atoms_potential_class is not None:
            obj._atoms_pot_type = generator._atoms_potential_class.pot_type
            obj._atoms_potential_parameters = generator._atoms_potential_class.get_potential_parameters()

        if generator._pores_potential_class is not None:
            obj._pores_pot_type = generator._pores_potential_class.pot_type
            obj._pores_potential_parameters = generator._pores_potential_class.get_potential_parameters()

        return obj
