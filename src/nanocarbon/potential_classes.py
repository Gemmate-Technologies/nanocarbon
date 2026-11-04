# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Energy functions for the carbon growth.

Atom-atom potentials (HarmonicPotential, kcal/mol; TersoffPotential, eV) and the
atom-pore potential (LJPoresPotential, in the unit of the atoms potential it is summed
with), behind the compute_energy interface RingGenerator calls to accept or reject a ring.
Lengths in Angstrom. PoresPotential is the base for a new atom-pore potential: supply
_energy and _energy_grad as functions of the distance to the pore surface.
"""
import numpy as np
from numpy.typing import NDArray

from . import geometry_functions as gf

class CustomPotential:
    """
    Base class of all potentials: stores the box and its periodicity, and routes
    compute_energy to the subclass's _compute_energy.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis; default periodic on all three.
    """

    def __init__(self,
                 Box: list[float],
                 periodic: list[int] = [1, 1, 1]) -> None:

        self.set_box(Box, periodic)
        self.set_potential_parameters()

    def set_box(self,
               Box: list[float],
               periodic: list[int]) -> None:
        """
        Set the box and its periodicity.

        Parameters
        ----------
        Box : list of float
            Box edges, Angstrom.
        periodic : list of int
            Periodic flags per axis.
        """
        self.Box = np.array(Box, dtype = float)
        self.periodic = np.array(periodic, dtype = int)

    def set_potential_parameters(self, *args, **kwargs):
        """
        Set the potential parameters; implemented by each subclass.

        Parameters
        ----------
        *args, **kwargs
            The subclass's parameters.
        """
        raise NotImplementedError("Class does not consider setting of parameters")

    def get_potential_parameters(self, *args, **kwargs):
        """
        Return the potential parameters as a dict; implemented by each subclass.

        Parameters
        ----------
        *args, **kwargs
            Unused.
        """
        raise NotImplementedError("Select specific potential class")

    def compute_energy(self, *args, **kwargs):
        """
        Energy of a configuration; the arguments and the return value are those of the
        subclass's _compute_energy.

        Parameters
        ----------
        *args, **kwargs
            Passed to _compute_energy.

        Returns
        -------
        float, or (float, (3,) ndarray)
            The energy (atoms potentials), or the energy and its gradient (pore potentials).
        """
        return self._compute_energy(*args, **kwargs)

    def _compute_energy(self, *args, **kwargs):
        """
        Energy of a configuration; implemented by each subclass.

        Parameters
        ----------
        *args, **kwargs
            The subclass's arguments.
        """
        raise NotImplementedError("Potential not set")

    def __str__(self) -> str:
        """Short name of the potential."""
        return("General potential superclass")

class PoresPotential(CustomPotential):
    """
    Base class of the atom-pore potentials.

    The energy of a point is the sum over the pores of _energy(d), with d the minimum-image
    distance from the point to the pore surface. A subclass supplies _energy and
    _energy_grad; keep the repulsion singular at d -> 0, or it cannot enforce a standoff.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis; default periodic on all three.
    """

    def __init__(self,
                 Box: list[float],
                 periodic: list[int] = [1,1,1]) -> None:

        super().__init__(Box, periodic)
        self.set_coords()

    def __str__(self) -> str:
        """Short name of the potential."""
        return("non-specific pores potential class")

    def set_coords(self,
                   pores_coord: NDArray[np.float64] | None = None,
                   pores_radius: NDArray[np.float64] | None= None) -> None:
        """
        Set the pores the potential acts with.

        Parameters
        ----------
        pores_coord : (P, 3) ndarray, optional
            Pore centers, Angstrom. None means no pores.
        pores_radius : (P,) ndarray, optional
            Pore radii, Angstrom. None means no pores.
        """

        if pores_coord is None:
            self.coord_pores = np.empty((0,3), dtype=float)
        else:
            self.coord_pores = pores_coord

        if pores_radius is None:
            self.R_pores = np.empty(0, dtype=float)
        else:
            self.R_pores = pores_radius

    def _energy(self, d):
        """
        Energy of a point at distances d from the pore surfaces; supplied by the subclass.

        Parameters
        ----------
        d : (P,) ndarray
            Distance from the point to each pore surface, Angstrom.

        Returns
        -------
        (P,) ndarray
            Energy contribution of each pore.
        """
        raise NotImplementedError("Select specific potential class")

    def _energy_grad(self, d):
        """
        Derivative of _energy with respect to d; supplied by the subclass.

        Parameters
        ----------
        d : (P,) ndarray
            Distance from the point to each pore surface, Angstrom.

        Returns
        -------
        (P,) ndarray
            dE/dd of each pore.
        """
        raise NotImplementedError("Select specific potential class")

    def _compute_energy(self,
                        points: NDArray[np.float64] | None = None,
                        *args, **kwargs) -> tuple[float, NDArray[np.float64]]:
        """
        Pore energy of a set of points, and its gradient.

        Each point interacts with every pore through the minimum-image distance to the pore
        surface; the contributions add up. A point on or inside a pore surface (distance
        below 1e-6 A), or a NaN, makes the energy infinite, so the placement is rejected.

        Parameters
        ----------
        points : (N, 3) ndarray, optional
            Coordinates, Angstrom. None or empty gives zero energy.
        *args, **kwargs
            Ignored; they keep the call signature common to all potentials.

        Returns
        -------
        E : float
            Summed energy, in the unit of the well depth (inf on contact).
        dE : (3,) ndarray
            Gradient with respect to the point coordinates, summed over the points (inf on
            contact).
        """
        E = 0
        dE = np.zeros(3)

        # safety check
        if self.coord_pores is None or len(self.coord_pores) == 0:
            return 0.0, np.zeros(3)
        if points is None: points = np.array([])

        for point in points:
            center_to_center_dist = gf.distance(point, self.coord_pores, self.Box, self.periodic)
            surface_to_surface_dist = center_to_center_dist - self.R_pores.ravel() # point-to-surface distance

            # Inf potential in the case of a point on the pores surface
            if np.any(surface_to_surface_dist < 1e-6):
                E = np.inf
                dE = np.inf*np.ones(3)
                break

            # fold the pores onto the image 'point' actually sees: differencing against the raw
            # centers would leave dD non-unit (and mis-directed) for a pore across a boundary,
            # since center_to_center_dist is already a minimum-image distance.
            folded_pores = gf.fold(point, self.coord_pores, self.Box, self.periodic)
            dD = (point - folded_pores)/center_to_center_dist[:, None]

            # np.sum: superposition of the contributions of all the pores.
            E_p = np.sum(self._energy(surface_to_surface_dist))

            dE_p = np.sum(dD * self._energy_grad(surface_to_surface_dist)[:,np.newaxis], axis = 0)

            # Inf potential in the case of NaN
            if np.any(np.isnan(E_p)) or np.any(np.isnan(dE_p)):
                E = np.inf
                dE = np.inf*np.ones(3)
                break
            E += E_p
            dE += dE_p

        return E, dE

class LJPoresPotential(PoresPotential):
    """
    Lennard-Jones wall around each pore, summed over the pores:
    E(d) = 4 A [(B/d)^e1 - (B/d)^e2], with d the distance from the pore surface.

    For 12-6 the minimum sits at 2^(1/6) B from the surface, which sets the carbon standoff.
    The LAMMPS anneal uses the same function as a wall/region, cut at that minimum (WCA).

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis; default periodic on all three.
    """
    def __str__(self):
        """Short name of the potential."""
        return("lj pores potential")

    def set_potential_parameters(self,
                                 A: float = 1e-1,
                                 B: float = 1.0,
                                 e1: int = 12,
                                 e2: int = 6,
                                 dict_params: dict | None = None,
                                 *args, **kwargs) -> None:
        """
        Set the wall parameters.

        Parameters
        ----------
        A : float, optional
            Well depth, in the energy unit of the atoms potential it is summed with (kcal/mol
            for harmonic, eV for tersoff): the scale of E_max. The LAMMPS writers convert it
            to eV with io_functions.WALL_UNIT_TO_EV. Default 0.1.
        B : float, optional
            Length scale, Angstrom: the wall crosses zero at d = B. Default 1.0.
        e1, e2 : int, optional
            Repulsive and attractive exponents. Default 12 and 6; LAMMPS has walls for
            (12, 6) and (9, 3) only.
        dict_params : dict, optional
            Parameters as returned by get_potential_parameters (e.g. read from a saved
            structure); when given, the other arguments are ignored.
        *args
            Ignored.
        **kwargs
            Not accepted: any keyword left over is an unknown parameter.

        Raises
        ------
        ValueError
            On an unknown keyword parameter.
        """
        if len(kwargs)>0:
            raise ValueError(f"unknown potential parameters: {list(kwargs)}")
        # parameters from a dict (e.g. a saved structure)
        if dict_params is not None:
            for key, value in dict_params.items():
                setattr(self, key, value)
            self.pot_type = "lj"
            return
        # otherwise, from the arguments
        self.A  = A
        self.B = B
        self.e1 = e1
        self.e2 = e2
        self.pot_type = "lj"

    def get_potential_parameters(self) -> dict:
        """
        Current parameters, for storing a structure and reloading it.

        Returns
        -------
        dict
            {'A', 'B', 'e1', 'e2'}; pass it back as set_potential_parameters(dict_params=...).
        """
        dict_params = {
            'A' : self.A,
            'B' : self.B,
            'e1' : self.e1,
            'e2' : self.e2,
            }
        return dict_params

    def _energy(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Wall energy of a point, pore by pore.

        Parameters
        ----------
        d : (P,) ndarray
            Distance from the point to each pore surface, Angstrom.

        Returns
        -------
        (P,) ndarray
            4 A [(B/d)^e1 - (B/d)^e2] for each pore.
        """
        return 4*self.A*( (self.B/d)**self.e1 - (self.B/d)**self.e2 )

    def _energy_grad(self, d: NDArray[np.float64]) -> NDArray[np.float64] :
        """
        Analytical derivative of _energy with respect to d.

        Parameters
        ----------
        d : (P,) ndarray
            Distance from the point to each pore surface, Angstrom.

        Returns
        -------
        (P,) ndarray
            dE/dd for each pore.
        """
        return 4*self.A*(self.e2*self.B**self.e2/d**(self.e2+1) - self.e1*self.B**self.e1/d**(self.e1+1))

class TersoffPotential(CustomPotential):
    """
    Tersoff bond-order potential for carbon, used as a placement criterion; energies in eV.

    It is evaluated one-sided: for a new atom i it sums 1/2 f_C(r_ij) [f_R(r_ij) +
    b_ji f_A(r_ij)] over the atoms j within rmax, with the bond order b_ji from the angles
    at the existing atom j only. The b_ij half and the change of the existing bonds' bond
    orders are not computed, so the result is not the Tersoff energy change and is not
    comparable with a LAMMPS Tersoff energy.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis; default periodic on all three.
    """
    def __init__(self,
                 Box: list[float],
                 periodic: list[int] = [1,1,1]) -> None :
        super().__init__(Box, periodic)
        self.pot_type = "tersoff"

    def __str__(self) -> str:
        """Short name of the potential."""
        return("Tersoff potential class")

    def set_potential_parameters(self,
                                 rmin: float = 1.8,
                                 rmax: float = 2.1,
                                 lambda1: float = 3.487,
                                 lambda2: float = 2.2119,
                                 A: float = 1393.6,
                                 B: float = 346.70,
                                 n: float = 0.727,
                                 beta: float =  1.5724e-7,
                                 c: float = 38049,
                                 d: float = 4.384,
                                 angular0: float = -0.57058,
                                 dict_params:dict | None = None,
                                 *args, **kwargs) -> None:
        """
        Set the potential parameters. The defaults are the C-C parameters of J. Tersoff,
        Phys. Rev. B 39, 5566 (1989), https://doi.org/10.1103/PhysRevB.39.5566, as
        implemented: B, lambda1 and n are rounded and d is 4.384 (published 4.3484).

        Parameters
        ----------
        rmin, rmax : float, optional
            Cutoff window, Angstrom: f_C is 1 below rmin and 0 above rmax.
        lambda1, lambda2 : float, optional
            Decay constants of the repulsive and attractive terms, 1/Angstrom.
        A, B : float, optional
            Prefactors of the repulsive and attractive terms, eV.
        n, beta : float, optional
            Bond-order exponent and prefactor.
        c, d, angular0 : float, optional
            Angular function g(theta) parameters; angular0 is h, the preferred cos(theta).
        dict_params : dict, optional
            Parameters as returned by get_potential_parameters; when given, the other
            arguments are ignored.
        *args
            Ignored.
        **kwargs
            Not accepted: any keyword left over is an unknown parameter.

        Raises
        ------
        ValueError
            On an unknown keyword parameter.
        """
        if len(kwargs)>0:
            raise ValueError(f"unknown potential parameters: {list(kwargs)}")

        # set parameters from input dictionary
        if dict_params is not None:
            for key, value in dict_params.items():
                setattr(self, key, value)
            return

        # default model parameters for C-C interaction
        self.rmin = rmin
        self.rmax = rmax
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.A = A
        self.B = B
        self.n = n
        self.beta = beta
        self.c = c
        self.d = d
        self.angular0 = angular0

    def get_potential_parameters(self) -> dict:
        """
        Current parameters, for storing a structure and reloading it.

        Returns
        -------
        dict
            One entry per parameter of set_potential_parameters; pass it back as
            set_potential_parameters(dict_params=...).
        """
        dict_params = {
            'rmin' : self.rmin,
            'rmax' : self.rmax,
            'lambda1' : self.lambda1,
            'lambda2' : self.lambda2,
            'A' : self.A,
            'B' : self.B,
            'n' : self.n,
            'beta' : self.beta,
            'c' : self.c,
            'd' : self.d,
            'angular0' : self.angular0
            }
        return dict_params

    def _fR(self, r):
        """
        Repulsive pair term A exp(-lambda1 r).

        Parameters
        ----------
        r : float or ndarray
            Distance, Angstrom.

        Returns
        -------
        float or ndarray
            Energy, eV.
        """
        return self.A*np.exp(-r*self.lambda1)
    def _fA(self, r):
        """
        Attractive pair term -B exp(-lambda2 r).

        Parameters
        ----------
        r : float or ndarray
            Distance, Angstrom.

        Returns
        -------
        float or ndarray
            Energy, eV.
        """
        return -self.B*np.exp(-r*self.lambda2)

    def _fC(self, r):
        """
        Smooth cutoff: 1 below rmin, 0 above rmax, a half cosine in between.

        Parameters
        ----------
        r : float or ndarray
            Distance, Angstrom.

        Returns
        -------
        float or ndarray
            Cutoff factor, shaped like r.
        """
        # vectorized cutoff: r can be scalar or array
        r = np.asarray(r)
        out = np.zeros_like(r, dtype=float)
        below = r <= self.rmin
        above = r >= self.rmax
        between = ~(below | above)
        out[below] = 1.0
        out[above] = 0.0
        if np.any(between):
            x = (r[between] - self.rmin) / (self.rmax - self.rmin)
            out[between] = 0.5 * (1.0 + np.cos(np.pi * x))
        # if scalar input, return scalar
        return out.item() if out.shape == () else out

    def _b(self, csi):
        """
        Bond order (1 + beta^n csi^n)^(-1/(2n)).

        Parameters
        ----------
        csi : float
            Coordination term: sum of f_C g(theta) over the other neighbors; negative
            values are clipped to 0.

        Returns
        -------
        float
            Bond order, at most 1.
        """
        csi = max(csi, 0.0) # prevents NaN
        return 1.0*(1.0+(self.beta**self.n)*(csi**self.n))**(-1/(2*self.n))

    def _g(self, theta):
        """
        Angular function 1 + (c/d)^2 - c^2 / (d^2 + (h - cos theta)^2), with h = angular0.

        Parameters
        ----------
        theta : float
            Bond angle, radians.

        Returns
        -------
        float
        """
        return 1.0 + (self.c/self.d)**2 - self.c**2/(self.d**2+(self.angular0-np.cos(theta))**2)


    def _compute_energy_per_atom(self,
                                 point: NDArray[np.float64],
                                 coord: NDArray[np.float64]) -> float:
        """
        One-sided Tersoff energy of a new atom (see the class docstring).

        Sums 1/2 f_C(r_ij) [f_R(r_ij) + b_ji f_A(r_ij)] over the atoms j of coord within
        rmax of point, with b_ji from the angles k-j-point at j, over the neighbors k of j.

        Parameters
        ----------
        point : (3,) ndarray
            The new atom, Angstrom.
        coord : (N, 3) ndarray
            Atoms it interacts with, Angstrom.

        Returns
        -------
        float
            Energy, eV; 0 when no atom is within rmax.
        """
        if not (self.Box is not None):
            raise RuntimeError("Set Box first")
        if not (self.periodic is not None):
            raise RuntimeError("Set PBC first")


        # checking if the distances considered are within bounds: [1e-12 and rmax] A
        dist = gf.distance(point, coord, self.Box, self.periodic)
        mask = np.where((dist > 1e-12) & (dist <= self.rmax))[0]
        # no atoms in range -> return no contribution to potential
        if mask.size == 0:
            return 0.0

        E = 0.0
        fails = 0
        # loop over the atoms j within range of the new atom
        for elem in mask:
            csi = 0 # coordination term of j (elem)
            d_ij = dist[elem]
            cutoff = self._fC(d_ij)

            C_folded = gf.fold(coord[elem,:], point, self.Box, self.periodic)

            # the neighbors k of j: each forms an angle k-j-point
            dist_angles = gf.distance(coord[elem,:], coord, self.Box, self.periodic)
            mask_angles = np.setdiff1d(np.nonzero(dist_angles <= self.rmax)[0], [elem])

            folded_angles = gf.fold(coord[elem,:], coord, self.Box, self.periodic)

            for elem_angles in mask_angles:
                # angle at j between k (A_folded) and the new atom (C_folded)
                A_folded = folded_angles[elem_angles,:]
                theta = gf.angle_three_points(A_folded, coord[elem,:], C_folded)
                if np.isnan(theta):
                    fails += 1
                    continue

                d_ik = dist_angles[elem_angles]
                cutoff2 = self._fC(d_ik)
                csi += cutoff2*1.0*self._g(theta)

            E += 0.5*cutoff*(self._fR(d_ij)+self._b(csi)*self._fA(d_ij))

        return E

    def _compute_energy(self,
                        new_idxes: list | None = None,
                        coord: NDArray[np.float64] | None = None,
                        neighList: list | None = None,
                        neigh_exclusion: list | None = None,
                        generation:bool = False,
                        *args, **kwargs) -> float:
        """
        Placement energy of new atoms, eV.

        Each new atom is evaluated alone against the atoms of neighList (the other new atoms
        are not included). With generation=True the atoms of neigh_exclusion are removed from
        neighList first, for the whole energy: the anchors and the atoms within
        lj_exclusion_bondnum bonds of them do not enter it.

        Parameters
        ----------
        new_idxes : list of int, optional
            Rows of coord to evaluate. Default none (zero energy).
        coord : (N, 3) ndarray, optional
            Coordinates of the existing and the new atoms, Angstrom.
        neighList : list of int, optional
            Rows of coord that may interact with the new atoms.
        neigh_exclusion : list of int, optional
            Rows removed from neighList when generation is True.
        generation : bool, optional
            Apply neigh_exclusion. Default False.
        *args, **kwargs
            Ignored (e.g. attraction_list, used by HarmonicPotential only).

        Returns
        -------
        float
            Summed energy of the new atoms, eV.
        """
        if new_idxes is None: new_idxes = []
        if coord is None: coord = np.empty((0, 3))
        if neighList is None: neighList = []
        if neigh_exclusion is None: neigh_exclusion = []

        E = 0
        if generation is True:
            neighList = list(set(neighList)-set(neigh_exclusion))
        for idx in new_idxes:
            point = coord[idx,:].copy()
            adjusted_coord = np.delete(coord, idx, axis = 0)
            adjusted_neighList = [item if (item <= idx) else item-1 for item in neighList]

            E += self._compute_energy_per_atom(point, adjusted_coord[adjusted_neighList, :])

        return E

class HarmonicPotential(CustomPotential):
    """
    Harmonic bonded terms plus non-bonded Lennard-Jones, energies in kcal/mol.

    Bonded: bond stretch, angle bend and a dihedral term toward flat sheets, over the atoms
    within rmax = 1.8 A. Non-bonded: a 12-6 LJ, plus two growth aids switched on by
    generation=True: a 9-3 attraction toward unsaturated atoms (on by default) and a
    short-range exponential repulsion (off by default). The default parameters are working
    values for the growth, not a fitted force field.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom.
    periodic : list of int, optional
        Periodic flags per axis; default periodic on all three.
    """
    def __init__(self,
                 Box: list[float],
                 periodic: list[int] = [1,1,1]) -> None:
        super().__init__(Box, periodic)
        self.pot_type = "harmonic"

    def __str__(self):
        """Short name of the potential."""
        return("Harmonic potential class")

    def set_potential_parameters(self,
                                 r0: float = 1.42,
                                 theta0: float = np.pi*2.0/3.0,
                                 dihedral0: float = 0.0,
                                 LJ_eps: float = 0.0684,
                                 LJ_sigma: float = 3.407,
                                 k_r: float = 1037.79,
                                 k_theta: float = 138.3724,
                                 k_dihedral: float = 1,
                                 nonBondedAttraction: bool = True,
                                 nonBondedAttraction_A: float = 0.5,
                                 shortRangeRepulsion: bool = False,
                                 shortRangeRepulsion_A: float = 1e5,
                                 shortRangeRepulsion_B: float = 0.4,
                                 dict_params: dict | None = None,
                                 *args, **kwargs) -> None:
        """
        Set the potential parameters. Lengths in Angstrom, angles in radians, energies in
        kcal/mol.

        Parameters
        ----------
        r0 : float, optional
            Equilibrium C-C bond length. Default 1.42.
        theta0 : float, optional
            Equilibrium C-C-C angle. Default 2 pi / 3.
        dihedral0 : float, optional
            Equilibrium dihedral angle. Default 0 (flat sheets).
        LJ_eps : float, optional
            12-6 LJ well depth. Default 0.0684.
        LJ_sigma : float, optional
            12-6 LJ zero-crossing distance. Default 3.407.
        k_r : float, optional
            Bond-stretch constant, kcal/mol/A^2. Default 1037.79.
        k_theta : float, optional
            Angle-bend constant, kcal/mol/rad^2. Default 138.3724.
        k_dihedral : float, optional
            Dihedral constant of 1/2 k sin^2(phi - dihedral0), kcal/mol. Default 1.
        nonBondedAttraction : bool, optional
            9-3 attraction toward unsaturated atoms during generation, 4 A [(r0/d)^9 -
            (r0/d)^3]: it pulls open fronts together to close planes. Default True.
        nonBondedAttraction_A : float, optional
            Its prefactor A. Default 0.5.
        shortRangeRepulsion : bool, optional
            Exponential repulsion A exp(-d/B) from every neighbor during generation, up to
            2 A. Default False.
        shortRangeRepulsion_A : float, optional
            Its prefactor A. Default 1e5.
        shortRangeRepulsion_B : float, optional
            Its decay length B, Angstrom. Default 0.4.
        dict_params : dict, optional
            Parameters as returned by get_potential_parameters; when given, the other
            arguments are ignored.
        *args
            Ignored.
        **kwargs
            Not accepted: any keyword left over is an unknown parameter.

        Raises
        ------
        ValueError
            On an unknown keyword parameter.
        """
        if len(kwargs)>0:
            raise ValueError(f"unknown potential parameters: {list(kwargs)}")

        if dict_params is not None:
            for key, value in dict_params.items():
                setattr(self, key, value)
            return
        # bonded terms act on the atoms within rmax of the new atom
        self.rmax = 1.8
        self.r0 = r0
        self.theta0 = theta0
        self.dihedral0 = dihedral0

        # non-bonded potential parameters
        self.LJ_eps = LJ_eps #kcal/mol
        self.LJ_sigma = LJ_sigma #A
        # bond parameters
        self.k_r = k_r #kcal/mol/A^2
        self.k_theta = k_theta #kcal/mol/rad^2
        self.k_dihedral = k_dihedral #kcal/mol (multiplies sin^2)

        self.nonBondedAttraction = nonBondedAttraction
        self.nonBonded_A = nonBondedAttraction_A
        self.shortRangeRepulsion = shortRangeRepulsion
        self.short_A = shortRangeRepulsion_A
        self.short_B = shortRangeRepulsion_B

    def get_potential_parameters(self) -> dict:
        """
        Current parameters, for storing a structure and reloading it.

        Returns
        -------
        dict
            The stored attribute names (rmax, r0, ..., nonBonded_A, short_A, short_B; these
            differ from some set_potential_parameters keywords); pass it back as
            set_potential_parameters(dict_params=...).
        """
        dict_params = {
            'rmax' : self.rmax,
            'r0' : self.r0,
            'theta0' : self.theta0,
            'dihedral0' : self.dihedral0,
            'LJ_eps' : self.LJ_eps,
            'LJ_sigma' : self.LJ_sigma,
            'k_r' : self.k_r,
            'k_theta' : self.k_theta,
            'k_dihedral' : self.k_dihedral,
            'nonBondedAttraction' : self.nonBondedAttraction,
            'nonBonded_A' : self.nonBonded_A,
            'shortRangeRepulsion' : self.shortRangeRepulsion,
            'short_A' : self.short_A,
            'short_B' : self.short_B,
            }
        return dict_params

    def _nonbonded(self,
                   D_LJ: NDArray,
                   D_rep: NDArray,
                   D_attr: NDArray,
                   use_repulsion: bool,
                   use_attraction:bool,
                   ) -> float:
        """
        Non-bonded energy of one new atom, kcal/mol, from its distances to three atom sets.

        The cutoffs (11 A for LJ, 10 A for the 9-3 attraction, 2 A for the repulsion) are
        nominal: the distances come from the anchor's neighbor list, complete to about 5 A
        from the anchor (auxiliary_classes.NeighborhoodManager).

        Parameters
        ----------
        D_LJ : (M,) ndarray
            Distances to the atoms of the 12-6 LJ, Angstrom.
        D_rep : (M,) ndarray
            Distances to all neighbors, for the short-range repulsion and the overlap check.
        D_attr : (M,) ndarray
            Distances to the unsaturated atoms of the 9-3 attraction.
        use_repulsion, use_attraction : bool
            Include the short-range repulsion / the 9-3 attraction.

        Returns
        -------
        float
            Summed energy; inf if any neighbor is within 1e-5 A.
        """
        def LJ_potential(d):
            """
            12-6 LJ, zero beyond 11 A.

            Parameters
            ----------
            d : (M,) ndarray
                Distances, Angstrom.

            Returns
            -------
            (M,) ndarray
                Energies, kcal/mol.
            """
            E = 4*self.LJ_eps*( (self.LJ_sigma/d)**12 - (self.LJ_sigma/d)**6 )
            E[d > 11.0] = 0.0
            return E

        def short_range_repulsion(d, A = self.short_A, B = self.short_B):
            """
            A exp(-d/B), zero beyond 2 A.

            Parameters
            ----------
            d : (M,) ndarray
                Distances, Angstrom.
            A : float, optional
                Prefactor, kcal/mol.
            B : float, optional
                Decay length, Angstrom.

            Returns
            -------
            (M,) ndarray
                Energies, kcal/mol.
            """
            rep = A*np.exp(-d/B)
            rep[d > 2.0] = 0.0  # beyond 2 Å negligible
            return rep

        def short_range_attraction(d, A = self.nonBonded_A):
            """
            9-3 attraction 4 A [(r0/d)^9 - (r0/d)^3]: zero at r0, minimum at 3^(1/6) r0, zero
            beyond 10 A.

            Parameters
            ----------
            d : (M,) ndarray
                Distances, Angstrom.
            A : float, optional
                Prefactor, kcal/mol.

            Returns
            -------
            (M,) ndarray
                Energies, kcal/mol.
            """
            attr = 4*A*( (self.r0/d)**9 - (self.r0/d)**3 )
            attr[d > 10.0] = 0.0  # beyond 10 Å negligible
            return attr

        if np.any(D_rep <= 1e-5):
            return np.inf

        E = np.sum(LJ_potential(D_LJ))

        if use_repulsion:
            E += np.sum(short_range_repulsion(D_rep))

        if use_attraction:
            E += np.sum(short_range_attraction(D_attr))

        return E

    def _linearBonds(self, D: NDArray) -> float:
        """
        Bond-stretch energy 1/2 k_r (r - r0)^2 over the neighbors within rmax.

        Parameters
        ----------
        D : (M,) ndarray
            Distances from the new atom to its neighbors, Angstrom.

        Returns
        -------
        float
            Energy, kcal/mol.
        """
        # masking according to a distance threshold
        mask = np.nonzero(D <= self.rmax)[0]
        E = np.sum(0.5*self.k_r*(D[mask]-self.r0)**2)

        return E

    def _angularBonds(self,
                      point: NDArray,
                      coord: NDArray,
                      D: NDArray) -> float:
        """
        Angle and dihedral energy of a new atom.

        For every atom j within rmax of the new atom and every atom k within rmax of j, adds
        1/2 k_theta (theta - theta0)^2 for the angle k-j-point, and for every atom l within
        rmax of k, 1/2 k_dihedral sin^2(phi - dihedral0) for the dihedral l-k-j-point.

        Parameters
        ----------
        point : (3,) ndarray
            The new atom, Angstrom.
        coord : (M, 3) ndarray
            Its neighbors, Angstrom.
        D : (M,) ndarray
            Distances from point to coord, Angstrom.

        Returns
        -------
        float
            Energy, kcal/mol; inf on a degenerate angle or dihedral.
        """

        mask = np.nonzero(D <= self.rmax)[0]
        E = 0

        # the atoms j bonded to the new atom
        for elem in mask:
            C_folded = gf.fold(coord[elem,:], point, self.Box, self.periodic)
            dist_angles = gf.distance(coord[elem,:], coord, self.Box, self.periodic)
            mask_angles = np.setdiff1d(np.nonzero(dist_angles <= self.rmax)[0], [elem])

            folded_angles = gf.fold(coord[elem,:], coord, self.Box, self.periodic)

            for elem_angles in mask_angles:
                # angle at j between k (A_folded) and the new atom (C_folded)
                A_folded = folded_angles[elem_angles,:]
                theta = gf.angle_three_points(A_folded, coord[elem,:], C_folded)
                if np.isnan(theta):
                    return np.inf

                E += 0.5*self.k_theta*(theta-self.theta0)**2

                # dihedral: every neighbor l of k closes the chain l-k-j-point
                dist_dihedral = gf.distance(coord[elem_angles,:], coord, self.Box, self.periodic)
                mask_dihedral = np.setdiff1d(np.nonzero(dist_dihedral <= self.rmax)[0], [elem, elem_angles])

                folded_dihedral = gf.fold(A_folded, coord, self.Box, self.periodic)

                for elem_dihedral in mask_dihedral:
                    D_folded = folded_dihedral[elem_dihedral,:]
                    dihedral = gf.dihedral_angle(A_folded, coord[elem,:], C_folded, D_folded)
                    if np.isnan(dihedral):
                        return np.inf
                    E += 0.5*self.k_dihedral*np.sin(dihedral-self.dihedral0)**2

        return E

    def _compute_energy(self,
                        new_idxes: list | None = None,
                        coord: NDArray | None = None,
                        neighList: list | None = None,
                        neigh_exclusion: list | None = None,
                        attraction_list: list | None = None,
                        generation: bool = False,
                        *args, **kwargs) -> float:
        """
        Placement energy of new atoms, kcal/mol.

        Each new atom is evaluated against neighList and the other new atoms: the bonded
        terms over all of them within rmax; the 12-6 LJ over neighList minus neigh_exclusion
        and attraction_list; with generation=True, the 9-3 attraction over attraction_list
        and, if enabled, the short-range repulsion over all neighbors.

        Parameters
        ----------
        new_idxes : list of int, optional
            Rows of coord to evaluate. Default none (zero energy).
        coord : (N, 3) ndarray, optional
            Coordinates of the existing and the new atoms, Angstrom.
        neighList : list of int, optional
            Existing atoms that may interact with the new ones.
        neigh_exclusion : list of int, optional
            Atoms left out of the 12-6 LJ: the anchors and the atoms within
            lj_exclusion_bondnum bonds of them.
        attraction_list : list of int, optional
            Unsaturated atoms: 9-3 attraction (with generation=True) instead of LJ.
        generation : bool, optional
            Switch on the growth aids (attraction, repulsion). Default False.
        *args, **kwargs
            Ignored.

        Returns
        -------
        float
            Summed energy, kcal/mol; inf on an overlap or a degenerate geometry.
        """
        # "new_idxes" are not included in "neighList" or "neigh_exclusion"
        if new_idxes is None or len(new_idxes) == 0:
            return 0.0
        if coord is None: coord = np.empty((0, 3))
        if neighList is None: neighList = []
        if neigh_exclusion is None: neigh_exclusion = []
        if attraction_list is None: attraction_list = []


        # outside generation the synthetic attraction and repulsion are off
        use_repulsion  = self.shortRangeRepulsion if generation else False
        use_attraction = self.nonBondedAttraction if generation else False

        E_total = 0.0

        new_idxes_set = set(list(new_idxes))

        # Looping over the items of "new_idxes" to evaluate the energy contribution
        for idx in new_idxes:
            # row idx is removed from coord below, so every index above it shifts down by one;
            # idx never appears in its own lists.
            adjusted_idxes = [item if (item < idx) else item-1 for item in new_idxes_set - set([idx])]

            adjusted_neighList = [item if item < idx else item - 1 for item in neighList] + adjusted_idxes
            # the other new atoms are not attraction targets
            adjusted_attraction_list = [item if (item < idx) else item-1 for item in attraction_list]

            adjusted_LJ_exclusion = [item if (item < idx) else item-1 for item in neigh_exclusion] + adjusted_idxes

            point = coord[idx,:].copy()
            adjusted_coord = np.delete(coord, idx, axis = 0)

            # distances for the bonded terms (and the short-range repulsion)
            D_neigh = gf.distance(point, adjusted_coord[adjusted_neighList, :], self.Box, self.periodic)

            # 12-6 LJ: neighbors minus the excluded and the unsaturated atoms
            adjusted_neighList_filtered = [item for item in set(adjusted_neighList)-set(adjusted_LJ_exclusion)-set(adjusted_attraction_list)]
            D_LJ = gf.distance(point, adjusted_coord[adjusted_neighList_filtered,: ], self.Box, self.periodic)
            # 9-3 attraction: the unsaturated atoms
            D_attr = gf.distance(point, adjusted_coord[adjusted_attraction_list,: ], self.Box, self.periodic)

            E_nonbonded = self._nonbonded(D_LJ,
                                          D_neigh,
                                          D_attr,
                                          use_repulsion= use_repulsion,
                                          use_attraction= use_attraction,
            )
            E_bond = self._linearBonds(D_neigh)
            E_angle = self._angularBonds(point, adjusted_coord[adjusted_neighList, :], D_neigh)

            E_total += E_nonbonded + E_bond + E_angle

        if np.any(np.isnan(E_total)) or np.any(np.isinf(E_total)):
            E_total = np.inf

        return E_total
