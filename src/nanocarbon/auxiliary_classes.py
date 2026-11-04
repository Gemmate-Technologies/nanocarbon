# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Cell lists for the carbon growth.

NeighborhoodManager keeps, for every atom, the atoms of the nearby cells of a uniform grid
under periodic boundary conditions, so the growth checks scan a few cells instead of the
whole structure. Post-processing does not use it: pair counts there come from a periodic
KD-tree.
"""
import numpy as np
from numpy.typing import NDArray


class NeighborhoodManager:
    """
    Cell lists over a grid of N_box**3 equal cells (cell numbers in the M matrix).

    An atom's neighbor list holds the atoms of its own cell and of the adjacent cell on its
    nearer side along each axis (box_identify): a half-shell, complete to half a cell, which
    is at least 5 A with cells of 10 A or more (geometry_functions.set_N_box). Meant for the
    short-range growth checks. Lists are reciprocal: adding an atom also lists it as a
    neighbor of the atoms it sees.

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

        self.MyBoxes = {}
        self.local_neighbors = {}
        self.M = None

        self.Box = np.array(Box)
        self.periodic = np.array(periodic)

    def assign_to_MyBoxes(self,
                          box_id: int,
                          atom_idx: NDArray) -> None:
        """
        Add atoms to a cell.

        Parameters
        ----------
        box_id : int
            Cell number (see set_M_matrix).
        atom_idx : int or ndarray of int
            Atom index, or indices, to add.
        """
        if not type(atom_idx) == np.ndarray:
            atom_idx = np.array([atom_idx], dtype = int)

        if box_id not in self.MyBoxes:
            self.MyBoxes[box_id] = np.array([atom_idx])
        else:
            self.MyBoxes[box_id] = np.unique(np.append(self.MyBoxes[box_id], atom_idx))

    def set_M_matrix(self, N_box: int) -> None:
        """
        Number the cells; must be called before any cell lookup.

        Parameters
        ----------
        N_box : int
            Cells per axis. M becomes the (N_box, N_box, N_box) matrix of cell numbers
            1..N_box**3, in Fortran order.
        """
        M = np.arange(1, N_box**3 + 1).reshape((N_box, N_box, N_box), order='F')
        self.M = M
        self.N_box = N_box

    def get_atoms_in_MyBoxes(self, box_id: int) -> NDArray:
        """
        Atoms in a cell.

        Parameters
        ----------
        box_id : int
            Cell number; an empty cell is created on first access.

        Returns
        -------
        ndarray of int
            Atom indices in the cell.
        """
        try:
            atom_list = self.MyBoxes[box_id]
        except KeyError:
            self.assign_to_MyBoxes(box_id, np.empty(0, dtype = int))
            atom_list = self.get_atoms_in_MyBoxes(box_id)

        return atom_list

    def get_M_matrix(self):
        """The cell-numbering matrix M (None before set_M_matrix)."""
        return self.M

    def set_local_neighbors(self,
                            atom_idx: int,
                            atoms_array: NDArray) -> None:
        """
        Make an atom and a set of atoms neighbors of each other (both lists are updated).

        Parameters
        ----------
        atom_idx : int
            The atom being added.
        atoms_array : int or ndarray of int
            Its neighbors; atom_idx is also added to each of their lists.
        """
        if not type(atoms_array) == np.ndarray:
            atoms_array = np.array([atoms_array])

        # Neighbor lists are kept as python sets: insertion is O(1), whereas growing
        # an ndarray one element at a time (np.append + np.unique per neighbor) made
        # the reciprocal loop below quadratic in the neighborhood size.
        atom_idx = int(atom_idx)
        self.local_neighbors.setdefault(atom_idx, set()).update(int(a) for a in atoms_array)
        # reciprocally, atom_idx becomes a neighbor of each atom in atoms_array
        for atom in atoms_array:
            self.local_neighbors.setdefault(int(atom), set()).add(atom_idx)

    def get_local_neighbors(self, atom_idx: int) -> NDArray:
        """
        Neighbors of an atom.

        Parameters
        ----------
        atom_idx : int
            Atom index.

        Returns
        -------
        ndarray of int
            Sorted neighbor indices; empty if the atom has none.
        """
        try:
            neigh = self.local_neighbors[atom_idx]
        except KeyError:
            return np.empty(0, dtype=int)
        # Sorted, so every consumer iterates neighbors in a deterministic order.
        return np.sort(np.fromiter(neigh, dtype=int, count=len(neigh)))

    def box_identify(self, coord: NDArray) -> tuple[NDArray, int]:
        """
        Cell of a point and the cells of its half-shell.

        Along each axis the half-shell takes the point's own cell and the adjacent cell on
        the side of the nearer face, wrapping at the box edge; a point within 0.01 A of a
        cell center takes both adjacent cells on that axis.

        Parameters
        ----------
        coord : (3,) ndarray
            Point coordinates, Angstrom.

        Returns
        -------
        neighBox : ndarray of int
            Numbers of the half-shell cells, own cell included.
        numBox : int
            Number of the cell containing the point.
        """

        if not (self.M is not None):
            raise RuntimeError("Set M matrix first")

        DB = self.Box/self.N_box

        num = np.floor(coord / DB).astype(int) + 1
        num = np.minimum(num, self.N_box)

        C = (num-1)*DB + DB/2 # center of the point's cell
        numBox = int(self.M[num[0]-1, num[1]-1, num[2]-1]) # number of the point's cell

        num2 = np.sign((coord-C)/(DB/2)).astype(int) # -1 for lower part of the box on the i-th axis, +1 for the upper part
        at_center_dims = np.isclose(coord, C, atol = 1e-2)
        mask = np.nonzero(at_center_dims)[0]

        for col in mask:
            num2 = np.tile(num2, (2, 1))
            half = num2.shape[0] // 2
            num2[:half, col] = -1
            # Assign +1 to the bottom half in the specified column
            num2[half:, col] = 1

        # Prepend a row of zeros
        num2 = np.vstack((np.zeros((1, 3), dtype=int), num2))

        # Extract unique values in each column
        X_vals = np.unique(num2[:, 0])
        Y_vals = np.unique(num2[:, 1])
        Z_vals = np.unique(num2[:, 2])

        # Create a 3D grid and flatten it
        X, Y, Z = np.meshgrid(X_vals, Y_vals, Z_vals, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()

        # Stack into a matrix and add `num`
        M2 = np.stack((X, Y, Z), axis=-1) + num

        # Replace 0s with N_box
        M2[M2 == 0] = self.N_box
        # Wrap-around: values > N_box become 1
        M2[M2 > self.N_box] = 1

        neighBox = self.M[M2[:,0]-1, M2[:,1]-1, M2[:,2]-1]

        # Get unique neighbor box indices
        neighBox = np.unique(neighBox)

        return neighBox, numBox

    def assign_atoms_to_box(self,
                            indeces: list[int],
                            coord: NDArray) -> None:
        """
        Put atoms in their cells and make each a neighbor of the atoms already in its
        half-shell.

        Parameters
        ----------
        indeces : list of int
            Atoms to add.
        coord : (N, 3) ndarray
            Coordinates the indices point into, Angstrom.
        """
        if not (self.M is not None):
            raise RuntimeError("Set M matrix first")

        for index in indeces:
            neighBox , box_id = self.box_identify(coord[index])
            self.assign_to_MyBoxes(box_id, np.array(index, dtype = int))
            #collect all the atoms in each neighBox
            neigh_atoms_list = [np.ravel(self.get_atoms_in_MyBoxes(b)) for b in neighBox]
            neigh_atoms = np.unique(np.concatenate(neigh_atoms_list))
            self.set_local_neighbors(index, np.setdiff1d(neigh_atoms,[index]))
