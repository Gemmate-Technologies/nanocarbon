# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Geometry helpers for generation and analysis.

Distances and folding under periodic boundary conditions (minimum image), angles and
dihedrals, ring templates and their placement, pore images for LAMMPS, voxel grids, the
bond graph and the primitive-ring test. Stateless functions on NumPy arrays; lengths in
Angstrom, angles in radians.
"""

import warnings
from itertools import product

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.transform import Rotation as R

def angle_three_points(A: NDArray,
                       B: NDArray,
                       C: NDArray) -> float:
    """
    Angle at B of the triangle A-B-C.

    Parameters
    ----------
    A, B, C : (3,) ndarray
        Point coordinates, Angstrom. With C=None, A and B are taken as the two vectors
        themselves (e.g. two plane normals).

    Returns
    -------
    float
        Angle in radians; nan if a side is shorter than 1e-8.
    """
    if C is None:
        V1 = A
        V2 = B
    else:
        V1 = A-B
        V2 = C-B

    norm1 = np.linalg.norm(V1)
    norm2 = np.linalg.norm(V2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return np.nan

    cos_angle = np.dot(V1,V2)/ (np.linalg.norm(V1)*np.linalg.norm(V2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # round-off can push it just past +-1
    theta = np.arccos(cos_angle)
    return theta

def _dihedral_normals(A: NDArray,
                      X: NDArray,
                      C: NDArray,
                      D: NDArray) -> tuple[NDArray | None, NDArray | None, float]:
    """
    Unit normals of the planes (A, X, C) and (D, A, X), and the cosine between them.

    Parameters
    ----------
    A, X, C, D : (3,) ndarray
        Atom coordinates, Angstrom; A-X is the edge the two planes share.

    Returns
    -------
    n1, n2 : (3,) ndarray or None
        Unit normals; None on a degenerate (collinear) plane.
    cos_phi : float
        Cosine of the dihedral angle, clipped to [-1, 1]; nan on a degenerate plane.
    """
    n1 = plane_normal(A, X, C)
    n2 = plane_normal(D, A, X)

    if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8:
        return None, None, np.nan

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return n1, n2, cos_phi

def dihedral_angle(A: NDArray,
                   X: NDArray,
                   C: NDArray,
                   D: NDArray) -> float:
    """
    Dihedral angle between the planes (A, X, C) and (D, A, X).

    Parameters
    ----------
    A, X, C, D : (3,) ndarray
        Atom coordinates, Angstrom; A-X is the edge the two planes share.

    Returns
    -------
    float
        Angle in radians; nan on a degenerate (collinear) plane.
    """
    _, _, cos_phi = _dihedral_normals(A, X, C, D)
    if np.isnan(cos_phi):           # degenerate plane
        return np.nan
    return np.arccos(cos_phi)

def set_N_box(Box: NDArray) -> int:
    """
    Number of neighbor-grid cells per axis.

    The same count splits every axis, so it is sized on the shortest edge: every cell is at
    least 10 A wide when the box allows.

    Parameters
    ----------
    Box : (3,) ndarray
        Box edges, Angstrom.

    Returns
    -------
    int
        floor(shortest edge / 10 A), at least 1.
    """
    return max(1, int(np.floor(np.min(Box)/10)))


def carbon_rings_template(n: int,
                          d: float
                          ) -> NDArray:
    """
    Regular polygon in the xy plane, centered on the origin.

    Parameters
    ----------
    n : int
        Number of atoms in the ring.
    d : float
        Side (bond) length, Angstrom.

    Returns
    -------
    (n, 3) ndarray
        Atom coordinates, in order around the ring.
    """
    theta = 2 * np.pi / n
    R = d / (2 * np.sin(np.pi / n))

    cc = np.array([
        [R * np.cos(i * theta),
         R * np.sin(i * theta),
         0.0]
        for i in range(n)
    ])

    return cc

def random_unit_vector() -> NDArray:
    """
    Random direction, uniformly distributed on the sphere.

    Returns
    -------
    (3,) ndarray
        Unit vector.
    """
    cc = np.random.normal(size = 3)
    return cc/np.linalg.norm(cc)

def plane_normal(p1: NDArray,
                 p2: NDArray,
                 p3: NDArray) -> NDArray:
    """
    Normal to the plane through three points.

    Parameters
    ----------
    p1, p2, p3 : (3,) ndarray
        Points on the plane.

    Returns
    -------
    (3,) ndarray
        (p2 - p1) x (p3 - p1), not normalized: zero for collinear points.
    """
    v1 = p2-p1
    v2 = p3-p1

    normal = np.cross(v1, v2)
    return normal

def align_ring_to_normal (template_coords: NDArray,
                          target_normal: NDArray) -> NDArray:
    """
    Rotate a planar ring so its normal points along a target direction.

    Parameters
    ----------
    template_coords : (n, 3) ndarray
        Planar ring; its normal is taken from rows 0, 1 and -1.
    target_normal : (3,) ndarray
        Target direction, any length. A zero vector (the normal of a degenerate plane) is
        replaced by a random direction.

    Returns
    -------
    (n, 3) ndarray
        The ring after the minimal rotation about the origin.
    """

    template_normal = plane_normal(template_coords[0, :], template_coords[1, :], template_coords[-1, :])
    template_normal /= np.linalg.norm(template_normal)
    if np.linalg.norm(target_normal) < 1e-8:
        # degenerate (collinear) plane: any orientation is as good, and NaN would drop the ring
        target_normal = random_unit_vector()
    target = target_normal / np.linalg.norm(target_normal)

    # Minimal rotation template_normal -> target (Rodrigues). Built directly
    # instead of R.align_vectors, which is underdetermined for a single vector
    # pair (rotation about the shared axis is free) and warns on every call.
    axis = np.cross(template_normal, target)
    s = np.linalg.norm(axis)
    c = float(np.dot(template_normal, target))
    if s < 1e-8:                                  # already (anti)parallel
        if c > 0:
            return template_coords.copy()         # identity
        perp = np.cross(template_normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-8:
            perp = np.cross(template_normal, [0.0, 1.0, 0.0])
        rot = R.from_rotvec(np.pi * perp / np.linalg.norm(perp))
    else:
        rot = R.from_rotvec(np.arctan2(s, c) * axis / s)

    return rot.apply(template_coords)

def distance(coord_single: NDArray,
             coord_multip: NDArray,
             Box: NDArray,
             periodic: NDArray | None = None
             ) -> NDArray:
    """
    Distances from one point to many, under the minimum-image convention.

    Parameters
    ----------
    coord_single : (3,) ndarray
        Reference point, Angstrom.
    coord_multip : (N, 3) or (3,) ndarray
        Points to measure, Angstrom.
    Box : (3,) ndarray
        Box edges, Angstrom.
    periodic : (3,) array of 0/1, optional
        Periodic flags per axis; default periodic on all three.

    Returns
    -------
    (N,) ndarray
        Distances, Angstrom.
    """
    if periodic is None:
        periodic = np.ones(3)
    coord_multip = np.atleast_2d(coord_multip) # ensures size is (N,3) always
    delta = coord_single - coord_multip
    delta -= Box*periodic*np.round(delta/Box)

    dist = np.linalg.norm(delta,axis=1)
    return dist

def fold(coord_ref: NDArray,
         coord_target: NDArray,
         Box: NDArray,
         periodic: NDArray | None = None
         )-> NDArray:
    """
    Periodic image of a point nearest to a reference point (minimum image).

    Use it to build a minimum-image *vector*: fold first, then subtract.

    Parameters
    ----------
    coord_ref : (3,) ndarray
        Reference point, Angstrom.
    coord_target : (3,) or (N, 3) ndarray
        Point(s) to fold, Angstrom.
    Box : (3,) ndarray
        Box edges, Angstrom.
    periodic : (3,) array of 0/1, optional
        Periodic flags per axis; only periodic axes are folded. Default all periodic.

    Returns
    -------
    ndarray, shaped like coord_target
        The folded coordinates.
    """
    if periodic is None:
        periodic = np.ones(3)
    return coord_target - Box * periodic * np.round((coord_target - coord_ref) / Box)

def axis_rotation(original_coords: NDArray,
                  rotation_axis: NDArray,
                  angle: float,
                  anchor_idx: int
                  ) -> NDArray:
    """
    Rotate a set of points about an axis through one of them.

    Parameters
    ----------
    original_coords : (N, 3) ndarray
        Points to rotate, Angstrom.
    rotation_axis : (3,) ndarray
        Rotation axis; must be a unit vector.
    angle : float
        Rotation angle, radians.
    anchor_idx : int
        Row of original_coords the axis passes through.

    Returns
    -------
    (N, 3) ndarray
        The rotated points.
    """
    flip_rot = R.from_rotvec(angle * rotation_axis)
    center_of_rotation = original_coords[anchor_idx, :].copy()
    new_ring = original_coords - center_of_rotation
    new_ring = flip_rot.apply(new_ring)
    new_ring += center_of_rotation

    return new_ring

def ring_rotation(ring_coords: NDArray,
                  current_vec: NDArray,
                  target_vec: NDArray,
                  ) -> NDArray:
    """
    Rotate points about the origin so that one direction turns into another.

    Parameters
    ----------
    ring_coords : (N, 3) ndarray
        Points to rotate.
    current_vec, target_vec : (3,) ndarray
        Unit vectors: current_vec is turned onto target_vec by the minimal rotation (180
        degrees about a perpendicular axis when they are anti-parallel).

    Returns
    -------
    (N, 3) ndarray
        The rotated points; ring_coords itself when the vectors are already parallel.
    """

    rotation_axis = np.cross(current_vec, target_vec)
    norm_axis = np.linalg.norm(rotation_axis)
    if norm_axis < 1e-8:
        # cross approx 0 for BOTH parallel and anti-parallel: split on the dot sign.
        if np.dot(current_vec, target_vec) > 0:
            aligned_ring = ring_coords # already aligned, don't do anything
        else:
            # anti-parallel: 180 deg about any axis perpendicular to current_vec
            perp = np.cross(current_vec, [1.0, 0.0, 0.0])
            if np.linalg.norm(perp) < 1e-8: # current_vec parallel to x, use y instead
                perp = np.cross(current_vec, [0.0, 1.0, 0.0])
            perp /= np.linalg.norm(perp)
            rot = R.from_rotvec(np.pi * perp)
            aligned_ring = rot.apply(ring_coords)
    else:
        rotation_axis /= norm_axis
        angle = np.arccos(np.clip(np.dot(current_vec, target_vec), -1.0, 1.0))
        rot = R.from_rotvec(angle * rotation_axis)
        aligned_ring = rot.apply(ring_coords)

    return aligned_ring

def apply_box_constraints(ring: NDArray,
                          Box: NDArray,
                          periodic: NDArray[np.int32] = np.ones(3, dtype=int) #default is periodic
                          ) -> NDArray:
    """
    Bring points back into the box: wrap along periodic axes, clip along the others.

    Parameters
    ----------
    ring : (N, 3) ndarray
        Points, Angstrom; modified in place.
    Box : (3,) ndarray
        Box edges, Angstrom.
    periodic : (3,) array of 0/1, optional
        Periodic flags per axis; default periodic on all three.

    Returns
    -------
    (N, 3) ndarray
        ring itself, wrapped or clipped.
    """
    for j in range(3):
        if periodic[j]:
            ring[:, j] = ring[:, j] % Box[j]
        else:
            ring[:, j] = np.clip(ring[:, j], 0.0, Box[j])
    return ring

def pore_images(coord: NDArray,
                R: NDArray,
                Box: NDArray,
                periodic: NDArray,
                reach: float = 0.0
                ) -> tuple[NDArray, NDArray, NDArray]:
    """
    Periodic images of the pores whose influence sphere reaches into the box.

    LAMMPS regions do not wrap, while the generation potential sees pores through the
    minimum-image convention; writing every image whose sphere of radius R + reach still
    touches the box (after a shift by one lattice vector) gives LAMMPS the same pores.
    Warns when an influence radius exceeds half the shortest edge: an atom can then sit
    within reach of two images of one pore, so LAMMPS sums more of the lattice than the
    generation potential did.

    Parameters
    ----------
    coord : (P, 3) ndarray
        Pore centers, Angstrom.
    R : (P,) ndarray
        Pore radii, Angstrom.
    Box : (3,) ndarray
        Box edges, Angstrom.
    periodic : (3,) array of int
        Periodic flags per axis; images are made only along periodic axes.
    reach : float, optional
        Wall cutoff measured outward from the pore surface, Angstrom. 0 for the bare sphere.

    Returns
    -------
    centers : (M, 3) ndarray
        Image centers, the original pores included (zero shift).
    radii : (M,) ndarray
        Image radii.
    parent : (M,) ndarray of int
        Index of the pore each image comes from.
    """
    coord = np.atleast_2d(np.asarray(coord, dtype = float))
    radii = np.asarray(R, dtype = float).ravel()
    Box = np.asarray(Box, dtype = float)
    periodic = np.asarray(periodic, dtype = int)

    influence = radii + reach

    # Beyond half the smallest edge an atom can lie within the cutoff of two images of the same
    # pore: LAMMPS then sums a fuller periodic lattice than the min-image generation potential did.
    if influence.size > 0 and influence.max() > 0.5*np.min(Box):
        n_over = int(np.sum(influence > 0.5*np.min(Box)))
        warnings.warn(f"{n_over} pore(s) reach {influence.max():.2f} A > half the smallest box edge "
                      f"({0.5*np.min(Box):.2f} A): the emitted images sum a fuller periodic lattice "
                      f"than the minimum-image convention used at generation.")

    shifts = np.array(list(product(*[(-1, 0, 1) if p else (0,) for p in periodic])), dtype = float)

    coord_img, R_img, parent = [], [], []
    for i, centre in enumerate(coord):
        for shift in shifts:
            centre_img = centre + shift*Box
            # nearest point of the box to the shifted center: the image is kept only if its
            # influence sphere still overlaps the box (exact sphere-AABB test).
            closest = np.clip(centre_img, 0.0, Box)
            if np.linalg.norm(centre_img - closest) <= influence[i]:
                coord_img.append(centre_img)
                R_img.append(radii[i])
                parent.append(i)

    return (np.array(coord_img, dtype = float).reshape(-1, 3),
            np.array(R_img, dtype = float),
            np.array(parent, dtype = int))

def place_ring(template_coords: NDArray,
               coord_idx: NDArray,
               coord_second_anchor: NDArray,
               coord_idx3: NDArray,
               Box: NDArray,
               periodic: NDArray[np.int32] = np.ones((1,3), dtype=int),
               bond_dist: float = 1.42,
               anchor_idx: list[int] = [0, 1]
               ) -> NDArray:
    """
    Place a ring template on an existing bond.

    The template edge anchor_idx[0] -> anchor_idx[1] is rotated onto the bond
    coord_idx -> coord_second_anchor (minimum image) and translated so that anchor_idx[0]
    sits on coord_idx. If one of the two ring atoms next to that edge falls within bond_dist
    of coord_idx3, the ring is flipped by 180 degrees about the bond.

    Parameters
    ----------
    template_coords : (n, 3) ndarray
        Ring template, e.g. from carbon_rings_template.
    coord_idx, coord_second_anchor : (3,) ndarray
        The two atoms the new ring shares with the structure.
    coord_idx3 : (3,) ndarray
        An atom bonded to coord_second_anchor, which the new ring must point away from.
    Box : (3,) ndarray
        Box edges, Angstrom.
    periodic : (3,) array of int, optional
        Periodic flags per axis.
    bond_dist : float, optional
        Bond length, Angstrom.
    anchor_idx : list of int, optional
        Template indices of the two shared atoms.

    Returns
    -------
    (n, 3) ndarray
        The placed ring, not yet wrapped into the box.
    """
    target_vec = fold(coord_idx, coord_second_anchor, Box, periodic) - coord_idx
    current_vec = template_coords[anchor_idx[1]] - template_coords[anchor_idx[0]]

    # Normalize
    target_vec /= np.linalg.norm(target_vec)
    current_vec /= np.linalg.norm(current_vec)

    # rotate the template edge onto the bond, then translate it onto coord_idx
    aligned_ring = ring_rotation(template_coords, current_vec, target_vec)

    translation = coord_idx - aligned_ring[anchor_idx[0]]
    aligned_ring += translation

    # the two ring atoms next to the shared edge must not sit on top of coord_idx3
    if np.any(distance(coord_idx3, aligned_ring[[2, len(aligned_ring)-1],:], Box, periodic) <= bond_dist):
        # flip by 180 degrees about the shared bond
        aligned_ring = axis_rotation(aligned_ring, target_vec, np.pi, anchor_idx[0])

    return aligned_ring

def build_voxel_centers(Box: NDArray,
                        dx: float,
                        with_index: bool = False
                        ) -> tuple[NDArray, NDArray | None, tuple[int, int, int]]:
    """
    Centers of a regular grid of cubic voxels over the box.

    Parameters
    ----------
    Box : (3,) ndarray
        Box edges, Angstrom.
    dx : float
        Voxel edge, Angstrom. Each axis holds int(edge / dx) voxels, so a remainder thinner
        than dx is not sampled.
    with_index : bool, optional
        Also return the integer (i, j, k) index of every voxel.

    Returns
    -------
    centers : (Nvox, 3) ndarray
        Voxel centers in C order (k fastest), Angstrom.
    voxel_index : (Nvox, 3) ndarray of int, or None
        Voxel indices, when with_index is True.
    shape : tuple of int
        Grid shape (nx, ny, nz).
    """
    nx = int(Box[0] / dx)
    ny = int(Box[1] / dx)
    nz = int(Box[2] / dx)

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    voxel_centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    voxel_index = None
    if with_index:
        I, J, K = np.meshgrid(np.arange(nx),
                                np.arange(ny),
                                np.arange(nz),
                                indexing='ij')
        voxel_index = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)

    voxel_shape = (nx, ny, nz)

    return voxel_centers, voxel_index, voxel_shape

def build_adjacency(vector: NDArray) -> dict:
    """
    Bond graph from a list of bonds.

    Parameters
    ----------
    vector : (M, 2) array of int
        Bonded atom pairs.

    Returns
    -------
    dict
        {atom: set of bonded atoms}, symmetric.
    """
    adjacency = {}
    for a, b in vector:
        if a not in adjacency:
            adjacency[a] = set()
        if b not in adjacency:
            adjacency[b] = set()
        adjacency[a].add(int(b))
        adjacency[b].add(int(a))
    return adjacency

def get_n_level_neighbors(adjacency,
                          idx: int,
                          n_level: int = 2) -> NDArray[np.int64]:
    """
    Atoms within a given number of bonds of one atom.

    Parameters
    ----------
    adjacency : dict
        Bond graph {atom: set of bonded atoms}.
    idx : int
        Reference atom, excluded from the result.
    n_level : int, optional
        Maximum number of bonds: 1 gives the first neighbors, 2 adds the second, and so on.

    Returns
    -------
    ndarray of int
        Sorted atom indices; empty if there are none.
    """
    current_layer = set([idx])
    all_neighbors = set()

    for _ in range(n_level):
        next_layer = set()
        for atom in current_layer:
            next_layer |= adjacency.get(atom, set())

        # Exclude already visited atoms and the origin
        next_layer -= all_neighbors
        next_layer.discard(idx)

        if not next_layer:
            break

        all_neighbors |= next_layer
        current_layer = next_layer

    return np.array(sorted(all_neighbors), dtype=int)

def _reconstruct_ring_path(ring_set: set,
                           adjacency: dict) -> list | None :
    """
    Atoms of a ring in cyclic order, walked through the bond graph.

    Parameters
    ----------
    ring_set : set of int
        Atom indices of the ring.
    adjacency : dict
        Bond graph {atom: set of bonded atoms}.

    Returns
    -------
    list of int or None
        The atoms in cyclic order; None if the set is not a simple ring (some atom does not
        have exactly two bonded neighbors inside it: a branched set or an open path).
    """
    ring = list(ring_set)
    intra = {a: adjacency.get(a, set()) & ring_set for a in ring}

    if any(len(v) != 2 for v in intra.values()):
        return None   # not a simple ring topology

    path = [ring[0]]
    prev = None
    for _ in range(len(ring) - 1):
        curr = path[-1]
        candidates = intra[curr] - ({prev} if prev is not None else set())
        if not candidates:
            return None
        nxt = next(iter(candidates))
        path.append(nxt)
        prev = curr
    return path

def ring_has_shortcut(ring_tuple: tuple,
                      adjacency: dict) -> bool:
    """
    True if the ring is not primitive and must be rejected.

    A ring is primitive (Franzblau, Phys. Rev. B 44, 4925, 1991) when no two of its atoms are joined through the
    bond graph by a path strictly shorter than the shorter arc between them along the ring;
    an equal-length path is allowed.

    Parameters
    ----------
    ring_tuple : tuple of int
        Atom indices of the ring, in any order.
    adjacency : dict
        Bond graph {atom: set of bonded atoms}.

    Returns
    -------
    bool
        True for a non-primitive ring, and for a set that is not a simple ring.
    """
    ring_set = set(ring_tuple)
    path = _reconstruct_ring_path(ring_set, adjacency)

    if path is None:
        return True   # cannot reconstruct simple ring -> reject

    n = len(path)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue   # atoms are adjacent in the cycle (wrap-around)

            ring_dist = min(j - i, n - (j - i))   # shorter arc
            # Check if path[j] is reachable from path[i] in fewer hops
            reachable = get_n_level_neighbors(adjacency, path[i], ring_dist - 1)
            if path[j] in reachable:
                return True   # shortcut found

    return False
