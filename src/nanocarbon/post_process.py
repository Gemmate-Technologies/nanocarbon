# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Post-processing of carbon structures, constructed or imported from MD.

Ring statistics, radial distribution function, coordination, pore-size distribution,
porosity, pore-network connectivity, accessible surface and volumes. Void measures (PSD,
porosity, connectivity) and pair counts are periodic on all three axes, whatever s.periodic
says; accessibility (compute_AVA) is not, since the box faces are the interface to the
outside. A point is void when its nearest-atom distance is strictly greater than the
threshold, in every function.
"""

import numpy as np
from numpy.typing import NDArray
from itertools import product
from dataclasses import dataclass

from .structures import CarbonStructure
from . import geometry_functions as gf

from scipy.spatial import Voronoi, KDTree
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import marching_cubes, mesh_surface_area


def count_rings(s: CarbonStructure, 
                max_ring_size: int = 12
                ) -> tuple[dict, list]:
    """
    Primitive rings of the bond graph, up to max_ring_size atoms.

    From every atom, all simple cycles through it up to max_ring_size atoms are enumerated by a
    depth-limited path search; a cycle is kept if it is primitive, with no shortcut path
    between two of its atoms (geometry_functions.ring_has_shortcut; Franzblau, Phys. Rev. B 44,
    4925, 1991). The cost grows quickly with max_ring_size.

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze; its stored bonds are used.
    max_ring_size : int, optional
        Largest ring size searched. Default 12.

    Returns
    -------
    bin_count : dict
        {ring size: number of rings}.
    rings : list of tuple of int
        Each ring as its sorted atom indices.
    """

    rings = set()
    Np = s.atoms_number

    for ii in range(Np):
        neighbors = list(s.adjacency.get(ii, []))
        if not neighbors:
            continue

        # initialize branches: [[ii, neighbor], ...]
        branches = [[ii, nb] for nb in neighbors]
        rings_iter = set()

        while branches:
            new_branches = []
            for path in branches:
                last = path[-1]
                prev = path[-2]
                candidates = s.adjacency.get(last, set())

                # exclude already visited and immediate predecessor
                candidates = candidates - set(path[1:]) - {prev}
                if ii in candidates: # closure
                    rings_iter.add(tuple(sorted(int(a) for a in path)))
                    candidates = candidates - {ii}

                if len(path) + 1 <= max_ring_size: # growth
                    for nb in candidates:
                        new_branches.append(path + [nb])
                # dead-end -> candidates empty -> nothing appended -> branch vanishes
            branches = new_branches

        # A ring holding another ring's atoms has a chord, which is a shortcut: the primitive
        # test alone removes it, so no separate subset filter is needed.
        for r in rings_iter:
            if not gf.ring_has_shortcut(r, s.adjacency):
                rings.add(r)

    # histogram
    bin_count = {}
    for ring in rings:
        n = len(ring)
        bin_count[n] = bin_count.get(n, 0) + 1

    return bin_count, list(rings)

def radial_distribution_function(s: CarbonStructure,
                                 size_bin: float,
                                 central_ids: NDArray | None = None,
                                 r_max: float = 10.0,
                                 ) -> tuple[NDArray, NDArray, NDArray]:

    """
    Radial distribution function g(r), from exact periodic pair counts.

    Pairs are counted with a periodic KD-tree up to r_max, capped at half the shortest box edge
    (the minimum-image limit). g(r) = counts / (rho * n_central * shell volume), with rho the
    number density of the whole structure.

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.
    size_bin : float
        Bin width, Angstrom.
    central_ids : ndarray of int, optional
        Atoms used as centers. Their neighbors still come from the whole structure and rho stays
        global, so a sparse region gives g(r) below 1 at large r. None (default) uses every atom:
        the standard RDF.
    r_max : float, optional
        Largest distance, Angstrom. Default 10.

    Returns
    -------
    bin_centres : (n_bins,) ndarray
        Bin midpoints, Angstrom.
    counts : (n_bins,) ndarray
        Number of (central atom, neighbor) pairs per bin.
    rdf : (n_bins,) ndarray
        g(r).
    """

    box = s.Box
    stop = min(r_max, float(np.min(box))/2)
    n_bins = int(stop/size_bin)
    bins   = size_bin * np.arange(n_bins + 1)

    num_atoms = s.atoms_number
    rho = num_atoms/(box[0]*box[1]*box[2])           # global reference density
    central = np.arange(num_atoms) if central_ids is None else np.asarray(central_ids)
    n_central = len(central)

    # count_neighbors returns, for each bin edge r, how many (central, any) pairs are closer
    # than r: cumulative counts, so their differences are the counts per bin. Each central atom
    # also pairs with itself at d = 0, hence the subtraction.
    coords = np.mod(s.coordinates, box)         # boxsize= requires coordinates inside [0, L)
    tree = KDTree(coords, boxsize=box)
    central_tree = KDTree(coords[central], boxsize=box)
    cumulative = central_tree.count_neighbors(tree, bins[1:]) - n_central
    counts = np.diff(cumulative, prepend=0).astype(float)

    vol = 4/3 * np.pi * ( bins[1:]**3 - bins[:-1]**3)
    rdf = counts /rho/n_central/vol    # divide by #central, not #all
    bin_centres = (bins[:-1] + bins[1:]) / 2   # midpoint of each bin

    return bin_centres, counts, rdf

def _local_rdf_stack(s: CarbonStructure, size_bin: float, n_samples: int,
                     sphere_radius: float, seed: int | None = None) -> NDArray | None:
    """
    Local RDFs: one g(r) per random sphere, with the atoms inside it as centers.

    Neighbors come from the whole structure and the density stays global, so an under-dense
    pocket plateaus below 1: intended, the local RDF is there to expose such pockets. Spheres
    holding no atom are skipped.

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.
    size_bin : float
        Bin width, Angstrom.
    n_samples : int
        Number of spheres.
    sphere_radius : float
        Sphere radius, Angstrom.
    seed : int, optional
        Seed of the sphere centers: the same structure and seed give the same spheres.

    Returns
    -------
    (M, n_bins, 2) ndarray or None
        [bin center, g(r)] for each non-empty sphere (M <= n_samples); None if all were empty.
    """
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_samples):
        centre = rng.uniform(0.0, 1.0, size=3) * s.Box          # random point in the box
        d = gf.distance(centre, s.coordinates, s.Box, s.periodic)
        central_ids = np.where(d < sphere_radius)[0]
        if central_ids.size == 0:
            continue
        bins, _, rdf = radial_distribution_function(s, size_bin, central_ids=central_ids)
        curves.append(np.column_stack([bins, rdf]))
    return np.stack(curves) if curves else None

def count_coordination_number(s: CarbonStructure) -> dict:
    """
    Percentage of atoms by number of bonds, from the stored bond graph.

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.

    Returns
    -------
    dict
        {'0 bonds': %, ..., '4 bonds': %, '>4 bonds': %}, summing to 100. Over-coordinated
        atoms can occur in constructed structures, a defect the anneal cures.
    """
    adj = s.adjacency
    N = s.atoms_number

    coordination_number = {}

    adj_atoms = np.array([len(adj.get(i, [])) for i in range(N)])

    for i in range(5):
        number_of_atoms = len(np.where(adj_atoms == i)[0])
        key = f'{i} bonds'
        coordination_number[key] = number_of_atoms*100/N

    # Carbon is at most 4-coordinated, but pre-anneal constructed structures can
    # contain over-coordinated atoms (a local defect the MD anneal cures). Report
    # them in an overflow bucket so the percentages always sum to 100.
    coordination_number['>4 bonds'] = len(np.where(adj_atoms > 4)[0])*100/N

    return coordination_number

# PSD

def _duplicate_box(s: CarbonStructure,
                   buffer_width: float | None = None
                   ) -> NDArray:
    """
    Atom coordinates tiled over the 26 periodic images, optionally cut to a buffer shell.

    Voronoi needs atoms beyond the box to get the cells near the faces right. With buffer_width
    w only images inside [-w, Box + w) on every axis are kept: sized to the largest empty-sphere
    radius (compute_connectivity), the in-box tessellation is unchanged while the input shrinks
    from 27 N to about (1 + 2w/L)^3 N atoms.

    Parameters
    ----------
    s : CarbonStructure
        Source structure.
    buffer_width : float, optional
        Buffer width, Angstrom. None (default) keeps all 27 copies.

    Returns
    -------
    (M, 3) ndarray
        Coordinates of the atoms and of the images kept, Angstrom.
    """
    box = s.Box
    coords = s.coordinates

    # Build the full (27, N, 3) tile first; pruning is a single boolean mask.
    offsets = np.array(list(product([-1, 0, 1], repeat=3)))   # (27, 3)
    shifts = offsets * box                                    # (27, 3)
    images = coords[None, :, :] + shifts[:, None, :]
    images = images.reshape(-1, 3)

    if buffer_width is None:
        return images

    # Keep atoms within `buffer_width` of every face of the primary cell.
    in_buffer = np.all(
        (images >= -buffer_width) & (images < box + buffer_width),
        axis=1,
    )
    return images[in_buffer]

def _compute_voxel_distances(coords: NDArray,
                             box: NDArray,
                             voxels: NDArray,
                             voxels_index: NDArray,
                             r_atom: float,
                             workers: int = 1
                             ) -> tuple[NDArray, NDArray, NDArray]:
    """
    PSD helper: for each voxel center, the radius of the largest sphere centered there that
    overlaps no atom (nearest-atom distance minus r_atom; periodic KD-tree).

    Parameters
    ----------
    coords : (N, 3) ndarray
        Atom coordinates, Angstrom.
    box : (3,) ndarray
        Box edges, Angstrom (KD-tree boxsize, so no image tiling is needed).
    voxels : (Nvox, 3) ndarray
        Voxel centers, Angstrom.
    voxels_index : (Nvox, 3) ndarray of int
        (i, j, k) grid index of each voxel.
    r_atom : float
        Atom radius, Angstrom.
    workers : int, optional
        KD-tree query threads. 1 (default) is deterministic; -1 is faster but can flip
        voxels within round-off of a threshold.

    Returns
    -------
    min_distances : (Nvox,) ndarray
        Sphere radius at each voxel, Angstrom (negative inside an atom).
    voxels, voxels_index
        The inputs, returned unchanged.
    """

    tree = KDTree(coords, boxsize=box)
    min_distances, _ = tree.query(voxels, k=1, workers=workers)
    min_distances = min_distances - r_atom

    return min_distances, voxels, voxels_index

def _compute_accessible_voxels(min_distances: NDArray, 
                               voxels: NDArray, 
                               voxels_index: NDArray, 
                               r_atom: float
                               ) -> tuple[NDArray, NDArray, NDArray, int]:
    """
    PSD helper: the voxels where a probe fits, and the voxels outside every atom.

    Parameters
    ----------
    min_distances : (Nvox,) ndarray
        Sphere radius at each voxel (_compute_voxel_distances), Angstrom.
    voxels : (Nvox, 3) ndarray
        Voxel centers, Angstrom.
    voxels_index : (Nvox, 3) ndarray of int
        (i, j, k) grid index of each voxel.
    r_atom : float
        Probe radius, Angstrom (the caller passes r_N).

    Returns
    -------
    probe_voxels : (K, 3) ndarray
        Centers of the voxels with min_distances > r_atom.
    probe_radii : (K,) ndarray
        Their sphere radii.
    point_index : (Q, 3) ndarray of int
        Grid indices of the voxels with min_distances > 0 (outside every atom).
    n_point : int
        Q.
    """
    free_probe = min_distances > r_atom
    free_point = min_distances > 0.0

    return (voxels[free_probe],
            min_distances[free_probe],
            voxels_index[free_point],
            int(free_point.sum()))

def _compute_cumulative_psd(N_iter: int, 
                            acc_voxels_N : NDArray,
                            min_distances_acc_N: NDArray,
                            acc_voxels_index_point : NDArray,
                            N_voxels_acc_point : int,  
                            voxels_len : float,  
                            vol: NDArray,  
                            bin_size: float, 
                            psd_cum: NDArray,
                            seed: int | None = None,
                            ) -> tuple[NDArray, list, list]:
    """
    PSD helper: Monte-Carlo accumulation of the cumulative pore-size distribution.

    Each iteration samples a voxel outside every atom and finds the largest sphere that
    contains its center among the spheres centered on probe-accessible voxels; that sphere's
    diameter is added to every cumulative bin up to it. A point no such sphere covers is
    skipped.

    Parameters
    ----------
    N_iter : int
        Number of samples.
    acc_voxels_N : (K, 3) ndarray
        Centers of the probe-accessible voxels, Angstrom.
    min_distances_acc_N : (K,) ndarray
        Their sphere radii, Angstrom.
    acc_voxels_index_point : (Q, 3) ndarray of int
        Grid indices of the voxels outside every atom (the sampling pool).
    N_voxels_acc_point : int
        Q.
    voxels_len : float
        Voxel edge, Angstrom.
    vol : (3,) ndarray
        Box edges, Angstrom (for the minimum image).
    bin_size : float
        Diameter bin width, Angstrom.
    psd_cum : (n_bins,) ndarray
        Zeroed accumulator, filled in place.
    seed : int, optional
        Sampler seed; None draws a fresh one.

    Returns
    -------
    psd_cum : (n_bins,) ndarray
        Cumulative counts per diameter bin.
    R_count : list of float
        Radius of the sphere found for each counted sample, Angstrom.
    coord_count : list of (3,) ndarray
        Center of that sphere.
    """
    R_count = []
    coord_count = []
    rng = np.random.default_rng(seed= seed)

    for i in range(N_iter):
        idx = rng.integers(0, N_voxels_acc_point)
        i_cube, j_cube, k_cube = acc_voxels_index_point[idx]

        coord = np.array([voxels_len*i_cube + voxels_len/2, 
                          voxels_len*j_cube + voxels_len/2, 
                          voxels_len*k_cube + voxels_len/2])

        delta = acc_voxels_N - coord
        delta -= vol*np.round(delta/vol)
        # distance from the point to every probe-accessible voxel center (minimum image)
        dists = np.linalg.norm(delta, axis = 1) 
        mask = dists <= min_distances_acc_N       

        if not np.any(mask):
            continue # no probe-accessible sphere covers this point

        # Pick the largest probe-fit radius among accessible voxels. Masked max,
        # not argmax on a sorted array: the result must stay independent of the
        # order of `min_distances_acc_N`.
        masked_distances = np.where(mask, min_distances_acc_N, -np.inf)
        first_valid_idx = int(np.argmax(masked_distances))
        r_max = min_distances_acc_N[first_valid_idx]

        # record the sample
        R_count.append(r_max)
        coord_count.append(acc_voxels_N[first_valid_idx])

        # Final binning
        bin_val = int(2*r_max/bin_size)
        psd_cum[:bin_val+1]+=1

    return psd_cum, R_count, coord_count

def compute_psd(s: CarbonStructure, 
                N_iter: int = 5000,
                bin_size: float = 0.25,
                voxel_len: float = 0.2,
                r_C: float = 1.71,
                r_N: float = 1.55,
                seed: int | None = None,
                workers: int = 1,
                ) -> tuple[NDArray, NDArray, NDArray, list, list]:
    """
    Geometric pore-size distribution, by Monte-Carlo sampling on a periodic voxel grid.

    A sample is a random voxel center outside every atom (nearest-atom distance > r_C). Its
    pore diameter is that of the largest sphere containing it that overlaps no atom and is
    centered where a probe of radius r_N fits (the geometric PSD definition of PoreBlazer, which
    the validation matched with r_C = 1.7155, r_N = 1.657). F(d) is the fraction of samples in
    pores wider than d, and the PSD is -dF/dd. Memory peaks near 95 bytes per voxel (voxel
    centers, indices and query results), about 12 GB for a 100 A box at voxel_len 0.2; raise
    voxel_len for larger boxes (0.5 A still resolves a 4 A pore).

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.
    N_iter : int, optional
        Number of samples. Default 5000.
    bin_size : float, optional
        Diameter bin width, Angstrom. Default 0.25.
    voxel_len : float, optional
        Voxel edge, Angstrom. Default 0.2.
    r_C : float, optional
        Carbon radius, Angstrom. Default 1.71.
    r_N : float, optional
        Probe (nitrogen) radius, Angstrom. Default 1.55.
    seed : int, optional
        Sampler seed; None draws a fresh one.
    workers : int, optional
        KD-tree query threads. 1 (default) is deterministic; -1 is faster but not
        bit-reproducible.

    Returns
    -------
    psd : (M, 2) ndarray
        [diameter (Angstrom), -dF/dd (1/Angstrom)].
    psd_cum : (M, 2) ndarray
        [diameter, F], with F(0) = 1.
    psd_cum_raw : (M, 2) ndarray
        [diameter, cumulative sample counts].
    R_count : list of float
        Pore radius found for each counted sample, Angstrom.
    coord_count : list of (3,) ndarray
        Center of the sphere found for each counted sample.
        With no void at all the three arrays are [[0, 0]] and the two lists are empty.
    """
    R_count = []
    coord_count = []
    vol = s.Box

    voxels, voxels_index, _ = gf.build_voxel_centers(vol, voxel_len, with_index= True)
    if voxels_index is None:  # with_index=True always returns it; the check narrows the type
        raise RuntimeError("build_voxel_centers returned no voxel index.")

    # Compute, per voxel, the largest probe radius that can sit at the voxel
    # without overlapping any atom (= nearest-atom distance - r_C). PBC is
    # handled natively inside _compute_voxel_distances via KDTree boxsize.
    min_distances, voxels, voxels_index = _compute_voxel_distances(coords = s.coordinates,
                                                                   box = vol,
                                                                   voxels = voxels,
                                                                   voxels_index = voxels_index,
                                                                   r_atom = r_C,
                                                                   workers = workers)

    # voxels where the probe fits, and voxels outside every atom
    acc_voxels_N, min_distances_acc_N, acc_voxels_index_point, N_voxels_acc_point = _compute_accessible_voxels(min_distances,
                                                                                                            voxels,
                                                                                                            voxels_index,
                                                                                                            r_atom = r_N)
    del voxels, voxels_index

    if N_voxels_acc_point == 0:
        sentinel = np.array([[0.0, 0.0]])
        return sentinel, sentinel, sentinel, [], []

    maxpore = 2.0 * np.ceil(np.max(min_distances)) # largest pore expected
    nbins = int(maxpore/bin_size) + 1 
    psd_cum = np.zeros(nbins) 
    psd_cum, R_count, coord_count = _compute_cumulative_psd(N_iter = N_iter, 
                                                            acc_voxels_N = acc_voxels_N, 
                                                            min_distances_acc_N = min_distances_acc_N,
                                                            acc_voxels_index_point = acc_voxels_index_point,
                                                            N_voxels_acc_point = N_voxels_acc_point,
                                                            voxels_len = voxel_len,
                                                            vol = vol,
                                                            bin_size = bin_size,
                                                            psd_cum = psd_cum,
                                                            seed = seed
                                                        )
    if psd_cum[0] == 0 : 
        sentinel = np.array([[0.0, 0.0]])
        return sentinel, sentinel, sentinel, [], []

    psd_cum_original = psd_cum.copy()    
    psd_cum = psd_cum / psd_cum[0]
    diameters = np.linspace(0, len(psd_cum)-1,
                            num = len(psd_cum)) * bin_size # bin index -> diameter

    psd_cum_original = np.column_stack((diameters,psd_cum_original)) 
    psd_cum = np.column_stack((diameters, psd_cum)) 


    # Differential PSD = -dF/dd, where F is the cumulative PSD over diameter d.
    # np.gradient uses central differences in the interior and one-sided
    # (forward / backward) differences at the endpoints.
    psd = -np.gradient(psd_cum[:, 1], bin_size)
    psd = np.column_stack((diameters, psd))

    return psd, psd_cum, psd_cum_original, R_count, coord_count

# POROSITY

def compute_porosity(s: CarbonStructure, 
                     N_iter: int = 100000,
                     r_atom: float = 1.71,
                     r_probe:float = 1.55,
                     voxel_len: float = 0.2,
                     seed: int  | None = None,
                    ) -> float:
    """
    Void fraction seen by a probe center, by Monte-Carlo sampling.

    A sampled voxel center counts as void when it lies farther than r_atom + r_probe from
    every atom, periodic images included. With r_probe = 0 this is the geometric porosity;
    with the default nitrogen probe it is the probe-center fraction, which is smaller.

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.
    N_iter : int, optional
        Number of sampled voxel centers. Default 100000.
    r_atom : float, optional
        Atom radius, Angstrom. Default 1.71 (carbon).
    r_probe : float, optional
        Probe radius, Angstrom. Default 1.55 (nitrogen); 0 gives the geometric porosity.
    voxel_len : float, optional
        Edge of the sampling-grid voxel, Angstrom. Default 0.2.
    seed : int, optional
        Sampler seed; None draws a fresh one.

    Returns
    -------
    float
        Fraction of the sampled points that are void, in [0, 1].
    """
    coords = s.coordinates
    vol = s.Box
    tree = KDTree(coords, boxsize=vol)

    nx, ny, nz =  [int(L / voxel_len) for L in vol] # int(): same truncation as build_voxel_centers
    n_vox = nx*ny*nz
    shape = (nx, ny, nz)

    rng = np.random.default_rng(seed = seed)
    sample_idx = rng.integers(0, n_vox, size = N_iter)
    ijk = np.stack(np.unravel_index(sample_idx, shape), axis=1)
    sample_coords = (ijk + 0.5) * voxel_len

    dists, _ = tree.query(sample_coords)
    n_free = int(np.sum(dists > (r_atom+r_probe))) # >: strictness coherence with PSD.

    return n_free / N_iter

# CONNECTIVITY VIA VORONOI

def _estimate_max_pore_radius(coords:NDArray, 
                              box:NDArray, 
                              probe_spacing: float = 1.5
                              )-> float:

    """
    Largest nearest-atom distance over a grid of points: an estimate of the radius of the
    largest empty sphere, low by at most about probe_spacing.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Atom coordinates, Angstrom.
    box : (3,) ndarray
        Box edges, Angstrom.
    probe_spacing : float, optional
        Grid spacing, Angstrom. Default 1.5.

    Returns
    -------
    float
        The largest distance, Angstrom.
    """
    grid = [np.arange(0, L, probe_spacing) for L in box]
    X, Y, Z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    grid_points = np.stack([X.ravel(),
                            Y.ravel(),
                            Z.ravel()], axis=1)  
    tree = KDTree(coords, boxsize=box)
    dists, _ = tree.query(grid_points)        # (M,) nearest-atom distance per probe
    return float(dists.max())

def _check_bottleneck(structure_tree: KDTree,
                      pore_coords: NDArray,
                      edges_list: NDArray,
                      r_struct: float,
                      r_probe: float,
                      n_check: int
                      ) -> NDArray:
    """
    Connectivity helper: keep the edges along which a probe fits.

    An edge v1 -> v2 is kept when n_check evenly spaced interior points of the segment all lie
    farther than r_struct + r_probe from every atom (the endpoints are checked by the caller).

    Parameters
    ----------
    structure_tree : KDTree
        Tree over the padded atom coordinates (images included), so distances are periodic.
    pore_coords : (V, 3) ndarray
        Voronoi vertices, Angstrom.
    edges_list : (E, 2) ndarray of int
        Candidate edges, as vertex indices.
    r_struct : float
        Atom radius, Angstrom.
    r_probe : float
        Probe radius, Angstrom.
    n_check : int
        Interior points tested per edge.

    Returns
    -------
    (E', 2) ndarray of int
        The edges that pass.
    """

    test_distance = np.linspace(0, 1, n_check+2)[1:-1]
    threshold = r_struct + r_probe
    idx_admissible = np.zeros_like(edges_list[:,0],dtype=bool)

    for i, (v1_idx, v2_idx) in enumerate(edges_list):

        v1 = pore_coords[v1_idx]
        v2 = pore_coords[v2_idx]

        delta_d = np.outer(test_distance, v2 - v1)
        test_points = v1 + delta_d

        min_dist,_ = structure_tree.query(test_points)
        idx_admissible[i] = not np.any(min_dist <= threshold)

    edges_list = edges_list[idx_admissible]

    return edges_list

def compute_connectivity(s: CarbonStructure, 
                         n_check: int = 10,
                         r_C: float = 1.71,
                         r_N: float = 1.55,
                         probe_spacing: float = 1.5,
                         ) -> float :
    """
    Mean pore connectivity k_avg: links per pore of the Voronoi pore network.

    The Voronoi vertices of the atoms are the candidate pore centers and the Voronoi edges the
    candidate links, computed on a periodic copy padded by the largest pore radius plus
    probe_spacing. A pore is a vertex in the box farther than r_C + r_N from every atom; a link
    is an edge whose two ends are such points and along which the probe fits at n_check
    interior points. A link crossing a face is counted once, through the copy whose midpoint
    lies in the box. k_avg = links / pores, half the mean degree of the network. No external
    tool computes it: use it to compare structures.

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.
    n_check : int, optional
        Interior points tested along each link. Default 10.
    r_C : float, optional
        Carbon radius, Angstrom. Default 1.71.
    r_N : float, optional
        Probe (nitrogen) radius, Angstrom. Default 1.55.
    probe_spacing : float, optional
        Grid spacing of the largest-pore estimate that sizes the padding, Angstrom.
        Default 1.5.

    Returns
    -------
    float
        k_avg; 0 if there is no pore.
    """

    box = s.Box
    # Buffer width for the periodic-image tile fed to Voronoi. The primary-
    # cell vertices and ridges depend only on atoms whose distance to the
    # cell is at most the largest empty-sphere radius in the structure.
    R_max  = _estimate_max_pore_radius(s.coordinates, box, probe_spacing)
    buffer = R_max + probe_spacing

    if buffer >= float(np.min(box)):
        coords = _duplicate_box(s) 
    else:
        coords = _duplicate_box(s, buffer_width=buffer)

    vor = Voronoi(coords)
    # voronoi vertices in extended box represent pores centers
    pore_coords = vor.vertices
    # faces edges are possible connections
    faces_list = vor.ridge_vertices

    # Build edge list from Voronoi ridges. Each ridge is a polygon face
    # (boundary between two Voronoi cells); its consecutive vertices form
    # 1-D Voronoi edges, which are the geometric segments along which pores
    # connect.  scipy marks unbounded faces with -1 in the vertex list —
    # those ridges extend to infinity and have no usable geometry, so they
    # are dropped (otherwise -1 would silently index pore_coords[-1]).
    edges = []
    for face in faces_list:
        if -1 in face:
            continue
        for j in range(len(face)-1):
            edges.append( [face[j],face[j+1]] )
        edges.append([face[-1],face[0]])

    edges = np.array(edges)
    edges = np.unique(np.sort(edges, axis=1), axis=0) # sort maps into [min, max], unique drops repeats from multiple polygons sharing an edge

    # A link crossing a periodic face appears twice in the tiled tessellation, once from each
    # side (v -> w' and v' -> w). Exactly one copy of every link has its MIDPOINT in the
    # primary box, so keeping those counts each link once.
    mask_vertices = np.all((pore_coords >= 0) & (pore_coords < box), axis=1)
    midpoints = 0.5*(pore_coords[edges[:, 0]] + pore_coords[edges[:, 1]])

    edges_filtered = edges[np.all((midpoints >= 0) & (midpoints < box), axis=1)]

    # Admissibility: both vertices must be outside atom exclusion zones. Vectorized.
    # The KDTree is built over the same padded `coords` used by Voronoi: this lets
    # us query distances from out-of-cell pore vertices directly without first
    # wrapping them into the primary box.
    structure_tree = KDTree(coords)

    vert1_dist, _ = structure_tree.query(pore_coords[edges_filtered[:, 0]])
    vert2_dist, _ = structure_tree.query(pore_coords[edges_filtered[:, 1]])

    admissible = (vert1_dist > r_C + r_N) & (vert2_dist > r_C + r_N)

    edges_filtered_s2 = edges_filtered[admissible]

    # Check if candidate link are probe-accessible via bottleneck-check
    edges_filtered_s3 = _check_bottleneck(structure_tree, 
                                        pore_coords, 
                                        edges_filtered_s2, 
                                        r_C, 
                                        r_N,
                                        n_check)

    # Pore-count denominator: count vertices appearing as either endpoint of an
    # edge — collecting only one column drops pores that always sit in the other
    # and inflates k_avg = N_links / N_pores. Restrict to primary-box pores after.
    # All edges, not the midpoint-filtered ones: a pore whose links all cross a face
    # keeps those links only through its image, yet it is still one pore.
    vert_list = np.unique(edges.ravel())
    vert_list_reduced = vert_list[mask_vertices[vert_list]]

    vert_dist, _ = structure_tree.query(pore_coords[vert_list_reduced])
    n_pores = int(np.sum(vert_dist > r_C + r_N))

    k_avg = len(edges_filtered_s3)/n_pores if n_pores > 0 else 0.0
    return k_avg

# Accessible surface area and volume
#
# The voxel grid is a regular cubic lattice: every interior voxel has exactly
# six face-adjacent neighbors at indices (i±1, j, k), (i, j±1, k), (i, j, k±1).
# Adjacency is therefore *implicit* - index arithmetic, no data structure needed.
# The free / occupied state is stored as a dense 3-D `bool` ndarray of shape
# (nx, ny, nz). At one byte per voxel this beats any pointer-based representation.
#
# Accessibility uses the "non-PBC vacuum interface" convention: the box face is
# the only "outside" the structure has, so a free region is accessible when it
# touches any of the six box faces; a pore fully enclosed inside the box is not.

def _build_voxel_centers_AVA(Box: NDArray, dx: float) -> tuple[NDArray, tuple[int, int, int]]:
    """
    AVA helper: voxel centers of the box, in Fortran order (x fastest).

    The order matches the reshape(shape, order='F') that turns the distances back into a grid.

    Parameters
    ----------
    Box : (3,) ndarray
        Box edges, Angstrom.
    dx : float
        Voxel edge, Angstrom; int(edge / dx) voxels per axis.

    Returns
    -------
    centers : (Nvox, 3) ndarray
        Voxel centers, Angstrom.
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
    voxel_centers = np.stack([X.ravel(order='F'),
                              Y.ravel(order='F'),
                              Z.ravel(order='F')], axis=1)

    voxel_shape = (nx, ny, nz)

    return voxel_centers, voxel_shape

@dataclass
class AVAResult:
    """
    Output of compute_AVA.

    Attributes
    ----------
    asa_smooth_A2 : float
        Accessible surface area, Angstrom^2.
    av_center_A3 : float
        Volume reachable by the probe center, Angstrom^3.
    av_occupiable_A3 : float
        Volume the probe can occupy (the center volume dilated by the probe), Angstrom^3.
    """
    asa_smooth_A2: float
    av_center_A3: float
    av_occupiable_A3: float

def compute_AVA(s: CarbonStructure, 
                voxel_len: float, 
                r_atom: float, 
                r_probe: float, 
                workers: int = 1,
                ) -> AVAResult:
    """
    Accessible surface area and volumes, non-periodic by design.

    The box faces are the interface to the outside: a free voxel (nearest-atom distance
    > r_atom + r_probe) is accessible when a face-connected path of free voxels joins it to a
    box face, so an enclosed pocket is excluded. From one nearest-atom distance grid:
    - center volume: the accessible voxels;
    - occupiable volume: the accessible set dilated by r_probe (Euclidean distance transform),
      clipped to voxels outside the atoms (distance > r_atom);
    - surface: marching cubes on the distance grid at the isolevel r_atom + r_probe, with the
      enclosed pockets sunk below it first, so only the accessible surface is measured.
    PoreBlazer's radii are r_atom = 1.7155 (C) and r_probe = 1.657 (N2).

    Parameters
    ----------
    s : CarbonStructure
        Structure to analyze.
    voxel_len : float
        Voxel edge, Angstrom.
    r_atom : float
        Atom radius, Angstrom.
    r_probe : float
        Probe radius, Angstrom.
    workers : int, optional
        KD-tree query threads. 1 (default) is deterministic; -1 is faster but not
        bit-reproducible.

    Returns
    -------
    AVAResult
        Surface (Angstrom^2), center and occupiable volumes (Angstrom^3).
    """
    cubes, shape = _build_voxel_centers_AVA(s.Box, voxel_len)
    dist = KDTree(s.coordinates).query(cubes, workers=workers)[0].reshape(shape, order='F')

    free = dist > (r_atom + r_probe)
    # label() numbers the 6-connected components of the free voxels; the accessible ones
    # are those with at least one voxel on a box face (label 0 is the occupied space).
    labels = np.zeros(free.shape, dtype=np.int32)
    label(free, output=labels)
    on_face = np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel(),
                                        labels[:, 0].ravel(), labels[:, -1].ravel(),
                                        labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    acc = np.isin(labels, on_face[on_face > 0])
    av_center = int(acc.sum()) * voxel_len**3

    # occupiable = accessible set dilated by the probe, clipped to non-atom voxels:
    # ~acc (feed the complement so the EDT measures each voxel's distance to the accessible set)
    probe_dist = np.asarray(distance_transform_edt(np.logical_not(acc), sampling=voxel_len))
    occ = (probe_dist <= r_probe) & (dist > r_atom)
    av_occ = int(occ.sum()) * voxel_len**3

    if not acc.any():
        asa = 0.0
    else:
        masked = dist.copy()
        masked[free & ~acc] = r_atom + r_probe - 1e-6      # sink enclosed pockets under the isolevel
        verts, faces, *_ = marching_cubes(masked, level=r_atom + r_probe, spacing=(voxel_len,)*3)
        asa = mesh_surface_area(verts, faces)

    return AVAResult(asa, av_center, av_occ)
