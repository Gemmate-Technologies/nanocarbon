# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
LAMMPS input and output.

Readers for dump trajectories (.lammpstrj) and XYZ files (.xyz), both returning
{frame: data} dicts for CarbonStructure.from_LAMMPS_file; writers for a carbon data file and
for the pores as spherical wall regions. LJ_WALL_MINIMUM and WALL_UNIT_TO_EV are the single
definitions of the wall cutoff and of the wall energy unit used by every deck writer.
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from . import geometry_functions as gf

# Multiple of sigma at which each LAMMPS wall shape is minimized -- the WCA cut, so only the
# repulsive branch survives. 12-6: 2^(1/6). The LAMMPS 9-3 wall is a different function,
# E = eps[(2/15)(s/r)^9 - (s/r)^3], minimized at (2/5)^(1/6) -- 1.31x closer in than the 12-6
# value, so reusing 2^(1/6) for a 9-3 wall leaves its attractive tail alive and the carbon
# drapes onto the pore shells during the anneal. Single definition for every deck writer.
LJ_WALL_MINIMUM = {'126': 2 ** (1/6), '93': (2/5) ** (1/6)}

# The pore-wall depth A is summed with the atoms potential against one E_max, so it carries
# that potential's energy unit: kcal/mol for harmonic, eV for tersoff (real Tersoff C-C
# parameters). Factor to eV, the unit of a `units metal` LAMMPS deck.
WALL_UNIT_TO_EV = {'harmonic': 1/23.0605, 'tersoff': 1.0}

def import_from_LAMMPS (file_path: Path, 
                        atom_types_hash: dict = {'C': 1}) -> dict:
    """
    Read a trajectory with the reader matching its extension.

    Parameters
    ----------
    file_path : Path or str
        A '.lammpstrj' (read_LAMMPS_traj, all frames) or '.xyz' (read_LAMMPS_xyz) file.
    atom_types_hash : dict, optional
        Element symbol -> integer atom type, used for .xyz only. Default {'C': 1}.

    Returns
    -------
    dict
        {frame_index: frame data}, as returned by the reader.

    Raises
    ------
    ValueError
        On any other extension.
    """

    filename = Path(file_path).name
    extention = filename.split(".")[-1]
    if not (extention in ['lammpstrj', 'xyz']):
        raise ValueError("File extension not supported")

    if extention == 'lammpstrj':
        dict_out = read_LAMMPS_traj(file_path, max_frame_index = None )
    else:
        dict_out = read_LAMMPS_xyz(file_path, atom_types_hash = atom_types_hash)

    return dict_out

def read_LAMMPS_xyz (file_path: Path, 
                     atom_types_hash: dict = {'C': 1}) -> dict:
    """
    Read an XYZ trajectory (LAMMPS `dump xyz`, MDAnalysis XYZWriter).

    Each atom line is "<type> <x> <y> <z>". A numeric type (the LAMMPS default) is kept as
    is; an element name (`dump_modify element`, MDAnalysis) is mapped to an integer type
    through atom_types_hash. The timestep is the number after "Timestep:" (LAMMPS) or "frame"
    (MDAnalysis) on the comment line, else the frame index. A line that is not an atom line ends
    the frame, so EVERY element name in the file must be in atom_types_hash. XYZ has no box:
    the bounds are stored as None and from_LAMMPS_file needs box=.

    Parameters
    ----------
    file_path : Path or str
        The .xyz file.
    atom_types_hash : dict, optional
        Element symbol -> integer atom type, for files written with element names. An
        element missing from it ends the frame. Default {'C': 1}.

    Returns
    -------
    dict
        {frame_index: {'timestep': float, 'box_x' / 'box_y' / 'box_z': [None, None],
        'coord': (N, 5) ndarray with columns atom_id, atom_type, x, y, z}}.
    """
    names_box = ["box_x", "box_y", "box_z"]
    xyz_file = {}

    frame_index = 0
    atom_num = 0

    read_frame = True
    read_data = False
    with open(file_path,'r') as infile:
        for line_num, line in enumerate(infile): 
            words = line.split()

            if line_num == 0: continue

            if read_frame:
                xyz_file[frame_index] = {}
                for box in names_box:
                    xyz_file[frame_index][box] = [None, None]
                xyz_file[frame_index]['timestep'] = float(frame_index)
                for key, value in zip(words, words[1:]):
                    if key.lower().rstrip(':') in ('timestep', 'frame') and value.isdigit():
                        xyz_file[frame_index]['timestep'] = float(value)
                xyz_file[frame_index]['coord'] = []
                read_frame = False
                read_data = True
                continue

            if read_data:
                try:
                    atom_num += 1
                    # LAMMPS writes the numeric type unless dump_modify element is set
                    atom_type = int(words[0]) if words[0].isdigit() else atom_types_hash[words[0]]
                    xyz_file[frame_index]['coord'].append([
                        atom_num, atom_type, float(words[1]), float(words[2]), float(words[3])
                              ])
                except (KeyError, ValueError, IndexError):
                    read_frame = True
                    read_data = False
                    atom_num = 0
                    xyz_file[frame_index]['coord'] = np.array(xyz_file[frame_index]['coord'], dtype= float)
                    frame_index += 1

    # finalize the trailing frame (no separator line follows the last frame)
    for frame in xyz_file.values():
        if isinstance(frame['coord'], list):
            frame['coord'] = np.array(frame['coord'], dtype=float)

    return xyz_file

def read_LAMMPS_traj (file_path: Path,
                      max_frame_index: int | None = None
                      ) -> dict:
    """
    Read a LAMMPS dump trajectory (.lammpstrj).

    The ATOMS columns are found from the header, so any dump with at least id, type, x, y, z
    works. Atom ids are remapped to 1-based indices that stay the same across frames, rows are
    sorted by them, and coordinates are wrapped into the box. Orthogonal boxes only.

    Parameters
    ----------
    file_path : Path or str
        The .lammpstrj file.
    max_frame_index : int, optional
        Stop after this many frames; None (default) reads the whole file.

    Returns
    -------
    dict
        {frame_index: {'timestep': float, 'box_x' / 'box_y' / 'box_z': [lo, hi] in Angstrom,
        'coord': (N, 5) ndarray with columns atom_id, atom_type, x, y, z}}.

    Raises
    ------
    ValueError
        On a triclinic box (tilt factors in BOX BOUNDS).
    """
    def read_atoms_line (line, atom_mapping, new_atom_number, dict_keys, box = None):
        """
        Parse one ATOMS line.

        Parameters
        ----------
        line : str
            The line.
        atom_mapping : dict
            LAMMPS id -> remapped id, extended in place with new ids.
        new_atom_number : list of int
            One-element counter of the next remapped id, advanced in place.
        dict_keys : dict
            Column index of 'id', 'type', 'x', 'y', 'z'.
        box : (3, 2) ndarray, optional
            [lo, hi] per axis; the coordinates are wrapped into it when given.

        Returns
        -------
        list or None
            [remapped id, type, x, y, z], or None for a line with fewer than 3 fields (the end of
            the ATOMS block).

        Raises
        ------
        ValueError
            If dict_keys is empty.
        """
        if not (len(dict_keys)>0):
            raise ValueError("Provide a dictionary with keys")

        words = line.split()
        if len(words) >= 3:
            atom_type = int(words[dict_keys.get('type')])

            x_value = float(words[dict_keys.get('x')])
            y_value = float(words[dict_keys.get('y')])
            z_value = float(words[dict_keys.get('z')])

            atom_number = int(words[dict_keys.get('id')])
            if atom_number not in atom_mapping:
                atom_mapping[atom_number] = new_atom_number[0]
                new_atom_number[0] += 1

            if box is not None:
                return [atom_mapping[atom_number], atom_type,
                        box[0,0] + (x_value - box[0,0]) % (box[0,1] - box[0,0]),
                        box[1,0] + (y_value - box[1,0]) % (box[1,1] - box[1,0]),
                        box[2,0] + (z_value - box[2,0]) % (box[2,1] - box[2,0])]
            else:
                return [atom_mapping[atom_number], atom_type, x_value, y_value, z_value]
        else:
            return None 

    key_atom = 'ATOMS'
    key_timestep = 'TIMESTEP'
    key_box = 'BOX'

    atom_mapping = {} # Keep atoms mapping consistent between different frames
    new_atom_number = [1]

    frame_index = 0
    trajectory = {}


    lines_box= 0
    names_box = ["box_x", "box_y", "box_z"]
    go_in_box = False
    go_in_timestep = False
    go_in_atoms = False   

    dict_keys = {}

    with open(file_path,'r') as infile:

        for line in infile: 
            words = line.split()
            # stop after max_frame_index frames
            if max_frame_index is not None:
                if frame_index >= max_frame_index:
                    break

            # the line after ITEM: TIMESTEP opens a new frame
            if go_in_timestep:
                trajectory[frame_index] = {}
                trajectory[frame_index]['timestep'] = float(words[0])
                trajectory[frame_index]['coord'] = []
                go_in_timestep = False
                continue

            if go_in_box:
                trajectory[frame_index][names_box[lines_box]] = [float(words[0]), float(words[1])]
                lines_box += 1

                if lines_box>= 3:
                    lines_box= 0
                    go_in_box = False
                continue

            if go_in_atoms:
                box = np.array([
                    trajectory[frame_index]['box_x'],
                    trajectory[frame_index]['box_y'],
                    trajectory[frame_index]['box_z']
                    ])
                atoms_out = read_atoms_line(line, atom_mapping, new_atom_number, dict_keys, box = box)
                if atoms_out is None:
                    go_in_atoms = False
                    frame_index += 1
                else:
                    trajectory[frame_index]['coord'].append(atoms_out)

            if len(words) >= 2 and words[1] == key_timestep:
                go_in_timestep = True
                continue

            if len(words) >= 2 and words[1] == key_box: # the next three lines are the bounds
                if 'xy' in words:
                    raise ValueError(f"{file_path}: triclinic box (tilt factors in BOX BOUNDS) is not supported.")
                go_in_box = True
                continue

            if len(words) >= 2 and words[1] == key_atom:
                words[words.index(key_atom)] += '_'+words[words.index('id')]
                words.pop(words.index('id'))
                words = words[1:]
                dict_keys = {'id': words.index(key_atom+"_id"),
                             'type': words.index("type"),
                             'x': words.index("x"),
                             'y': words.index("y"),
                             'z': words.index("z")}
                go_in_atoms = True
                continue

    for frame in trajectory.keys():
        coord = np.array(trajectory[frame]['coord'], dtype= float)
        # reorder atoms in rows according to their progressive id.
        trajectory[frame]['coord'] = coord[np.argsort(coord[:, 0])]

    return trajectory

def print_LAMMPS_structure(Box: list | None = None,
                           coord: NDArray | None = None,
                           bonds: NDArray | None = None,
                           write_bonds: bool = True,
                           name: str = 'carbon_structure') -> None:
    """
    Write a LAMMPS data file for a carbon structure (one atom type, mass 12.0107).

    The 'full' atom style with a Bonds section is used when bonds are given, the 'atomic'
    style otherwise. write_bonds=False forces 'atomic' for a bonded structure: a reactive pair
    style (EDIP, Tersoff, ReaxFF) rebuilds bonds from the geometry and ignores the list, while
    LAMMPS still has to keep every bonded partner communicable, which aborts with "Bond atom
    missing in image check" once atoms move, or forces a wide comm_modify cutoff.

    Parameters
    ----------
    Box : list of float
        Box edges, Angstrom; the box is written from 0.
    coord : (N, 3) ndarray
        Atom coordinates, Angstrom.
    bonds : (M, 2) ndarray of int, optional
        Bonds as 0-based atom indices.
    write_bonds : bool, optional
        Write the bonds when there are any. Default True.
    name : str, optional
        Output path without extension; '.data' is appended. Default 'carbon_structure'.

    Raises
    ------
    ValueError
        If Box or coord is missing.
    """
    if not (Box is not None):
        raise ValueError("A simulation box is required")
    if not (coord is not None):
        raise ValueError('Generate structure first')

    # without bonds: 'atomic' style (no Bonds section, no molecule id or charge per atom)
    file_name = name + '.data'

    with open(file_name,'w') as file:
        file.write("LAMMPS data file. \n\n")
        if write_bonds and bonds is not None and len(bonds) > 0:
            file.write(f"{coord.shape[0]} atoms\n{bonds.shape[0]} bonds\n\n")
            file.write("1 atom types\n1 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n")
        else:
            file.write(f"{coord.shape[0]} atoms\n\n")
            file.write("1 atom types\n0 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n")

        file.write(f"0.0 {Box[0]:.3f} xlo xhi\n0.0 {Box[1]:.3f} ylo yhi\n0.0 {Box[2]:.3f} zlo zhi\n\n")
        file.write("Masses\n\n1 12.0107\n\n")

        if write_bonds and bonds is not None and len(bonds) > 0:
            file.write("Atoms # full\n\n")
            for kk, k_coord in enumerate(coord, start=1):
                x, y, z = k_coord
                file.write(f"{kk} 1 1 0.0 {x:.4f} {y:.4f} {z:.4f}\n")
            file.write("\n\nBonds \n\n")
            for kk, row in enumerate(bonds, start=1):
                file.write(f"{kk} 1 {int(row[0]+1)} {int(row[1]+1)}\n")
        else:
            file.write("Atoms # atomic\n\n")
            for kk, k_coord in enumerate(coord, start=1):
                x, y, z = k_coord
                file.write(f"{kk} 1 {x:.4f} {y:.4f} {z:.4f}\n")

        file.write("# --- END OF CARBON GEOMETRY --- #\n\n")

def print_LAMMPS_pores(coord: NDArray,
                       R: NDArray,
                       Box: NDArray,
                       periodic: NDArray,
                       LJ_vars: list = [0.1, 1.0, None],
                       LJ_type: str = '126',
                       name: str = 'pores_coord'
                       ) -> None:
    """
    Write the pores as LAMMPS spherical regions, each with a wall/region LJ fix.

    The radii are written unchanged: nanocarbon's lj pore potential and LAMMPS' lj126
    wall/region are the same function of the same variable, the distance from the atom to the
    pore surface, so passing the generation (A, B) as (epsilon, sigma) makes the anneal wall the
    generation wall. A LAMMPS region does not wrap, so every pore whose wall reaches across a
    periodic face is also written as image regions (geometry_functions.pore_images). Each wall
    gets `fix_modify ... energy yes`, so its energy enters pe.

    epsilon is written as given. A is in the atoms potential's energy unit (kcal/mol for
    harmonic, eV for tersoff) and a `units metal` deck reads eV, so convert it first with
    WALL_UNIT_TO_EV (batch._wall_params does); unconverted, a harmonic wall is 23 times too
    deep and welds carbon to the pore surface.

    Parameters
    ----------
    coord : (P, 3) ndarray
        Pore centers, Angstrom.
    R : (P,) or (P, 1) ndarray
        Pore radii, Angstrom.
    Box : (3,) ndarray
        Box edges, Angstrom.
    periodic : (3,) array of 0/1
        Periodic flags; images are written along periodic axes only.
    LJ_vars : list, optional
        [epsilon (eV for a metal deck), sigma (Angstrom), cutoff (Angstrom)]. cutoff has no
        generation counterpart (the generation potential is untruncated); None (default) cuts
        each wall at its own minimum, LJ_WALL_MINIMUM[LJ_type] * sigma, so the wall only repels
        (WCA). Default [0.1, 1.0, None], the LJPoresPotential defaults.
    LJ_type : {'126', '93'}, optional
        LAMMPS wall shape. The 9-3 wall is a different function with its own minimum, which a
        None cutoff accounts for. Default '126'.
    name : str, optional
        Output path without extension; '.pores_settings' is appended. Default 'pores_coord'.

    Raises
    ------
    ValueError
        On another LJ_type, or LJ_vars without exactly three items.
    """
    if not (LJ_type == '126' or LJ_type == '93'):
        raise ValueError("Wall potential type not among those considered")
    if not (len(LJ_vars) == 3):
        raise ValueError("LJ parameters not consistent. Provide a list with 3 items.")

    epsilon, sigma, cutoff = LJ_vars
    if cutoff is None:
        cutoff = LJ_WALL_MINIMUM[LJ_type] * sigma

    radiusses = np.asarray(R, dtype = float).ravel()

    # the wall acts up to 'cutoff' beyond the pore surface, so an image is needed whenever that
    # shell reaches back into the box, not only when the sphere itself crosses a face.
    coord_img, R_img, parent = gf.pore_images(coord, radiusses, Box, periodic, reach = cutoff)

    file_name = name+'.pores_settings'

    with open(file_name,'w') as file:
        # full precision: rounding sigma and cutoff separately moves the cut off the minimum
        file.write(f"variable epsilon equal {epsilon:.6f}\n")
        file.write(f"variable sigma equal {sigma:.6f}\n")
        file.write(f"variable cutoff equal {cutoff:.6f}\n\n")

        file.write(f"# --- set of regions and lj{LJ_type} walls defining pores --- #\n")
        file.write(f"# --- {len(coord_img)} regions for {len(radiusses)} pores (periodic images included) --- #\n\n")

        # name each region after its parent pore, so a LAMMPS abort on 'sph_42_3' names the pore
        # that leaked an atom rather than an opaque running index.
        images_seen: dict[int, int] = {}
        for centre, radius, pore in zip(coord_img, R_img, parent):
            k = images_seen.get(int(pore), 0)
            images_seen[int(pore)] = k + 1

            ID = f"sph_{int(pore)+1}_{k}"
            x, y, z = centre
            file.write(f"region {ID} sphere {x:.4f} {y:.4f} {z:.4f} {radius:.4f} side out\n")
            ID2 = f"wall_{int(pore)+1}_{k}"
            file.write(f"fix {ID2} all wall/region {ID} lj{LJ_type} ${{epsilon}} ${{sigma}} ${{cutoff}}\n")
            file.write(f"fix_modify {ID2} energy yes\n\n")

        file.write("# --- end of lines defining the pores --- #\n\n")
