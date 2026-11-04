# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Write a saved CarbonStructure as a LAMMPS data file.

Usage: python use_cases/write_lammps_data.py
"""

import json
from nanocarbon import CarbonStructure
from nanocarbon import io_functions as io
from pathlib import Path



data_dir = Path(__file__).parent / "data"
s_file = data_dir / "example_structure.json"
data = json.load(open(s_file))
s_loaded = CarbonStructure.from_dict(data)
out_dir = Path(__file__).parent / 'output' / 'lammps_data'
out_dir.mkdir(parents=True, exist_ok=True)
io.print_LAMMPS_structure(Box = list(s_loaded.Box), 
                          coord = s_loaded.coordinates,
                          bonds = s_loaded.bonds,
                          name = str(out_dir / 'example_structure'))