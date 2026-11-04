# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Load a CarbonStructure saved with to_dict (stored bonds, no re-bonding).

Usage: python use_cases/load_from_json.py
"""

from pathlib import Path
from nanocarbon import CarbonStructure
from nanocarbon import io_functions as io
import json

data_dir = Path(__file__).parent / "data"
s_file = data_dir / "example_structure.json"

data = json.load(open(s_file))
s_loaded = CarbonStructure.from_dict(data)
print(f"Finished loading from data file")


print(f'Number of atoms:{s_loaded.atoms_number}')
print(f'Coordinate of the first 5 atoms: {s_loaded.coordinates[:5]}')
print(f'First 5 atoms bonds: {s_loaded.bonds[:5]}')
