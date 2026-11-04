# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
nanocarbon: amorphous nanoporous carbon by explicit pore pre-conditioning.

Pores are placed first, carbon is grown ring by ring around them, and the structure is
written for a LAMMPS anneal; the post-processing measures constructed and imported
structures alike. The names imported here are the public API.
"""
from .structures import CarbonStructure, PoreStructure
from .generators import PoresGenerator, RingGenerator, RandomGenerator
from .potential_classes import HarmonicPotential, TersoffPotential, LJPoresPotential
from .config import GenerationConfig
from .results import RunResult, BatchResults, load_psd
from .pipeline import run_generation
from .batch import run_batch
from .objective import Anchors, psd_emd, anchor_cost
from .calibrate import run_calibration
