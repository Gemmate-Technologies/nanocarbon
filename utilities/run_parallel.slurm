#!/bin/bash
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
#
# Run a nanocarbon batch on one node: `sbatch utilities/run_parallel.slurm` from the repository
# root; the log goes to slurm-<jobid>.out there. The batch is one Python process with a process
# pool: one task with many CPUs, not MPI. Keep N_WORKERS (config file) at most --cpus-per-task
# and low enough that peak memory per worker x N_WORKERS fits the node; a worker killed for
# memory stops the batch, and resubmitting resumes from the runs already on disk.

#SBATCH --job-name=nanocarbon_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# module load ...   # site-specific: whatever your cluster needs for numpy/scipy
ulimit -s unlimited # the Voronoi tessellation in compute_connectivity can overflow a small stack
source "$HOME/venv_nanocarbon/bin/activate"

python -m nanocarbon generate utilities/batch_config.py
