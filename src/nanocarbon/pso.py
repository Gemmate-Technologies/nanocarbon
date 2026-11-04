# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Global-best particle swarm optimizer over a bounded continuous space.

Domain-agnostic: the caller (calibrate.py) calls step(), evaluates the positions and
passes the costs to update(). It imports no generation code, so it is tested on analytic
functions.
"""
from __future__ import annotations
import json
import os
import numpy as np

class ParticleSwarm:
    """
    Global-best particle swarm over a box of D bounded dimensions.

    Each step moves every particle by its velocity: inertia plus pulls toward its own best
    position (pbest) and the swarm's best (gbest). Velocities are clipped to v_max_frac of each
    range; a particle leaving the box is put on the wall with that velocity component zeroed.

    Parameters
    ----------
    bounds : list of (float, float)
        (low, high) of each dimension.
    swarm_size : int, optional
        Number of particles P. Default 16.
    w : float, optional
        Inertia weight. Default 0.7.
    c1, c2 : float, optional
        Cognitive (toward pbest) and social (toward gbest) weights. Default 1.5 each.
    v_max_frac : float, optional
        Velocity cap, as a fraction of each dimension's range. Default 0.2.
    rng : numpy.random.Generator, optional
        Random generator; default a fresh one.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        swarm_size: int = 16,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max_frac: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:

        self.bounds = np.array(bounds, dtype=float)   # D × 2
        self.D = len(bounds)
        self.P = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = rng if rng is not None else np.random.default_rng()

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self._v_range = v_max_frac * (hi - lo)

        self.X = self.rng.uniform(lo, hi, (self.P, self.D))
        self.V = self.rng.uniform(-self._v_range, self._v_range, (self.P, self.D))
        self.pbest = self.X.copy()  # best position of each particle
        self.pbest_cost = np.full(self.P, np.inf)
        self.gbest = self.X[0].copy()
        self.gbest_cost = np.inf
        self.history: list[dict] = []

        self.meta: dict = {} 

    def update(self, costs: np.ndarray, extra: list | None = None) -> None:
        """
        Update pbest and gbest with the costs of the current positions, and append the iteration
        to history.

        Parameters
        ----------
        costs : (P,) array
            Cost of each particle at the current X (inf for a failed candidate).
        extra : list, optional
            Per-particle data stored as is in history (calibrate passes the per-anchor terms and
            the replica variance).
        """
        improved = costs < self.pbest_cost
        self.pbest[improved] = self.X[improved]
        self.pbest_cost[improved] = costs[improved]

        best_idx = int(np.argmin(self.pbest_cost))
        if self.pbest_cost[best_idx] < self.gbest_cost:
            self.gbest = self.pbest[best_idx].copy()
            self.gbest_cost = float(self.pbest_cost[best_idx])

        self.history.append({'gbest_cost': self.gbest_cost, 
                             'gbest': self.gbest.tolist(),
                             'X': self.X.tolist(), # positions that produced these costs
                             'costs': np.asarray(costs).tolist(), # surrogate target; accepts an array or a list
                             'extra': extra, # the caller's per-particle data, stored as is (or None)
                             })

    def step(self) -> np.ndarray:
        """
        Move every particle one step.

        Returns
        -------
        (P, D) ndarray
            The new positions X.
        """
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        r1 = self.rng.uniform(0.0, 1.0, (self.P, self.D))
        r2 = self.rng.uniform(0.0, 1.0, (self.P, self.D))

        self.V = (self.w * self.V # inertia term
                  + self.c1 * r1 * (self.pbest - self.X) # cognitive term: toward the particle's best
                  + self.c2 * r2 * (self.gbest - self.X)) # social term: toward the swarm's best
        self.V = np.clip(self.V, -self._v_range, self._v_range)

        self.X = self.X + self.V
        at_wall = (self.X < lo) | (self.X > hi)
        self.X = np.clip(self.X, lo, hi)
        self.V[at_wall] = 0.0

        return self.X

    def consider(self, x: np.ndarray, cost: float) -> None:
        """
        Offer an outside candidate (the surrogate's proposal) as gbest: it is adopted only if its
        cost beats the current gbest. The swarm rows are not changed.

        Parameters
        ----------
        x : (D,) array
            Candidate position.
        cost : float
            Its cost.

        Raises
        ------
        ValueError
            If x does not have D components.
        """
        if np.shape(np.asarray(x).ravel()) != np.shape(self.gbest.ravel()):
            raise ValueError(f"Shape of candidate differ from gbest: x: {np.shape(np.asarray(x).ravel())} vs. {np.shape(self.gbest.ravel())}")
        if cost < self.gbest_cost:
            self.gbest_cost = float(cost)
            self.gbest = np.asarray(x).ravel().copy()

    def checkpoint(self, path: str) -> None:
        """
        Write the swarm state and its full history to JSON (the restart file).

        Parameters
        ----------
        path : str
            Output file; its directory is created if missing.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(
                {'X': self.X.tolist(),
                 'V': self.V.tolist(),
                 'pbest': self.pbest.tolist(),
                 'pbest_cost': self.pbest_cost.tolist(),
                 'gbest': self.gbest.tolist(),
                 'gbest_cost': self.gbest_cost,
                 'history': self.history,
                 'meta': self.meta},
                f,
                indent=2,
            )

    @classmethod
    def restore(cls, bounds: list, path: str, **kwargs):

        """
        Rebuild a swarm from a checkpoint.

        Parameters
        ----------
        bounds : list of (float, float)
            The bounds of the original run.
        path : str
            The checkpoint file.
        **kwargs
            The other constructor arguments of the original run (swarm_size, w, ...).

        Returns
        -------
        ParticleSwarm

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        ValueError
            If the stored positions do not match swarm_size and the number of bounds.
        """
        if not (os.path.exists(path)):
            raise FileNotFoundError("PSO restore file .json not found.")

        with open(path, 'r') as f:
            params = json.load(f)

        obj = cls(bounds, **kwargs)

        obj.X = np.array(params['X'])
        if not (obj.X.shape == (obj.P, obj.D)):
            raise ValueError(f"checkpoint shape {obj.X.shape} != swarm ({obj.P},{obj.D}); bounds/swarm_size mismatch")

        obj.V = np.array(params['V'])
        obj.pbest = np.array(params['pbest'])
        obj.pbest_cost = np.array(params['pbest_cost'])
        obj.gbest = np.array(params['gbest'])
        obj.gbest_cost = float(params['gbest_cost'])
        obj.history = params['history']

        obj.meta = params.get('meta', {})

        return obj

