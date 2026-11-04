# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Response-surface surrogate (the "ML particle") of the surrogate-assisted PSO.

fit_surrogate fits a forward model f-hat: parameters -> cost on the swarm history: a
polynomial expansion of the raw parameters, each resulting feature standardized, then an
elastic-net regression. propose minimizes f-hat over the bounds to suggest one candidate
per iteration, which enters the swarm only through ParticleSwarm.consider, never as a
swarm row. No generation code is imported, so it is tested on analytic functions.
"""

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from scipy.optimize import minimize, OptimizeResult


def fit_surrogate(parameters: np.ndarray,  
                  cost: np.ndarray, 
                  cost_var: np.ndarray,
                  degree: int = 2,
                  alpha: float = 1e-3,
                  l1: float = 0.5,
                  max_iter: int = 5000,
                  n_split: int = 5,
                  eps: float = 1e-3,
                  rng: int | None = None,
                  ) -> tuple[Pipeline, float]:
    """
    Fit the surrogate f-hat: parameters -> cost.

    Pipeline: PolynomialFeatures(degree) on the raw parameters, StandardScaler on each
    resulting feature, ElasticNet(alpha, l1). Samples are weighted by 1 / (cost_var + eps), so
    candidates whose replicas agree count more. A K-fold cross-validated R^2 is returned as a
    diagnostic only.

    Parameters
    ----------
    parameters : (M, D) ndarray
        Positions of the M evaluated candidates with a finite cost.
    cost : (M,) ndarray
        Their replica-averaged costs.
    cost_var : (M,) ndarray
        Variance of the cost across each candidate's replicas.
    degree : int, optional
        Polynomial degree. Default 2.
    alpha : float, optional
        Regularization strength. Default 1e-3.
    l1 : float, optional
        L1 share of the penalty (0: ridge, 1: lasso). Default 0.5.
    max_iter : int, optional
        Coordinate-descent iterations of ElasticNet. On a ConvergenceWarning raise this, not
        alpha, to keep the regularization light. Default 5000.
    n_split : int, optional
        Folds of the cross-validation (capped at M). Default 5.
    eps : float, optional
        Floor of the variance in the weights. Default 1e-3.
    rng : int, optional
        Seed of the fold shuffling.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline.
    cv_score : float
        Mean cross-validated R^2; negative when the model predicts worse than the mean.
    """

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree = degree, include_bias = False)),
        ("scale", StandardScaler()),
        ("en", ElasticNet(alpha = alpha, l1_ratio = l1, max_iter = max_iter))
    ])

    w = 1.0 / (cost_var + eps)

    n_split = min(n_split, len(cost)) # no more folds than samples

    kf = KFold(n_splits = n_split, shuffle = True, random_state = rng)

    cv_score = cross_val_score(estimator = pipe, 
                              X = parameters,
                              y = cost, 
                              cv = kf, 
                              params={"en__sample_weight": w} # type: ignore[call-arg]
                              ).mean()

    pipe.fit(parameters, cost, en__sample_weight = w)

    return pipe, cv_score

def propose(model: Pipeline, 
            bounds: list[tuple], 
            n_starts: int = 20, 
            tol: float = 1e-5,
            rng: int | None = None
            ) -> np.ndarray | None: 
    """
    Candidate that minimizes the surrogate over the bounds.

    L-BFGS-B (finite-difference gradient) from n_starts random starting points; the best end
    point is returned.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted surrogate (fit_surrogate).
    bounds : list of (float, float)
        (low, high) of each dimension.
    n_starts : int, optional
        Random starting points. Default 20.
    tol : float, optional
        Tolerance of each minimization. Default 1e-5.
    rng : int, optional
        Seed of the starting points.

    Returns
    -------
    (D,) ndarray or None
        The candidate; None only if no minimization returned a finite value.
    """

    def _pipeline_predict(X: np.ndarray) -> float:
        """
        The surrogate as a scalar function of one position, for scipy's minimize.

        Parameters
        ----------
        X : (D,) ndarray
            Position.

        Returns
        -------
        float
            Predicted cost.
        """

        data = X.reshape(1,-1) # one sample per row, the sklearn convention
        return float(model.predict(data)[0]) 

    gen = np.random.default_rng(rng)
    lows  = np.array([lo for lo, hi in bounds])
    highs = np.array([hi for lo, hi in bounds])
    starts = gen.uniform(lows, highs, size=(n_starts, len(bounds))) #returns (n_starts, D)

    best_x = None
    best_cost:float = float('inf')
    for start in starts:
        res: OptimizeResult = minimize(_pipeline_predict, start, jac = '2-point', method='L-BFGS-B', bounds = bounds, tol = tol)

        x_iter = np.asarray(res.x)
        cost_iter = float(res.fun)

        if cost_iter < best_cost:
            best_cost = cost_iter
            best_x = x_iter.copy()

    return best_x
