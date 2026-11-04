"""IPCA benchmarks estimated by alternating least squares (paper Section 6.2).

Context7 was consulted for an off-the-shelf IPCA implementation (the `ipca`
PyPI package implements Kelly, Pettenuzzo and Shahidi 2019 ALS on balanced
panels indexed by (entity, date)); it is not part of the pinned dependency
set and its unbalanced-panel handling does not match the paper's forecasting
design, so the estimator below follows Equation (29) and the paper's
alternating-least-squares description directly.
"""

from __future__ import annotations

import numpy as np


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def fit_ipca(
    X_by_month: list[np.ndarray],
    r_by_month: list[np.ndarray],
    H: int,
    with_intercept: bool,
    n_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Alternating least squares estimate of Gamma (and gamma_alpha), Eq. (29).

    Args:
        X_by_month: list of (N_t, P) characteristic matrices, training months.
        r_by_month: list of (N_t,) excess-return vectors, same months.
        H: number of latent factors.
        with_intercept: add an intercept loading estimated as in IPCA+int.
    Returns:
        dict with Gamma ((P+1, H)), gamma_alpha ((P+1,) or None), factors
        (T, H) realized training factors, sigma_e2 (pooled residual variance).
    """
    rng = np.random.default_rng(seed)
    P = X_by_month[0].shape[1]
    T = len(X_by_month)
    Gamma = rng.standard_normal((P + 1, H)) * 0.1
    gamma_alpha = rng.standard_normal(P + 1) * 0.1 if with_intercept else None
    factors = np.zeros((T, H))
    Xi = [_add_intercept(X) for X in X_by_month]

    prev_loss = np.inf
    for _ in range(n_iter):
        # --- factor step: given Gamma, solve least squares per month ---
        for t in range(T):
            Z = Xi[t] @ Gamma  # (N_t, H)
            resid = r_by_month[t].copy()
            if with_intercept:
                resid = resid - Xi[t] @ gamma_alpha
            factors[t], *_ = np.linalg.lstsq(Z, resid, rcond=None)

        # --- loading step: given factors, solve least squares pooling months ---
        design_rows, target_rows = [], []
        for t in range(T):
            Xt = Xi[t]
            design = np.einsum("np,h->nph", Xt, factors[t]).reshape(Xt.shape[0], -1)
            design_rows.append(design)
            target_rows.append(r_by_month[t])
        design_full = np.concatenate(design_rows, axis=0)
        target_full = np.concatenate(target_rows, axis=0)
        if with_intercept:
            intercept_design = np.concatenate(Xi, axis=0)
            design_full = np.concatenate([design_full, intercept_design], axis=1)
        coef, *_ = np.linalg.lstsq(design_full, target_full, rcond=None)
        if with_intercept:
            Gamma = coef[: (P + 1) * H].reshape(P + 1, H)
            gamma_alpha = coef[(P + 1) * H:]
        else:
            Gamma = coef.reshape(P + 1, H)

        # --- normalization: orthonormal factor loadings (Kelly et al. 2019) ---
        U, S, Vt = np.linalg.svd(Gamma, full_matrices=False)
        Gamma = U
        factors = factors @ (np.diag(S) @ Vt).T

        preds = [Xi[t] @ Gamma @ factors[t] + (Xi[t] @ gamma_alpha if with_intercept else 0.0) for t in range(T)]
        loss = sum(np.sum((r_by_month[t] - preds[t]) ** 2) for t in range(T))
        if abs(prev_loss - loss) < tol * max(1.0, abs(prev_loss)):
            break
        prev_loss = loss

    resid_all = np.concatenate([r_by_month[t] - preds[t] for t in range(T)])
    sigma_e2 = float(np.mean(resid_all ** 2))
    return {"Gamma": Gamma, "gamma_alpha": gamma_alpha, "factors": factors, "sigma_e2": sigma_e2}


def ipca_forecast(ipca_fit: dict, X_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equation (29): mu_t, Sigma_t for a new cross-section of characteristics."""
    Xi = _add_intercept(X_t)
    Gamma = ipca_fit["Gamma"]
    factors = ipca_fit["factors"]
    phi_bar = factors.mean(axis=0)
    Sigma_phi = np.cov(factors.T) if factors.shape[1] > 1 else np.array([[factors.var()]])
    mu = Xi @ Gamma @ phi_bar
    if ipca_fit["gamma_alpha"] is not None:
        mu = mu + Xi @ ipca_fit["gamma_alpha"]
    loadings = Xi @ Gamma
    Sigma = loadings @ Sigma_phi @ loadings.T + ipca_fit["sigma_e2"] * np.eye(Xi.shape[0])
    return mu, Sigma
