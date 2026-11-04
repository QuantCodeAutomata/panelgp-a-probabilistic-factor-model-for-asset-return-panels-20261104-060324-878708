"""Least-squares factor rotation used to align an estimated PanelGP solution
with the known simulation truth (paper Section 5.1, step 3).

Custom implementation (paper equation for the rotation is not spelled out;
the construction below is the standard factor-model identification fix:
estimated factors and loadings are identified only up to an invertible
H x H transform Upsilon, since beta_tilde^T phi_tilde = (Upsilon^-T
beta_tilde)^T (Upsilon phi_tilde) for any invertible Upsilon).
"""

import numpy as np


def fit_rotation(phi_true: np.ndarray, phi_hat: np.ndarray) -> np.ndarray:
    """Upsilon minimizing sum_t || phi_true_t - Upsilon @ phi_hat_t ||^2.

    Args:
        phi_true: (T, H) true realized factors.
        phi_hat: (T, H) estimated (posterior-mean) realized factors.
    Returns:
        Upsilon, (H, H), solving the no-intercept multivariate least squares
        problem phi_true ~ phi_hat @ Upsilon^T.
    """
    # phi_true (T,H) = phi_hat (T,H) @ Upsilon^T  =>  Upsilon^T = pinv(phi_hat) @ phi_true
    UpsilonT, *_ = np.linalg.lstsq(phi_hat, phi_true, rcond=None)
    return UpsilonT.T


def apply_rotation(upsilon: np.ndarray, phi_hat: np.ndarray, beta_hat: np.ndarray):
    """Transform factors by Upsilon and loadings by Upsilon^{-T} (Section 5.1)."""
    upsilon_inv_T = np.linalg.inv(upsilon).T
    phi_aligned = phi_hat @ upsilon.T
    beta_aligned = beta_hat @ upsilon_inv_T.T
    return phi_aligned, beta_aligned


def transform_innovation_covariances(upsilon: np.ndarray, D_beta: np.ndarray, D_phi: np.ndarray):
    """D_phi -> Upsilon D_phi Upsilon^T, D_beta -> Upsilon^-T D_beta Upsilon^-1."""
    upsilon_inv = np.linalg.inv(upsilon)
    D_phi_new = upsilon @ D_phi @ upsilon.T
    D_beta_new = upsilon_inv.T @ D_beta @ upsilon_inv
    return D_beta_new, D_phi_new


def r_squared(true_values: np.ndarray, fitted_values: np.ndarray) -> float:
    """Pooled R^2 of fitted values against true values on a grid (Section 5.1)."""
    num = np.sum((true_values - fitted_values) ** 2)
    den = np.sum((true_values - true_values.mean()) ** 2)
    return float(1.0 - num / den)
