"""Synthetic panel generator for the simulation-recovery experiment.

Implements the design of paper Section 5.1 ("Design"): a balanced panel of
80 assets over 50 months, fixed asset characteristics on a 9x9 grid in
[0,1]^2 with the upper-right corner omitted, a scalar monthly context, and
two known smooth maps phi_h(c) and beta_h(x) (H=2) that generate returns
through the bilinear factor model r_{i,t+1} = beta_tilde_it^T phi_tilde_t +
sigma_r * epsilon_{i,t+1}.
"""

import numpy as np

from src import config


def true_phi(c: np.ndarray) -> np.ndarray:
    """phi_1(c), phi_2(c), Section 5.1."""
    phi1 = 0.5 + np.sin(2 * np.pi * c) + 0.5 * np.cos(4 * np.pi * c) + (1.0 / 3.0) * np.sin(6 * np.pi * c)
    phi2 = -0.5 + np.cos(2 * np.pi * c) - 0.5 * np.sin(4 * np.pi * c) + (1.0 / 3.0) * np.cos(6 * np.pi * c)
    return np.stack([phi1, phi2], axis=-1)


def true_beta(x: np.ndarray) -> np.ndarray:
    """beta_1(x), beta_2(x), Section 5.1."""
    x1, x2 = x[..., 0], x[..., 1]
    beta1 = (
        np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)
        + 0.5 * np.cos(2 * np.pi * (x1 + x2))
        + (1.0 / 3.0) * np.sin(4 * np.pi * x1) * np.sin(2 * np.pi * x2)
    )
    beta2 = (
        np.cos(2 * np.pi * x2) * np.sin(np.pi * x1)
        - 0.5 * np.sin(2 * np.pi * (x1 - x2))
        + (1.0 / 3.0) * np.cos(2 * np.pi * x1) * np.cos(4 * np.pi * x2)
    )
    return np.stack([beta1, beta2], axis=-1)


def asset_grid() -> np.ndarray:
    """A regular 9x9 grid in [0,1]^2 with the upper-right corner omitted,
    giving 80 = 9*9 - 1 assets (Section 5.1)."""
    grid = np.linspace(0.0, 1.0, 9)
    points = np.array([[x1, x2] for x1 in grid for x2 in grid])
    points = points[~((points[:, 0] == 1.0) & (points[:, 1] == 1.0))]
    assert points.shape[0] == config.SIM_N_ASSETS
    return points


def generate_panel(seed: int):
    """Simulate the balanced panel described in Section 5.1.

    Returns a dict with x (N_assets,P), c (T,), phi_tilde (T,H), beta_tilde
    (N_assets,T,H), returns (N_assets,T) [return for month t realized at
    t+1, indexed by t=0..T-1], and the true smooth maps evaluated on the
    training grid for later comparison.
    """
    rng = np.random.default_rng(seed)
    x = asset_grid()                                   # (80,2)
    T = config.SIM_N_MONTHS
    c = (np.arange(T)) / (T - 1)                        # c_t=(t-1)/49, Section 5.1 (0-indexed here)

    phi0 = true_phi(c)                                   # (T,2)
    beta0 = true_beta(x)                                 # (80,2)

    sigma_beta = np.sqrt(config.SIM_SIGMA_BETA2)
    sigma_phi = np.sqrt(config.SIM_SIGMA_PHI2)
    sigma_r = np.sqrt(config.SIM_SIGMA_R2)

    N, H = x.shape[0], config.SIM_H_TRUE
    phi_tilde = phi0 + sigma_phi * rng.standard_normal((T, H))
    beta_tilde = beta0[:, None, :] + sigma_beta * rng.standard_normal((N, T, H))

    signal = np.einsum("nth,th->nt", beta_tilde, phi_tilde)
    noise = sigma_r * rng.standard_normal((N, T))
    returns = signal + noise

    return {
        "x": x, "c": c, "phi0": phi0, "beta0": beta0,
        "phi_tilde": phi_tilde, "beta_tilde": beta_tilde,
        "returns": returns, "true_signal": signal,
    }


def dense_context_grid(n: int = config.SIM_DENSE_GRID_POINTS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def dense_characteristic_grid(n: int = config.SIM_DENSE_GRID_POINTS) -> np.ndarray:
    g = np.linspace(0.0, 1.0, n)
    return np.array([[a, b] for a in g for b in g])
