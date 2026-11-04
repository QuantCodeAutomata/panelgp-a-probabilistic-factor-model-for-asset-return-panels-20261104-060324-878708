"""Experiment 1: simulation recovery of smooth maps and variance parameters
(paper Section 5.1 "Design", Section 5.2 "Results", Table 1).

A Monte Carlo simulation verifies that the PanelGP estimator recovers known
smooth loading and factor maps, the realized training factors, and the
innovation/noise variances from generated panel returns.

Run: python -m exp.exp1 [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import config
from src.alignment import apply_rotation, fit_rotation, r_squared, transform_innovation_covariances
from src.gp_model import PanelGPConfig, PanelGPModel, fit_panelgp
from data.generate_synthetic import (
    dense_characteristic_grid,
    dense_context_grid,
    generate_panel,
    true_beta,
    true_phi,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _oracle_variance(beta0: np.ndarray, phi0: np.ndarray) -> np.ndarray:
    """Equation (6)-style oracle predictive variance for the known
    data-generating process: Var(r_{i,t+1}) = beta0(x_i)^T Sigma_phi beta0(x_i)
    + phi0(c_t)^T Sigma_beta phi0(c_t) + tr(Sigma_beta Sigma_phi) + sigma_r^2,
    where Sigma_beta=sigma_beta^2 I_H, Sigma_phi=sigma_phi^2 I_H are the true
    innovation covariances (Section 5.1)."""
    sb2, sf2, sr2 = config.SIM_SIGMA_BETA2, config.SIM_SIGMA_PHI2, config.SIM_SIGMA_R2
    H = beta0.shape[1]
    term_phi_innov = sf2 * np.sum(beta0 ** 2, axis=1)            # (N,)
    term_beta_innov = sb2 * np.sum(phi0 ** 2, axis=1)            # (T,)
    floor = sb2 * sf2 * H + sr2
    return term_phi_innov[:, None] + term_beta_innov[None, :] + floor  # (N,T)


def _fitted_variance(model: PanelGPModel, X: torch.Tensor, C_train: torch.Tensor, s_r: float) -> np.ndarray:
    """Total fitted predictive variance V^0+V^phi+V^beta+tr(D_beta D_phi)+sigma_r^2
    per (asset, month), converted back to original return units (x s_r^2)."""
    T = C_train.shape[0]
    out = np.zeros((X.shape[0], T))
    for t in range(T):
        V0, Vphi, Vbeta, trDbDp, sigma_r2 = model.variance_decomposition(X, C_train[t], C_train)
        total = (V0 + Vphi + Vbeta + trDbDp + sigma_r2).numpy()
        out[:, t] = total * (s_r ** 2)
    return out


def run_one_replication(seed: int, epochs: int, n_inducing: int, verbose: bool = False) -> dict:
    panel = generate_panel(seed)
    x, c = panel["x"], panel["c"]
    returns = panel["returns"]  # (N,T)
    N, T = returns.shape

    s_r = float(np.sqrt(np.mean(returns ** 2)))
    returns_scaled = returns / s_r

    X_train = torch.tensor(np.repeat(x, T, axis=0), dtype=torch.float32)          # (N*T, P)
    t_idx_train = torch.tensor(np.tile(np.arange(T), N), dtype=torch.long)        # (N*T,)
    R_train = torch.tensor(returns_scaled.reshape(-1), dtype=torch.float32)       # (N*T,)
    C_train = torch.tensor(c, dtype=torch.float32).unsqueeze(-1)                 # (T, Q=1)

    cfg = PanelGPConfig(
        P=config.SIM_N_CHARACTERISTICS, Q=config.SIM_N_CONTEXT, H=config.SIM_H_TRUE,
        T=T, M=n_inducing, loading_kernel="rbf", factor_kernel="rbf",
        isotropic_innovations=True,
    )
    model = PanelGPModel(cfg, seed=seed)
    model.set_initial_values(d_loading=1.0)  # d=1 for the RBF loading kernel, Equation C.6

    fit_panelgp(
        model, X_train, t_idx_train, R_train, C_train, epochs=epochs,
        batch_size=min(config.BATCH_SIZE_PAPER, N * T), lr_variational=config.LR_VARIATIONAL,
        lr_model=config.LR_MODEL, n_factor_samples=config.N_FACTOR_SAMPLES,
        grad_clip=config.GRAD_CLIP_NORM, lr_halving_fractions=config.LR_HALVING_FRACTIONS,
        seed=seed, verbose=verbose,
    )

    # --- alignment: rotate the estimated solution onto the simulation truth ---
    X_assets = torch.tensor(x, dtype=torch.float32)
    phi_hat_train, _ = model.realized_training_factors(C_train)
    phi_hat_train = phi_hat_train.numpy()
    upsilon = fit_rotation(panel["phi_tilde"], phi_hat_train)

    dense_c = dense_context_grid()
    dense_x = dense_characteristic_grid()
    phi_hat_dense, _ = model.smooth_factor(torch.tensor(dense_c, dtype=torch.float32).unsqueeze(-1), C_train)
    beta_hat_dense, _ = model.smooth_loading(torch.tensor(dense_x, dtype=torch.float32), C_train)
    phi_hat_dense_aligned, beta_hat_dense_aligned = apply_rotation(upsilon, phi_hat_dense.numpy(), beta_hat_dense.numpy())

    D_beta_hat, D_phi_hat = model.innovation_covariances()
    D_beta_aligned, D_phi_aligned = transform_innovation_covariances(upsilon, D_beta_hat.numpy(), D_phi_hat.numpy())
    sigma_beta2_hat = float(np.mean(np.diag(D_beta_aligned)))
    sigma_phi2_hat = float(np.mean(np.diag(D_phi_aligned)))
    sigma_r2_hat = float(model.sigma_r2().item()) * (s_r ** 2)

    # --- R^2 of fitted smooth maps against the true maps on dense grids ---
    phi_true_dense = true_phi(dense_c)
    beta_true_dense = true_beta(dense_x)
    r2_factor = r_squared(phi_true_dense, phi_hat_dense_aligned * s_r)
    r2_loading = r_squared(beta_true_dense, beta_hat_dense_aligned)

    # --- correlation of recovered realized training factors with the truth ---
    phi_hat_train_aligned, _ = apply_rotation(upsilon, phi_hat_train, phi_hat_train)
    factor_corr = float(np.mean([
        np.corrcoef(panel["phi_tilde"][:, h], phi_hat_train_aligned[:, h] * s_r)[0, 1]
        for h in range(cfg.H)
    ]))

    # --- predictive means: beta_hat(x_i)^T phi_hat(c_t) vs true beta0(x_i)^T phi0(c_t) ---
    phi_hat_train_s, _ = model.smooth_factor(C_train, C_train)
    beta_hat_train_s, _ = model.smooth_loading(X_assets, C_train)
    _, beta_hat_train_aligned = apply_rotation(upsilon, phi_hat_train_s.numpy(), beta_hat_train_s.numpy())
    phi_hat_train_s_aligned, _ = apply_rotation(upsilon, phi_hat_train_s.numpy(), phi_hat_train_s.numpy())
    pred_mean = (beta_hat_train_aligned @ phi_hat_train_s_aligned.T) * s_r  # (N,T) original units
    true_cond_mean = panel["beta0"] @ panel["phi0"].T
    r2_pred_mean = r_squared(true_cond_mean, pred_mean)

    # --- predictive variance vs oracle variance (Equation 6) ---
    oracle_var = _oracle_variance(panel["beta0"], panel["phi0"])
    fitted_var = _fitted_variance(model, X_assets, C_train, s_r)
    var_corr = float(np.corrcoef(oracle_var.reshape(-1), fitted_var.reshape(-1))[0, 1])
    var_ratio = float(fitted_var.mean() / oracle_var.mean())

    return {
        "seed": seed,
        "r2_factor": r2_factor,
        "r2_loading": r2_loading,
        "factor_correlation": factor_corr,
        "sigma_beta2_hat": sigma_beta2_hat,
        "sigma_phi2_hat": sigma_phi2_hat,
        "sigma_r2_hat": sigma_r2_hat,
        "r2_predictive_mean": r2_pred_mean,
        "predictive_variance_correlation": var_corr,
        "predictive_variance_ratio": var_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's epochs/replications instead of the reduced, time-budgeted configuration")
    args = parser.parse_args()

    epochs = config.SIM_EPOCHS_PAPER if args.full else config.SIM_EPOCHS_RUN
    n_reps = config.SIM_N_REPLICATIONS_PAPER if args.full else config.SIM_N_REPLICATIONS_RUN
    n_inducing = config.SIM_M_INDUCING

    t0 = time.time()
    records = []
    for rep in range(n_reps):
        seed = config.RANDOM_SEED_BASE + rep
        print(f"[exp1] replication {rep + 1}/{n_reps}, seed={seed}, epochs={epochs}")
        rec = run_one_replication(seed, epochs, n_inducing, verbose=(rep == 0))
        print(f"[exp1]   r2_factor={rec['r2_factor']:.4f} r2_loading={rec['r2_loading']:.4f} "
              f"sigma_r2_hat={rec['sigma_r2_hat']:.4f}")
        records.append(rec)
    elapsed = time.time() - t0

    arr = {k: np.array([r[k] for r in records]) for k in records[0] if k != "seed"}
    summary = {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0} for k, v in arr.items()}

    out = {
        "config": {"epochs": epochs, "n_replications": n_reps, "n_inducing": n_inducing, "full": args.full},
        "elapsed_seconds": elapsed,
        "replications": records,
        "summary": summary,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[exp1] done in {elapsed:.1f}s, summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
