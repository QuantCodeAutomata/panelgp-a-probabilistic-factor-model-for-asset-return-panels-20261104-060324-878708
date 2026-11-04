"""Experiment 5: higher moments of the bilinear predictive distribution
(paper Section 6.7 "Higher moments of the predictive distribution", Appendix
B.3, Table S3).

Reuses the annually re-estimated ("rolling") RBF/RBF PanelGP (H=5) fits of
experiment 4 (same cache, so no retraining is needed) and, for every
stock-month forecast, compares the exact bilinear predictive log-density
(Equation B.14-B.15, by Fourier inversion of the cumulant generating
function) with the Gaussian log-density that has the same mean and variance
(Equation 35).

Run: python -m exp.exp5 [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config
from src.predictive_distribution import bilinear_log_density, build_A_j, gaussian_log_density
from src.sp500_pipeline import (
    CONTEXT_COLS,
    build_stock_panel,
    load_universe_bars,
    rolling_cache_path,
    rolling_panelgp_fit_cached,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _warmup_start(train_start: str) -> str:
    return f"{int(train_start[:4]) - 3}-01-01"


def load_data(full: bool) -> dict:
    universe = config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN]
    train_start = config.SP500_TRAIN_START_PAPER if full else config.SP500_TRAIN_START_RUN
    eval_start = config.SP500_EVAL_START if full else config.SP500_EVAL_START_RUN
    eval_end = config.SP500_EVAL_END if full else config.SP500_EVAL_END_RUN
    window_months = config.ROLLING_WINDOW_MONTHS_PAPER if full else config.ROLLING_WINDOW_MONTHS_RUN
    first_year = config.ROLLING_FIRST_FORECAST_YEAR_PAPER if full else int(eval_start[:4])
    last_year = config.ROLLING_LAST_FORECAST_YEAR_PAPER if full else int(eval_end[:4])
    stride = 1 if full else config.ROLLING_REESTIMATION_STRIDE_RUN
    forecast_years = list(range(first_year, last_year + 1, stride))
    rolling_start_year = first_year - (window_months // 12) - 1
    bars_start_year = min(int(train_start[:4]), rolling_start_year)
    bars, meta = load_universe_bars(universe, _warmup_start(f"{bars_start_year}-01-01"), eval_end)
    panel, context_raw, char_cols = build_stock_panel(bars)
    print(f"[exp5] panel rows={len(panel)} forecast_years={forecast_years} window_months={window_months}")
    return {
        "panel": panel, "context_raw": context_raw, "char_cols": char_cols,
        "window_months": window_months, "forecast_years": forecast_years,
    }


def rolling_fits(data: dict, H: int, M: int, epochs: int, n_seeds: int, cache_suffix: str) -> list[dict]:
    """Same trailing-window schedule and cache tag as experiment 4's
    "PanelGP rolling" model, so a rerun of experiment 4 is reused here."""
    cache = rolling_cache_path(f"panelgp_{cache_suffix}")
    return rolling_panelgp_fit_cached(
        cache, data["panel"], data["context_raw"], data["char_cols"], CONTEXT_COLS,
        window_months=data["window_months"], forecast_years=data["forecast_years"],
        H=H, M=M, epochs=epochs, n_seeds=n_seeds, loading_kernel="rbf", factor_kernel="rbf",
        seed_base=config.RANDOM_SEED_BASE,
    )


def log_score_gains(fits: list[dict], H: int) -> pd.DataFrame:
    """For every stock-month forecast of every rolling window and seed,
    Equation (35): g = log p_bil(r) - log N(r | mu_hat, Sigma_hat_jj).

    The model is estimated on returns scaled by s_r (Section 6.1), and the
    gain is scale-invariant: log p(r) = log p_scaled(r/s_r) - log(s_r) holds
    for both densities, so the -log(s_r) term cancels in the difference and
    the gain can be computed directly in scaled units (implementation_notes:
    "subtract log s_r for original units" only matters for the density
    level, not for the gain)."""
    A = build_A_j(H)
    rows = []
    for year_fit in fits:
        year = year_fit["year"]
        for seed_idx, fitted in enumerate(year_fit["fitted"]):
            for date, group in year_fit["panel_year"].groupby("date", sort=True):
                c_row = year_fit["context"].loc[date, fitted.context_cols].to_numpy(dtype=np.float32)
                if np.isnan(c_row).any():
                    continue
                X_new = torch.tensor(group[fitted.char_cols].to_numpy(dtype=np.float32))
                c_new = torch.tensor(c_row)
                mu_s, Sigma_s = fitted.model.cross_section_predictive(X_new, c_new, fitted.C_train_tensor)
                r_scaled = group["ret_next"].to_numpy(dtype=np.float64) / fitted.s_r
                for j, ticker in enumerate(group["ticker"].tolist()):
                    m, S_tilde, sigma_r2 = fitted.model.bilinear_state(X_new[j], c_new, fitted.C_train_tensor)
                    mu_j, sigma2_j = float(mu_s[j]), float(Sigma_s[j, j])
                    r_j = float(r_scaled[j])
                    log_p_bil = bilinear_log_density(
                        r_j, m.numpy().astype(np.float64), S_tilde.numpy().astype(np.float64), A,
                        float(sigma_r2), tol=config.CGF_QUAD_TOL,
                    )
                    log_p_gauss = gaussian_log_density(r_j, mu_j, max(sigma2_j, 1e-12))
                    std_err = (r_j - mu_j) / np.sqrt(max(sigma2_j, 1e-12))
                    rows.append({
                        "date": date, "ticker": ticker, "seed": seed_idx, "year": year,
                        "gain": log_p_bil - log_p_gauss, "std_err": std_err,
                    })
    return pd.DataFrame(rows)


def bin_by_std_error(df: pd.DataFrame) -> dict:
    bins = [0.0, *config.STD_ERROR_BINS, np.inf]
    labels = ["<1", "1-2", "2-3", ">=3"]
    df = df.copy()
    df["bin"] = pd.cut(df["std_err"].abs(), bins=bins, labels=labels, include_lowest=True)
    grouped = df.groupby("bin", observed=False)["gain"].agg(["mean", "count"])
    return {str(k): {"mean_gain": float(v["mean"]), "count": int(v["count"])} for k, v in grouped.iterrows()}


def subperiod_comparison(df: pd.DataFrame) -> dict:
    years = sorted(df["year"].unique())
    mid = years[len(years) // 2] if len(years) > 1 else years[0]
    early = df[df["year"] < mid]["gain"]
    late = df[df["year"] >= mid]["gain"]
    return {
        f"before_{mid}": {"mean_gain": float(early.mean()) if len(early) else None, "n": int(len(early))},
        f"from_{mid}": {"mean_gain": float(late.mean()) if len(late) else None, "n": int(len(late))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's universe/period/epochs/seeds instead of the reduced, time-budgeted configuration")
    args = parser.parse_args()

    H = config.H_FACTORS_PAPER
    M = config.M_INDUCING_PAPER if args.full else config.M_INDUCING_RUN
    epochs = config.EPOCHS_APPLICATION_PAPER if args.full else config.EPOCHS_APPLICATION_RUN
    n_seeds = config.N_SEEDS_PAPER if args.full else config.N_SEEDS_RUN
    cache_suffix = "full" if args.full else "run"

    t0 = time.time()
    data = load_data(args.full)
    fits = rolling_fits(data, H, M, epochs, n_seeds, cache_suffix)
    df = log_score_gains(fits, H)
    elapsed = time.time() - t0

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["gain"].mean().sort_index()
    trailing = monthly.rolling(12, min_periods=1).mean()

    out = {
        "config": {"H": H, "M": M, "epochs": epochs, "n_seeds": n_seeds, "full": args.full,
                    "forecast_years": data["forecast_years"], "n_stock_months": int(len(df))},
        "elapsed_seconds": elapsed,
        "mean_log_score_gain": float(df["gain"].mean()),
        "gain_by_abs_std_error_bin": bin_by_std_error(df),
        "subperiod_comparison": subperiod_comparison(df),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp5_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    monthly.to_frame("mean_gain").assign(trailing_12m=trailing).to_csv(RESULTS_DIR / "exp5_monthly_gain.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, alpha=0.4, label="monthly mean gain")
    ax.plot(trailing.index, trailing.values, color="firebrick", label="12-month trailing average")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Experiment 5: log-score gain of the bilinear density over the Gaussian density")
    ax.set_xlabel("month")
    ax.set_ylabel("log-score gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_log_score_gain.png", dpi=120)

    print(f"[exp5] done in {elapsed:.1f}s, {len(df)} stock-month observations")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
