"""Experiment 6: predicted variance terms and risk calibration (paper
Section 6.8 "Predicted variance terms", Figure 4, Section 4.3 Equation 26-27).

Reuses the annually re-estimated ("rolling") RBF/RBF PanelGP (H=5) fits of
experiment 4 (same cache) and decomposes the predicted return variance into
posterior uncertainty V^0, factor-innovation V^phi, loading-innovation
V^beta, and the tr(D_beta D_phi) + sigma_r^2 variance floor, then checks
whether months and stocks with higher predicted variance also have higher
realized squared forecast errors.

Run: python -m exp.exp6 [--full]
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
from scipy import stats

from src import config
from src.sp500_pipeline import (
    CONTEXT_COLS,
    build_stock_panel,
    load_universe_bars,
    panelgp_variance_terms,
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
    print(f"[exp6] panel rows={len(panel)} forecast_years={forecast_years} window_months={window_months}")
    return {
        "panel": panel, "context_raw": context_raw, "char_cols": char_cols,
        "window_months": window_months, "forecast_years": forecast_years,
    }


def rolling_fits(data: dict, H: int, M: int, epochs: int, n_seeds: int, cache_suffix: str) -> list[dict]:
    cache = rolling_cache_path(f"panelgp_{cache_suffix}")
    return rolling_panelgp_fit_cached(
        cache, data["panel"], data["context_raw"], data["char_cols"], CONTEXT_COLS,
        window_months=data["window_months"], forecast_years=data["forecast_years"],
        H=H, M=M, epochs=epochs, n_seeds=n_seeds, loading_kernel="rbf", factor_kernel="rbf",
        seed_base=config.RANDOM_SEED_BASE,
    )


def variance_table(fits: list[dict]) -> pd.DataFrame:
    """One row per (seed, month, stock) with the Eq. 26-27 decomposition and
    the realized squared forecast error e_it^2 = (r_i,t+1 - mu_hat_it)^2."""
    rows = []
    for year_fit in fits:
        for seed_idx, fitted in enumerate(year_fit["fitted"]):
            terms = panelgp_variance_terms(fitted, year_fit["panel_year"], year_fit["context"])
            for month in terms:
                for j, ticker in enumerate(month["tickers"]):
                    rows.append({
                        "date": month["date"], "ticker": ticker, "seed": seed_idx,
                        "V0": float(month["V0"][j]), "Vphi": float(month["Vphi"][j]),
                        "Vbeta": float(month["Vbeta"][j]), "trDbDp": float(month["trDbDp"][j]),
                        "sigma_r2": float(month["sigma_r2"][j]), "e2": float(month["e2"][j]),
                    })
    return pd.DataFrame(rows)


def calibration_correlations(df: pd.DataFrame) -> dict:
    """Spearman correlation of V_t^beta with the monthly average e^2 across
    months, and of V_it^phi with e_it^2 within month across stocks, averaged
    over seeds (Section 6.8)."""
    by_month = df.groupby(["seed", "date"]).agg(Vbeta=("Vbeta", "mean"), e2=("e2", "mean")).reset_index()
    rho_beta = by_month.groupby("seed").apply(
        lambda g: stats.spearmanr(g["Vbeta"], g["e2"]).statistic if len(g) > 2 else np.nan,
        include_groups=False,
    )
    within_month = []
    for (seed, date), g in df.groupby(["seed", "date"]):
        if len(g) > 2 and g["Vphi"].nunique() > 1:
            within_month.append(stats.spearmanr(g["Vphi"], g["e2"]).statistic)
    return {
        "spearman_Vbeta_vs_monthly_avg_e2_mean": float(np.nanmean(rho_beta)),
        "spearman_Vbeta_vs_monthly_avg_e2_by_seed": {int(k): float(v) for k, v in rho_beta.items()},
        "spearman_Vphi_vs_e2_within_month_mean": float(np.nanmean(within_month)) if within_month else None,
        "n_month_seed_pairs": int(len(within_month)),
    }


def quintile_calibration(values: np.ndarray, e2: np.ndarray, n_quintiles: int = 5) -> dict:
    """Section 6.8 / Eq. (36): group by quintile of the predicted-variance
    term and compare average predicted risk with realized RMS error (the
    number of quintiles is not specified in the paper; 5 is the conventional
    choice for this kind of sort)."""
    order = np.argsort(values)
    groups = np.array_split(order, n_quintiles)
    out = []
    for q, idx in enumerate(groups, start=1):
        out.append({
            "quintile": q,
            "mean_predicted_risk": float(np.sqrt(np.mean(values[idx]))),
            "realized_rms_error": float(np.sqrt(np.mean(e2[idx]))),
        })
    return out


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
    df = variance_table(fits)
    elapsed = time.time() - t0

    monthly_beta = df.groupby("date").agg(Vbeta=("Vbeta", "mean"), e2=("e2", "mean")).reset_index()
    temporal_quintiles = quintile_calibration(monthly_beta["Vbeta"].to_numpy(), monthly_beta["e2"].to_numpy())
    cross_sectional_quintiles = quintile_calibration(df["Vphi"].to_numpy(), df["e2"].to_numpy())
    correlations = calibration_correlations(df)

    out = {
        "config": {"H": H, "M": M, "epochs": epochs, "n_seeds": n_seeds, "full": args.full,
                    "forecast_years": data["forecast_years"], "n_rows": int(len(df))},
        "elapsed_seconds": elapsed,
        "calibration_correlations": correlations,
        "temporal_quintiles_Vbeta": temporal_quintiles,
        "cross_sectional_quintiles_Vphi": cross_sectional_quintiles,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp6_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    df.to_csv(RESULTS_DIR / "exp6_variance_terms.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, quint, title in [
        (axes[0], temporal_quintiles, "Temporal quintiles of V_t^beta"),
        (axes[1], cross_sectional_quintiles, "Cross-sectional quintiles of V_it^phi"),
    ]:
        q = [r["quintile"] for r in quint]
        ax.plot(q, [r["mean_predicted_risk"] for r in quint], marker="o", label="predicted risk (sqrt V)")
        ax.plot(q, [r["realized_rms_error"] for r in quint], marker="s", label="realized RMS error")
        ax.set_yscale("log")
        ax.set_xlabel("quintile")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Experiment 6: predicted variance vs. realized forecast-error dispersion (Figure 4)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_variance_calibration.png", dpi=120)

    print(f"[exp6] done in {elapsed:.1f}s, {len(df)} stock-month-seed rows")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
