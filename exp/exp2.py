"""Experiment 2: kernel comparison on the S&P 500 with fixed training
(paper Section 6.1 "Data and estimation design", Section 6.4 "Kernel
comparison", Table 2, Appendix D.3 Table S1).

Four PanelGP kernel pairs (affine/constant, affine/RBF, RBF/constant,
RBF/RBF) and two IPCA benchmarks (no intercept, with intercept) are all
estimated once on a fixed training window and evaluated out-of-sample on a
later period, using the data path and metrics shared with every other S&P
500 experiment in this repository (src/sp500_pipeline.py).

Run: python -m exp.exp2 [--full]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.evaluation import holm_adjust, newey_west_statistic
from src.sp500_pipeline import (
    CONTEXT_COLS,
    build_stock_panel,
    evaluate_forecasts,
    fit_ipca_variant,
    fit_panelgp_variant,
    ipca_monthly_forecasts,
    load_universe_bars,
    panelgp_monthly_forecasts,
    slice_period,
    standardize_context_for_training,
    to_training_tensors,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
KERNEL_PAIRS = [("affine", "constant"), ("affine", "rbf"), ("rbf", "constant"), ("rbf", "rbf")]


def _warmup_start(train_start: str) -> str:
    year = int(train_start[:4]) - 3
    return f"{year}-01-01"


def load_data(full: bool):
    universe = config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN]
    train_start = config.SP500_TRAIN_START_PAPER if full else config.SP500_TRAIN_START_RUN
    train_end = config.SP500_TRAIN_END_PAPER if full else config.SP500_TRAIN_END_RUN
    valid_start = config.SP500_VALID_START if full else config.SP500_VALID_START_RUN
    valid_end = config.SP500_VALID_END if full else config.SP500_VALID_END_RUN
    eval_start = config.SP500_EVAL_START if full else config.SP500_EVAL_START_RUN
    eval_end = config.SP500_EVAL_END if full else config.SP500_EVAL_END_RUN

    bars, meta = load_universe_bars(universe, _warmup_start(train_start), eval_end)
    panel, context_raw, char_cols = build_stock_panel(bars)
    context = standardize_context_for_training(context_raw, train_start, train_end)

    panel_train = slice_period(panel, train_start, train_end)
    panel_valid = slice_period(panel, valid_start, valid_end)
    panel_eval = slice_period(panel, eval_start, eval_end)
    X, t_idx, R, C, dates = to_training_tensors(panel_train, context, char_cols)

    print(f"[exp2] universe={len(universe)} tickers, non-massive sources: {meta['non_massive_source']}")
    print(f"[exp2] train rows={len(panel_train)} months={len(dates)} valid rows={len(panel_valid)} eval rows={len(panel_eval)}")
    return {
        "panel_train": panel_train, "panel_valid": panel_valid, "panel_eval": panel_eval,
        "context": context, "char_cols": char_cols, "X": X, "t_idx": t_idx, "R": R, "C": C,
        "train_start": train_start, "train_end": train_end, "eval_start": eval_start, "eval_end": eval_end,
    }


def run_panelgp_variant(data: dict, loading_kernel: str, factor_kernel: str, H: int, M: int, epochs: int, n_seeds: int):
    seed_results = []
    for s in range(n_seeds):
        seed = config.RANDOM_SEED_BASE + s
        fitted = fit_panelgp_variant(
            data["X"], data["t_idx"], data["R"], data["C"], H=H,
            loading_kernel=loading_kernel, factor_kernel=factor_kernel, M=M, epochs=epochs, seed=seed,
            char_cols=data["char_cols"], context_cols=CONTEXT_COLS,
            batch_size=config.BATCH_SIZE_PAPER, n_factor_samples=config.N_FACTOR_SAMPLES,
        )
        forecasts = panelgp_monthly_forecasts(fitted, data["panel_eval"], data["context"])
        seed_results.append(evaluate_forecasts(forecasts))
    return _aggregate_seeds(seed_results)


def _aggregate_seeds(seed_results: list[dict]) -> dict:
    scalar_keys = ["predictive_r2_pct", "rank_correlation_pct", "log_score", "tangency_sharpe", "long_short_sharpe", "gmv_vol_pct"]
    agg = {k: float(np.mean([r[k] for r in seed_results])) for k in scalar_keys}
    agg["tangency_returns"] = np.mean([r["tangency_returns"] for r in seed_results], axis=0)
    agg["long_short_returns"] = np.mean([r["long_short_returns"] for r in seed_results], axis=0)
    agg["gmv_returns"] = np.mean([r["gmv_returns"] for r in seed_results], axis=0)
    return agg


def run_ipca_variants(data: dict, H: int, with_intercept: bool):
    ipca_fit = fit_ipca_variant(data["panel_train"], data["char_cols"], H=H, with_intercept=with_intercept)
    forecasts = ipca_monthly_forecasts(ipca_fit, data["panel_eval"], data["char_cols"])
    return evaluate_forecasts(forecasts)


def pairwise_tests(results: dict, metric_returns_key: str, lower_is_better: bool = False) -> list[dict]:
    """Holm-adjusted Newey-West tests for every pair of models on one
    monthly-return series (Appendix D.3)."""
    names = list(results)
    pairs = list(itertools.combinations(names, 2))
    stats, pvals = [], []
    for a, b in pairs:
        diff = results[a][metric_returns_key] - results[b][metric_returns_key]
        if lower_is_better:
            diff = -diff
        t_stat, p_val = newey_west_statistic(diff, lags=config.NEWEY_WEST_LAGS)
        stats.append(t_stat)
        pvals.append(p_val)
    adj = holm_adjust(pvals)
    return [{"pair": f"{a} vs {b}", "t_stat": t, "p_value": p, "p_value_holm": pa}
            for (a, b), t, p, pa in zip(pairs, stats, pvals, adj)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's universe/period/epochs/seeds instead of the reduced, time-budgeted configuration")
    args = parser.parse_args()

    H = config.H_FACTORS_PAPER
    M = config.M_INDUCING_PAPER if args.full else config.M_INDUCING_RUN
    epochs = config.EPOCHS_APPLICATION_PAPER if args.full else config.EPOCHS_APPLICATION_RUN
    n_seeds = config.N_SEEDS_PAPER if args.full else config.N_SEEDS_RUN

    t0 = time.time()
    data = load_data(args.full)

    results = {}
    for loading_kernel, factor_kernel in KERNEL_PAIRS:
        name = f"{loading_kernel}/{factor_kernel}"
        print(f"[exp2] fitting PanelGP {name}, H={H}, M={M}, epochs={epochs}, seeds={n_seeds}")
        results[name] = run_panelgp_variant(data, loading_kernel, factor_kernel, H, M, epochs, n_seeds)
        print(f"[exp2]   R2={results[name]['predictive_r2_pct']:.3f}% rank={results[name]['rank_correlation_pct']:.2f}% "
              f"logscore={results[name]['log_score']:.3f} tangencySR={results[name]['tangency_sharpe']:.2f}")

    for name, with_intercept in [("IPCA", False), ("IPCA+int", True)]:
        print(f"[exp2] fitting {name}")
        results[name] = run_ipca_variants(data, H, with_intercept)
        print(f"[exp2]   R2={results[name]['predictive_r2_pct']:.3f}% rank={results[name]['rank_correlation_pct']:.2f}%")

    pairwise = {
        "predictive_r2": None,  # R2 is a pooled statistic, not a monthly series; excluded from NW tests as in Appendix D.3
        "tangency_sharpe": pairwise_tests(results, "tangency_returns"),
        "long_short_sharpe": pairwise_tests(results, "long_short_returns"),
        "gmv_vol": pairwise_tests(results, "gmv_returns", lower_is_better=True),
    }

    elapsed = time.time() - t0
    table = {name: {k: v for k, v in r.items() if k in
                     ["predictive_r2_pct", "rank_correlation_pct", "log_score", "tangency_sharpe", "long_short_sharpe", "gmv_vol_pct"]}
             for name, r in results.items()}

    out = {
        "config": {
            "universe_size": len(config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN]),
            "train_start": data["train_start"], "train_end": data["train_end"],
            "eval_start": data["eval_start"], "eval_end": data["eval_end"],
            "H": H, "M": M, "epochs": epochs, "n_seeds": n_seeds, "full": args.full,
        },
        "elapsed_seconds": elapsed,
        "metrics": table,
        "pairwise_newey_west": pairwise,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(table)
    axes[0].bar(names, [table[n]["rank_correlation_pct"] for n in names])
    axes[0].set_title("Rank correlation (%)")
    axes[1].bar(names, [table[n]["tangency_sharpe"] for n in names])
    axes[1].set_title("Tangency Sharpe ratio")
    axes[2].bar(names, [table[n]["gmv_vol_pct"] for n in names])
    axes[2].set_title("GMV volatility (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Experiment 2: kernel comparison, fixed training (Table 2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_kernel_comparison.png", dpi=120)

    print(f"[exp2] done in {elapsed:.1f}s")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
