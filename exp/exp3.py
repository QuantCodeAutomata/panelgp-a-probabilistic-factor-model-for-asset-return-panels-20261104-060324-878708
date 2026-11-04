"""Experiment 3: number of factors on the S&P 500 (paper Section 6.5 "Number
of factors", Figure 3, Appendix D.2 "Validation period and the number of
factors", Figure S2).

For each of the four PanelGP kernel pairs and for IPCA / IPCA+intercept,
sweeps the number of factors H and evaluates predictive R^2, rank
correlation, Gaussian log score, tangency Sharpe ratio and long-short Sharpe
ratio on both the validation period (used in the paper to pick H) and the
main evaluation period. Reuses the same data path and metrics as experiment 2
(src/sp500_pipeline.py), with a smaller universe so the larger H x kernel
grid fits the time budget (config.UNIVERSE_SIZE_RUN_SMALL, config.H_GRID_RUN).

Run: python -m exp.exp3 [--full]
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

from src import config
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
    return f"{int(train_start[:4]) - 3}-01-01"


def load_data(full: bool) -> dict:
    universe = config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN_SMALL]
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

    print(f"[exp3] universe={len(universe)} tickers, non-massive sources: {meta['non_massive_source']}")
    print(f"[exp3] train rows={len(panel_train)} months={len(dates)} valid rows={len(panel_valid)} eval rows={len(panel_eval)}")
    return {
        "panel_train": panel_train, "panel_valid": panel_valid, "panel_eval": panel_eval,
        "context": context, "char_cols": char_cols, "X": X, "t_idx": t_idx, "R": R, "C": C,
        "train_start": train_start, "train_end": train_end,
        "valid_start": valid_start, "valid_end": valid_end,
        "eval_start": eval_start, "eval_end": eval_end,
    }


def run_panelgp(data: dict, loading_kernel: str, factor_kernel: str, H: int, M: int, epochs: int, n_seeds: int) -> dict:
    valid_results, eval_results = [], []
    for s in range(n_seeds):
        seed = config.RANDOM_SEED_BASE + s
        fitted = fit_panelgp_variant(
            data["X"], data["t_idx"], data["R"], data["C"], H=H,
            loading_kernel=loading_kernel, factor_kernel=factor_kernel, M=M, epochs=epochs, seed=seed,
            char_cols=data["char_cols"], context_cols=CONTEXT_COLS,
            batch_size=config.BATCH_SIZE_PAPER, n_factor_samples=config.N_FACTOR_SAMPLES,
        )
        valid_results.append(evaluate_forecasts(panelgp_monthly_forecasts(fitted, data["panel_valid"], data["context"])))
        eval_results.append(evaluate_forecasts(panelgp_monthly_forecasts(fitted, data["panel_eval"], data["context"])))
    return {"validation": _aggregate(valid_results), "evaluation": _aggregate(eval_results)}


def run_ipca(data: dict, H: int, with_intercept: bool) -> dict:
    ipca_fit = fit_ipca_variant(data["panel_train"], data["char_cols"], H=H, with_intercept=with_intercept)
    valid = evaluate_forecasts(ipca_monthly_forecasts(ipca_fit, data["panel_valid"], data["char_cols"]))
    ev = evaluate_forecasts(ipca_monthly_forecasts(ipca_fit, data["panel_eval"], data["char_cols"]))
    return {"validation": _scalar(valid), "evaluation": _scalar(ev)}


_SCALAR_KEYS = ["predictive_r2_pct", "rank_correlation_pct", "log_score", "tangency_sharpe", "long_short_sharpe", "gmv_vol_pct"]


def _scalar(result: dict) -> dict:
    return {k: result[k] for k in _SCALAR_KEYS}


def _aggregate(seed_results: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in seed_results])) for k in _SCALAR_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's universe/period/epochs/seeds/H-grid instead of the reduced, time-budgeted configuration")
    args = parser.parse_args()

    M = config.M_INDUCING_PAPER if args.full else config.M_INDUCING_RUN
    epochs = config.EPOCHS_APPLICATION_PAPER if args.full else config.EPOCHS_APPLICATION_RUN
    n_seeds = config.N_SEEDS_PAPER if args.full else config.N_SEEDS_RUN
    H_grid = config.H_GRID_PAPER if args.full else config.H_GRID_RUN

    t0 = time.time()
    data = load_data(args.full)

    results = {name: {} for name in [f"{lk}/{fk}" for lk, fk in KERNEL_PAIRS] + ["IPCA", "IPCA+int"]}
    for H in H_grid:
        print(f"[exp3] H={H}")
        for loading_kernel, factor_kernel in KERNEL_PAIRS:
            name = f"{loading_kernel}/{factor_kernel}"
            r = run_panelgp(data, loading_kernel, factor_kernel, H, M, epochs, n_seeds)
            results[name][H] = r
            print(f"[exp3]   {name}: eval R2={r['evaluation']['predictive_r2_pct']:.3f}% tangencySR={r['evaluation']['tangency_sharpe']:.2f}")
        for name, with_intercept in [("IPCA", False), ("IPCA+int", True)]:
            results[name][H] = run_ipca(data, H, with_intercept)
            print(f"[exp3]   {name}: eval R2={results[name][H]['evaluation']['predictive_r2_pct']:.3f}%")

    elapsed = time.time() - t0
    out = {
        "config": {
            "universe_size": len(config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN_SMALL]),
            "train_start": data["train_start"], "train_end": data["train_end"],
            "valid_start": data["valid_start"], "valid_end": data["valid_end"],
            "eval_start": data["eval_start"], "eval_end": data["eval_end"],
            "H_grid": H_grid, "M": M, "epochs": epochs, "n_seeds": n_seeds, "full": args.full,
        },
        "elapsed_seconds": elapsed,
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("predictive_r2_pct", "Predictive R2 (%)"), ("rank_correlation_pct", "Rank correlation (%)"),
        ("log_score", "Gaussian log score"), ("tangency_sharpe", "Tangency Sharpe ratio"),
        ("long_short_sharpe", "Long-short Sharpe ratio"), ("gmv_vol_pct", "GMV volatility (%)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for name in results:
            ys = [results[name][H]["evaluation"][key] for H in H_grid]
            ax.plot(H_grid, ys, marker="o", label=name)
        ax.set_xlabel("H")
        ax.set_title(title)
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle("Experiment 3: number of factors, evaluation period (Figure 3)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_h_sweep.png", dpi=120)

    print(f"[exp3] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
