"""Experiment 4: annual re-estimation and trading costs on the S&P 500
(paper Section 6.6 "Annual re-estimation", Table 3, Table 4, Appendix D.3
Table S2).

Compares the fixed-training RBF/RBF PanelGP (H=5) of experiment 2 with an
annually re-estimated ("rolling") version, and with rolling IPCA / IPCA+
intercept, then adds proportional trading costs (Equation 33-34).

Run: python -m exp.exp4 [--full]
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
from src.evaluation import (
    annualized_sharpe,
    annualized_vol,
    break_even_cost_bps,
    holm_adjust,
    net_return,
    newey_west_statistic,
    turnover,
)
from src.sp500_pipeline import (
    CONTEXT_COLS,
    build_stock_panel,
    evaluate_forecasts,
    fit_ipca_variant,
    fit_panelgp_variant,
    ipca_monthly_forecasts,
    load_universe_bars,
    panelgp_monthly_forecasts,
    rolling_cache_path,
    rolling_ipca_fit,
    rolling_panelgp_fit_cached,
    slice_period,
    standardize_context_for_training,
    to_training_tensors,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RISK_FREE_RATE = 0.0  # monthly r_f; the paper does not specify the series used for Eq. (33), see RESULTS.md


def _warmup_start(train_start: str) -> str:
    year = int(train_start[:4]) - 3
    return f"{year}-01-01"


def load_data(full: bool):
    universe = config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN]
    train_start = config.SP500_TRAIN_START_PAPER if full else config.SP500_TRAIN_START_RUN
    train_end = config.SP500_TRAIN_END_PAPER if full else config.SP500_TRAIN_END_RUN
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
    print(f"[exp4] universe={len(universe)} tickers, non-massive sources: {meta['non_massive_source']}")
    print(f"[exp4] panel rows={len(panel)} forecast_years={forecast_years} window_months={window_months}")
    return {
        "panel": panel, "context_raw": context_raw, "char_cols": char_cols,
        "train_start": train_start, "train_end": train_end,
        "eval_start": eval_start, "eval_end": eval_end,
        "window_months": window_months, "forecast_years": forecast_years,
    }


def fixed_panelgp(data: dict, H: int, M: int, epochs: int, n_seeds: int):
    """Re-estimate the fixed 1960-1989 (paper) / executed training-window
    RBF/RBF PanelGP of experiment 2, evaluated on the same forecast years as
    the rolling model so the two series are directly comparable."""
    panel, context_raw, char_cols = data["panel"], data["context_raw"], data["char_cols"]
    context = standardize_context_for_training(context_raw, data["train_start"], data["train_end"])
    panel_train = slice_period(panel, data["train_start"], data["train_end"])
    X, t_idx, R, C, _ = to_training_tensors(panel_train, context, char_cols)
    panel_eval = panel[panel["date"].dt.year.isin(data["forecast_years"])].dropna(subset=["ret_next"])

    seed_results = []
    for s in range(n_seeds):
        seed = config.RANDOM_SEED_BASE + s
        fitted = fit_panelgp_variant(
            X, t_idx, R, C, H=H, loading_kernel="rbf", factor_kernel="rbf", M=M, epochs=epochs,
            seed=seed, char_cols=char_cols, context_cols=CONTEXT_COLS,
        )
        forecasts = panelgp_monthly_forecasts(fitted, panel_eval, context)
        seed_results.append(evaluate_forecasts(forecasts))
    return _aggregate_seeds(seed_results)


def _aggregate_seeds(seed_results: list[dict]) -> dict:
    scalar_keys = ["predictive_r2_pct", "rank_correlation_pct", "log_score", "tangency_sharpe", "long_short_sharpe", "gmv_vol_pct"]
    agg = {k: float(np.mean([r[k] for r in seed_results])) for k in scalar_keys}
    for key in ["tangency_returns", "long_short_returns", "gmv_returns"]:
        agg[key] = np.mean([r[key] for r in seed_results], axis=0)
    agg["gmv_weights_by_month"] = seed_results[0]["gmv_weights_by_month"]
    agg["tangency_weights_by_month"] = seed_results[0]["tangency_weights_by_month"]
    agg["long_short_weights_by_month"] = seed_results[0]["long_short_weights_by_month"]
    agg["tickers_by_month"] = seed_results[0]["tickers_by_month"]
    agg["r_true_by_month"] = seed_results[0]["r_true_by_month"]
    agg["dates"] = seed_results[0]["dates"]
    return agg


def rolling_panelgp(data: dict, H: int, M: int, epochs: int, n_seeds: int, cache_tag: str):
    cache = rolling_cache_path(cache_tag)
    fits = rolling_panelgp_fit_cached(
        cache, data["panel"], data["context_raw"], data["char_cols"], CONTEXT_COLS,
        window_months=data["window_months"], forecast_years=data["forecast_years"],
        H=H, M=M, epochs=epochs, n_seeds=n_seeds, loading_kernel="rbf", factor_kernel="rbf",
        seed_base=config.RANDOM_SEED_BASE,
    )
    per_seed_forecasts = [[] for _ in range(n_seeds)]
    for year_fit in fits:
        for s, fitted in enumerate(year_fit["fitted"]):
            per_seed_forecasts[s].extend(
                panelgp_monthly_forecasts(fitted, year_fit["panel_year"], year_fit["context"])
            )
    seed_results = [evaluate_forecasts(f) for f in per_seed_forecasts]
    return _aggregate_seeds(seed_results), fits


def rolling_ipca(data: dict, H: int, with_intercept: bool):
    fits = rolling_ipca_fit(
        data["panel"], data["char_cols"], window_months=data["window_months"],
        forecast_years=data["forecast_years"], H=H, with_intercept=with_intercept,
    )
    forecasts = []
    for year_fit in fits:
        forecasts.extend(ipca_monthly_forecasts(year_fit["ipca_fit"], year_fit["panel_year"], data["char_cols"]))
    return evaluate_forecasts(forecasts)


def trading_costs(result: dict, weight_key: str, panel: "pd.DataFrame") -> dict:
    """Equation (33)-(34): turnover from drifting last month's weights forward
    with realized single-stock returns, then net-of-cost return and
    break-even cost. Needs month-to-month ticker continuity; months where the
    ticker set changes contribute full turnover for the overlapping names
    only (Section 6.6 does not specify the treatment of entries/exits)."""
    weights_by_month = result[weight_key]
    tickers_by_month = result["tickers_by_month"]
    r_true_by_month = result["r_true_by_month"]
    return_key = "tangency_returns" if weight_key == "tangency_weights_by_month" else "long_short_returns"
    gross_returns = result[return_key]
    turnovers, net_returns = [], []
    for m in range(1, len(weights_by_month)):
        prev_tickers, curr_tickers = tickers_by_month[m - 1], tickers_by_month[m]
        common = [t for t in curr_tickers if t in prev_tickers]
        if not common:
            turnovers.append(0.0)
            net_returns.append(float(np.dot(weights_by_month[m], r_true_by_month[m])))
            continue
        idx_prev = [prev_tickers.index(t) for t in common]
        idx_curr = [curr_tickers.index(t) for t in common]
        w_prev = np.array([weights_by_month[m - 1][i] for i in idx_prev])
        w_curr_common = np.array([weights_by_month[m][i] for i in idx_curr])
        r_common = np.array([r_true_by_month[m - 1][i] for i in idx_prev])
        to = turnover(w_prev, w_curr_common, r_common, RISK_FREE_RATE)
        to += float(np.sum(np.abs(weights_by_month[m]))) - float(np.sum(np.abs(w_curr_common)))  # new names enter at full weight
        turnovers.append(to)
        net_returns.append(net_return(weights_by_month[m], r_true_by_month[m], to, config.TRADING_COST_BPS))
    turnovers, net_returns = np.array(turnovers), np.array(net_returns)
    gross_excess = gross_returns[1:] - RISK_FREE_RATE
    return {
        "mean_turnover_pct": 100.0 * float(np.mean(turnovers)),
        "net_sharpe": annualized_sharpe(net_returns),
        "break_even_cost_bps": break_even_cost_bps(gross_excess, turnovers),
    }


def pairwise_tests(results: dict, metric_returns_key: str, lower_is_better: bool = False) -> list[dict]:
    names = list(results)
    pairs = list(itertools.combinations(names, 2))
    tstats, pvals = [], []
    for a, b in pairs:
        n = min(len(results[a][metric_returns_key]), len(results[b][metric_returns_key]))
        diff = results[a][metric_returns_key][-n:] - results[b][metric_returns_key][-n:]
        if lower_is_better:
            diff = -diff
        t_stat, p_val = newey_west_statistic(diff, lags=config.NEWEY_WEST_LAGS)
        tstats.append(t_stat)
        pvals.append(p_val)
    adj = holm_adjust(pvals)
    return [{"pair": f"{a} vs {b}", "t_stat": t, "p_value": p, "p_value_holm": pa}
            for (a, b), t, p, pa in zip(pairs, tstats, pvals, adj)]


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
    cache_suffix = "full" if args.full else "run"

    print("[exp4] fitting fixed-training RBF/RBF PanelGP")
    results = {"PanelGP fixed": fixed_panelgp(data, H, M, epochs, n_seeds)}

    print("[exp4] rolling RBF/RBF PanelGP (annual re-estimation)")
    results["PanelGP rolling"], rolling_fits = rolling_panelgp(data, H, M, epochs, n_seeds, cache_tag=f"panelgp_{cache_suffix}")

    for name, with_intercept in [("IPCA rolling", False), ("IPCA+int rolling", True)]:
        print(f"[exp4] fitting {name}")
        results[name] = rolling_ipca(data, H, with_intercept)

    table = {name: {k: v for k, v in r.items() if k in
                     ["predictive_r2_pct", "rank_correlation_pct", "log_score", "tangency_sharpe", "long_short_sharpe", "gmv_vol_pct"]}
             for name, r in results.items()}

    costs = {
        "PanelGP rolling": {
            "tangency": trading_costs(results["PanelGP rolling"], "tangency_weights_by_month", data["panel"]),
            "long_short": trading_costs(results["PanelGP rolling"], "long_short_weights_by_month", data["panel"]),
        },
        "IPCA+int rolling": {
            "tangency": trading_costs(results["IPCA+int rolling"], "tangency_weights_by_month", data["panel"]),
            "long_short": trading_costs(results["IPCA+int rolling"], "long_short_weights_by_month", data["panel"]),
        },
    }

    pairwise = {
        "tangency_sharpe": pairwise_tests(results, "tangency_returns"),
        "long_short_sharpe": pairwise_tests(results, "long_short_returns"),
        "gmv_vol": pairwise_tests(results, "gmv_returns", lower_is_better=True),
    }

    elapsed = time.time() - t0
    out = {
        "config": {
            "universe_size": len(config.SP500_CANDIDATE_TICKERS[: config.UNIVERSE_SIZE_RUN]),
            "window_months": data["window_months"], "forecast_years": data["forecast_years"],
            "H": H, "M": M, "epochs": epochs, "n_seeds": n_seeds, "full": args.full,
            "risk_free_rate_assumption": RISK_FREE_RATE,
        },
        "elapsed_seconds": elapsed,
        "metrics": table,
        "trading_costs": costs,
        "pairwise_newey_west": pairwise,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp4_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(table)
    axes[0].bar(names, [table[n]["tangency_sharpe"] for n in names])
    axes[0].set_title("Tangency Sharpe ratio")
    axes[1].bar(names, [table[n]["gmv_vol_pct"] for n in names])
    axes[1].set_title("GMV volatility (%)")
    axes[2].bar(names, [table[n]["log_score"] for n in names])
    axes[2].set_title("Gaussian log score")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Experiment 4: fixed vs. annual re-estimation (Table 3)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_fixed_vs_rolling.png", dpi=120)

    print(f"[exp4] done in {elapsed:.1f}s")
    print(json.dumps(table, indent=2))
    print(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main()
