"""Shared data-to-evaluation pipeline for the S&P 500 application experiments
(exp_2-exp_6, paper Section 6). One implementation of the panel construction,
PanelGP/IPCA fitting, forecasting, and evaluation is used by every experiment
script so that kernel pairs, factor counts, and re-estimation schedules are
compared on exactly the same data and code path.

As noted in src/ipca.py and src/gp_model.py, neither the `ipca` package nor
standard sparse variational GP toolboxes (gpytorch/GPflow) cover the paper's
alternating-least-squares IPCA design or its bilinear two-input GP
likelihood, so both estimators are the paper's own equations, called here
through one shared data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.data_loader import load_ohlcv
from src.characteristics import (
    build_characteristics,
    build_context,
    monthly_returns_panel,
    stack_panel,
    standardize_context,
)
from src.evaluation import (
    annualized_sharpe,
    annualized_vol,
    gaussian_log_score,
    long_short_return,
    long_short_weights,
    monthly_rank_correlation,
    portfolio_weights,
    predictive_r2,
)
from src.gp_model import PanelGPConfig, PanelGPModel, fit_panelgp
from src.ipca import fit_ipca, ipca_forecast

CONTEXT_COLS = ["market_return", "log_realized_var", "avg_correlation"]


def load_universe_bars(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """OHLCV bars for the executed S&P 500 universe (see src/config.py
    SP500_CANDIDATE_TICKERS and results/RESULTS.md for the data-source
    deviation from the paper's WRDS/CRSP panel).

    Market data: QuantX Dataloader — https://quantx.limex.pro/

    Price history in this project is loaded through the QuantX Dataloader, a
    research-ready market-data service available through a REST API and as an
    MCP server for AI agents. It is built on market data from Massive
    (https://massive.com/) and prepares every stock history for backtesting:
      - prices adjusted for dividends, ready for total-return research;
      - one continuous history when a company changes its ticker;
      - clean separation when a ticker has belonged to different companies;
      - careful corporate-action handling and fine-grained data polishing.

    Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    MASSIVE_API_KEY at https://massive.com/.
    """
    return load_ohlcv(tickers, start, end, timeframe="1d", identity_date=end)


def build_stock_panel(daily_bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Long panel (date, ticker, <characteristics>, ret_next) and the raw
    (unstandardized) market-wide context series, both built from monthly bars
    derived from `daily_bars` (paper Section 6.1 standardization rules,
    substitute characteristics documented in src/characteristics.py)."""
    px = daily_bars.pivot_table(index="date", columns="ticker", values="close")
    px.index = pd.to_datetime(px.index)
    month_end_px = px.resample("ME").last()
    returns = monthly_returns_panel(daily_bars)
    volume_daily = daily_bars.pivot_table(index="date", columns="ticker", values="volume")
    volume_daily.index = pd.to_datetime(volume_daily.index)

    ranked = build_characteristics(month_end_px, returns, volume_daily)
    char_cols = sorted(ranked)
    panel = stack_panel(ranked, returns)
    context_raw = build_context(returns)
    return panel, context_raw, char_cols


def standardize_context_for_training(context_raw: pd.DataFrame, train_start: str, train_end: str) -> pd.DataFrame:
    """Zero-mean/unit-variance context standardization using only the training
    period's moments (Section 6.1), applied to the whole loaded window so the
    same transform covers validation and evaluation months."""
    train_mask = (context_raw.index >= pd.Timestamp(train_start)) & (context_raw.index <= pd.Timestamp(train_end))
    return standardize_context(context_raw, pd.Series(train_mask, index=context_raw.index))


def slice_period(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows with characteristic date t in [start, end] and a realized next
    return (drops the final, not-yet-realized month of the loaded window)."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mask = (panel["date"] >= start_ts) & (panel["date"] <= end_ts)
    return panel.loc[mask].dropna(subset=["ret_next"]).reset_index(drop=True)


def to_training_tensors(
    panel_train: pd.DataFrame, context: pd.DataFrame, char_cols: list[str], context_cols: list[str] = CONTEXT_COLS
):
    """Row-level (X, t_idx, R) tensors plus the (T, Q) unique-date context
    matrix C_train used both for PanelGP's prior and for later forecasting
    conditionals (Section 6.1: "each forecast uses current characteristics
    and context")."""
    dates = sorted(panel_train["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    X = panel_train[char_cols].to_numpy(dtype=np.float32)
    t_idx = panel_train["date"].map(date_to_idx).to_numpy(dtype=np.int64)
    R = panel_train["ret_next"].to_numpy(dtype=np.float32)
    ctx = context.reindex(dates)[context_cols]
    if ctx.isna().any().any():
        raise ValueError("context has missing values over the training window; extend the warm-up period")
    C = ctx.to_numpy(dtype=np.float32)
    return X, t_idx, R, C, dates


@dataclass
class FittedPanelGP:
    model: PanelGPModel
    s_r: float
    C_train_tensor: torch.Tensor
    char_cols: list[str]
    context_cols: list[str]


def fit_panelgp_variant(
    X: np.ndarray,
    t_idx: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    H: int,
    loading_kernel: str,
    factor_kernel: str,
    M: int,
    epochs: int,
    seed: int,
    char_cols: list[str],
    context_cols: list[str] = CONTEXT_COLS,
    lr_variational: float = 0.01,
    lr_model: float = 0.005,
    batch_size: int = 2048,
    n_factor_samples: int = 64,
    grad_clip: float = 1.0,
    lr_halving_fractions=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
    verbose: bool = False,
) -> FittedPanelGP:
    """Estimate one PanelGP kernel-pair variant on a fixed training window
    (Section 6.1: returns scaled by s_r=sqrt(mean(r_train^2)) without
    centering before optimization)."""
    s_r = float(np.sqrt(np.mean(R ** 2)))
    R_scaled = R / s_r
    T = C.shape[0]
    X_t = torch.tensor(X, dtype=torch.float32)
    t_idx_t = torch.tensor(t_idx, dtype=torch.long)
    R_t = torch.tensor(R_scaled, dtype=torch.float32)
    C_t = torch.tensor(C, dtype=torch.float32)

    cfg = PanelGPConfig(
        P=X.shape[1], Q=C.shape[1], H=H, T=T, M=M,
        loading_kernel=loading_kernel, factor_kernel=factor_kernel,
        isotropic_innovations=False,
    )
    model = PanelGPModel(cfg, seed=seed)
    # Equation C.6: d = 1 for the RBF loading kernel, d = 1 + mean||x||^2 for the affine kernel.
    d_loading = 1.0 if loading_kernel == "rbf" else 1.0 + float(np.mean(np.sum(X ** 2, axis=1)))
    model.set_initial_values(d_loading=d_loading)

    fit_panelgp(
        model, X_t, t_idx_t, R_t, C_t, epochs=epochs,
        batch_size=min(batch_size, X.shape[0]), lr_variational=lr_variational, lr_model=lr_model,
        n_factor_samples=n_factor_samples, grad_clip=grad_clip,
        lr_halving_fractions=lr_halving_fractions, seed=seed, verbose=verbose,
    )
    return FittedPanelGP(model=model, s_r=s_r, C_train_tensor=C_t, char_cols=char_cols, context_cols=context_cols)


def panelgp_monthly_forecasts(fitted: FittedPanelGP, panel_period: pd.DataFrame, context: pd.DataFrame):
    """Per-month (mu, Sigma, r_true, tickers) forecasts in original return
    units for every date present in `panel_period` (Eq. 24-27)."""
    out = []
    for date, group in panel_period.groupby("date", sort=True):
        c_row = context.loc[date, fitted.context_cols].to_numpy(dtype=np.float32)
        if np.isnan(c_row).any():
            continue
        X_new = torch.tensor(group[fitted.char_cols].to_numpy(dtype=np.float32))
        c_new = torch.tensor(c_row)
        mu_s, Sigma_s = fitted.model.cross_section_predictive(X_new, c_new, fitted.C_train_tensor)
        mu = mu_s.numpy() * fitted.s_r
        Sigma = Sigma_s.numpy() * (fitted.s_r ** 2)
        r_true = group["ret_next"].to_numpy(dtype=np.float64)
        out.append({"date": date, "tickers": group["ticker"].tolist(), "mu": mu, "Sigma": Sigma, "r_true": r_true})
    return out


def panelgp_variance_terms(fitted: FittedPanelGP, panel_period: pd.DataFrame, context: pd.DataFrame):
    """Per-month variance decomposition V^0, V^phi, V^beta, tr(D_beta D_phi),
    sigma_r^2 in original return units (Eq. 26-27, used by exp_6)."""
    out = []
    for date, group in panel_period.groupby("date", sort=True):
        c_row = context.loc[date, fitted.context_cols].to_numpy(dtype=np.float32)
        if np.isnan(c_row).any():
            continue
        X_new = torch.tensor(group[fitted.char_cols].to_numpy(dtype=np.float32))
        c_new = torch.tensor(c_row)
        V0, Vphi, Vbeta, trDbDp, sigma_r2 = fitted.model.variance_decomposition(X_new, c_new, fitted.C_train_tensor)
        scale = fitted.s_r ** 2
        out.append({
            "date": date, "tickers": group["ticker"].tolist(),
            "V0": V0.numpy() * scale, "Vphi": Vphi.numpy() * scale,
            "Vbeta": Vbeta.numpy() * scale, "trDbDp": trDbDp.numpy() * scale,
            "sigma_r2": sigma_r2.numpy() * scale,
            "e2": (group["ret_next"].to_numpy(dtype=np.float64)) ** 2,
        })
    return out


def fit_ipca_variant(panel_train: pd.DataFrame, char_cols: list[str], H: int, with_intercept: bool, seed: int = 0) -> dict:
    """Section 6.2: Gamma (and gamma_alpha) by alternating least squares on
    unscaled excess returns, one cross-section per training month."""
    X_by_month, r_by_month = [], []
    for _, group in panel_train.groupby("date", sort=True):
        X_by_month.append(group[char_cols].to_numpy(dtype=np.float64))
        r_by_month.append(group["ret_next"].to_numpy(dtype=np.float64))
    return fit_ipca(X_by_month, r_by_month, H=H, with_intercept=with_intercept, seed=seed)


def ipca_monthly_forecasts(ipca_fit: dict, panel_period: pd.DataFrame, char_cols: list[str]):
    out = []
    for date, group in panel_period.groupby("date", sort=True):
        X_t = group[char_cols].to_numpy(dtype=np.float64)
        mu, Sigma = ipca_forecast(ipca_fit, X_t)
        r_true = group["ret_next"].to_numpy(dtype=np.float64)
        out.append({"date": date, "tickers": group["ticker"].tolist(), "mu": mu, "Sigma": Sigma, "r_true": r_true})
    return out


def evaluate_forecasts(forecasts: list[dict]) -> dict:
    """Predictive R^2, monthly rank correlation, Gaussian log score, and GMV
    / tangency / long-short portfolio performance pooled over `forecasts`
    (Eq. 30-32, Section 6.3)."""
    mu_list = [f["mu"] for f in forecasts]
    r_list = [f["r_true"] for f in forecasts]
    r2 = predictive_r2(mu_list, r_list)
    ranks = monthly_rank_correlation(mu_list, r_list)
    log_scores, tangency_rets, gmv_rets, ls_rets = [], [], [], []
    for f in forecasts:
        mu, Sigma, r = f["mu"], f["Sigma"], f["r_true"]
        log_scores.append(gaussian_log_score(mu, Sigma, r))
        weights = portfolio_weights(mu, Sigma)
        tangency_rets.append(float(weights["tangency"] @ r))
        gmv_rets.append(float(weights["gmv"] @ r))
        ls_rets.append(long_short_return(mu, r))
    tangency_rets, gmv_rets, ls_rets = map(np.array, (tangency_rets, gmv_rets, ls_rets))
    return {
        "predictive_r2_pct": 100.0 * r2,
        "rank_correlation_pct": 100.0 * float(np.nanmean(ranks)),
        "log_score": float(np.mean(log_scores)),
        "tangency_sharpe": annualized_sharpe(tangency_rets),
        "long_short_sharpe": annualized_sharpe(ls_rets),
        "gmv_vol_pct": 100.0 * annualized_vol(gmv_rets),
        "tangency_returns": tangency_rets,
        "gmv_returns": gmv_rets,
        "long_short_returns": ls_rets,
        "gmv_weights_by_month": [portfolio_weights(f["mu"], f["Sigma"])["gmv"] for f in forecasts],
        "tangency_weights_by_month": [portfolio_weights(f["mu"], f["Sigma"])["tangency"] for f in forecasts],
        "long_short_weights_by_month": [long_short_weights(f["mu"]) for f in forecasts],
        "tickers_by_month": [f["tickers"] for f in forecasts],
        "r_true_by_month": [f["r_true"] for f in forecasts],
        "dates": [f["date"] for f in forecasts],
    }


# ---------------------------------------------------------------------------
# Annual rolling re-estimation (paper Section 6.6, used by exp_4-exp_6)
# ---------------------------------------------------------------------------
ROLLING_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def rolling_cache_path(tag: str) -> Path:
    ROLLING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ROLLING_CACHE_DIR / f"rolling_{tag}.pt"


def rolling_panelgp_fit(
    panel: pd.DataFrame,
    context_raw: pd.DataFrame,
    char_cols: list[str],
    context_cols: list[str],
    window_months: int,
    forecast_years: list[int],
    H: int,
    M: int,
    epochs: int,
    n_seeds: int,
    loading_kernel: str = "rbf",
    factor_kernel: str = "rbf",
    seed_base: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Annual re-estimation on the trailing `window_months` months (Section
    6.6: "first window ... forecasts 1995; last window forecasts 2024").
    Returns, for every forecast year, the list of per-seed fitted models, the
    panel rows forecast with them, and the context standardized on that
    window's own training months (each window gets "own initialization,
    context standardization, return scale", implementation_notes)."""
    dates_all = sorted(panel["date"].unique())
    results = []
    for year in forecast_years:
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        train_dates = [d for d in dates_all if d <= train_end]
        if len(train_dates) < window_months:
            continue
        train_dates = train_dates[-window_months:]
        train_start = train_dates[0]
        panel_train = panel[(panel["date"] >= train_start) & (panel["date"] <= train_end)].dropna(subset=["ret_next"])
        context_std = standardize_context_for_training(context_raw, str(train_start.date()), str(train_end.date()))
        X, t_idx, R, C, _ = to_training_tensors(panel_train, context_std, char_cols, context_cols)
        panel_year = slice_period(panel, f"{year}-01-01", f"{year}-12-31")
        fits = []
        for s in range(n_seeds):
            seed = seed_base + s
            fitted = fit_panelgp_variant(
                X, t_idx, R, C, H=H, loading_kernel=loading_kernel, factor_kernel=factor_kernel,
                M=M, epochs=epochs, seed=seed, char_cols=char_cols, context_cols=context_cols, verbose=verbose,
            )
            fits.append(fitted)
        results.append({"year": year, "fitted": fits, "panel_year": panel_year, "context": context_std,
                         "train_start": train_start, "train_end": train_end})
        print(f"[rolling] year={year} trained on {train_start.date()}..{train_end.date()}, "
              f"{len(panel_train)} rows, {len(panel_year)} forecast rows")
    return results


def rolling_panelgp_fit_cached(cache_path: Path | None, *args, **kwargs) -> list[dict]:
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path, weights_only=False)
    results = rolling_panelgp_fit(*args, **kwargs)
    if cache_path is not None:
        torch.save(results, cache_path)
    return results


def rolling_ipca_fit(
    panel: pd.DataFrame,
    char_cols: list[str],
    window_months: int,
    forecast_years: list[int],
    H: int,
    with_intercept: bool,
) -> list[dict]:
    """Annual re-estimation of IPCA by alternating least squares on the same
    trailing-window schedule as `rolling_panelgp_fit` (Section 6.6)."""
    dates_all = sorted(panel["date"].unique())
    results = []
    for year in forecast_years:
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        train_dates = [d for d in dates_all if d <= train_end]
        if len(train_dates) < window_months:
            continue
        train_dates = train_dates[-window_months:]
        train_start = train_dates[0]
        panel_train = panel[(panel["date"] >= train_start) & (panel["date"] <= train_end)].dropna(subset=["ret_next"])
        ipca_fit = fit_ipca_variant(panel_train, char_cols, H=H, with_intercept=with_intercept)
        panel_year = slice_period(panel, f"{year}-01-01", f"{year}-12-31")
        results.append({"year": year, "ipca_fit": ipca_fit, "panel_year": panel_year})
    return results
