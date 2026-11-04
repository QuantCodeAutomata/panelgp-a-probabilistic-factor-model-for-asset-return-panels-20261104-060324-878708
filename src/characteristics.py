"""Price-based stock characteristics and market-wide context built from
Dataloader OHLCV bars, used in place of the paper's 137 WRDS characteristics
from Jensen et al. (2023) and six Goyal et al. (2024) context variables
(paper Section 6.1), which are not obtainable through the QuantX Dataloader
or Massive REST APIs (see results/RESULTS.md, "Deviations from the paper").

Custom implementation: characteristic cross-sectional ranking to [-0.5, 0.5]
and context standardization follow the paper's Section 6.1/Appendix C.1
description exactly; the characteristic *definitions* themselves are
standard price/volume factors (momentum, volatility, size, liquidity), not a
library preset, so they are computed directly from monthly bars with `ta`
building blocks where applicable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_returns_panel(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Resample daily close prices to month-end simple returns per ticker.

    Args:
        daily_bars: tidy frame with columns date, ticker, close (data/data_loader.py).
    Returns:
        DataFrame indexed by month-end date, one return column per ticker.
    """
    px = daily_bars.pivot_table(index="date", columns="ticker", values="close")
    px.index = pd.to_datetime(px.index)
    month_end_px = px.resample("ME").last()
    returns = month_end_px.pct_change()
    return returns


def _cross_sectional_rank(row: pd.Series) -> pd.Series:
    """Rank a month's cross-section to [-0.5, 0.5] (paper Section 6.1/Appendix C.1);
    characteristics with no observation that month are set to zero."""
    valid = row.dropna()
    if len(valid) < 2:
        return row.fillna(0.0) * 0.0
    ranks = valid.rank(method="average")
    scaled = (ranks - 1) / (len(valid) - 1) - 0.5
    out = pd.Series(0.0, index=row.index)
    out.loc[valid.index] = scaled
    return out


def build_characteristics(month_end_px: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Twelve price/volume-based characteristics per (month, ticker), ranked
    cross-sectionally to [-0.5, 0.5] each month with missing values at zero,
    following the paper's standardization rule (Section 6.1).

    Characteristics: 1/3/6/12-month momentum, 1/12-month realized volatility,
    12-month and 36-month skewness of monthly returns, dollar-volume size
    (log average dollar volume over 12 months), short-term reversal (last
    month's return), 12-month max monthly return, 12-month min monthly
    return, and average dollar-volume turnover trend (6m change in 12m
    log dollar volume).
    """
    mom = {k: returns.rolling(k).apply(lambda s: np.prod(1 + s) - 1, raw=True) for k in (3, 6, 12)}
    mom1 = returns.shift(1)  # short-term reversal: previous month's return
    vol1 = returns.rolling(3).std()
    vol12 = returns.rolling(12).std()
    skew12 = returns.rolling(12).skew()
    skew36 = returns.rolling(36).skew()
    dollar_vol = (month_end_px * volume.resample("ME").sum().reindex(month_end_px.index)).rolling(12).mean()
    size = np.log(dollar_vol.clip(lower=1.0))
    max12 = returns.rolling(12).max()
    min12 = returns.rolling(12).min()
    size_trend = size - size.shift(6)

    raw = {
        "mom3": mom[3], "mom6": mom[6], "mom12": mom[12], "rev1": mom1,
        "vol3": vol1, "vol12": vol12, "skew12": skew12, "skew36": skew36,
        "size": size, "max12": max12, "min12": min12, "size_trend": size_trend,
    }
    ranked = {name: df.apply(_cross_sectional_rank, axis=1) for name, df in raw.items()}
    return ranked


def stack_panel(ranked: dict[str, pd.DataFrame], returns: pd.DataFrame) -> pd.DataFrame:
    """Stack the per-characteristic month x ticker frames into one long panel
    with columns date, ticker, <characteristic columns>, ret_next (the
    return realized in month t+1, aligned to row t, matching r_{i,t+1})."""
    names = list(ranked)
    frames = []
    for name in names:
        frames.append(ranked[name].stack(dropna=False).rename(name))
    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()
    ret_next = returns.shift(-1).stack(dropna=False).rename("ret_next").reset_index()
    ret_next.columns = ["date", "ticker", "ret_next"]
    panel = panel.merge(ret_next, on=["date", "ticker"], how="left")
    return panel


def build_context(returns: pd.DataFrame) -> pd.DataFrame:
    """Three market-wide context variables built from the loaded panel in
    place of the paper's six Goyal et al. (2024) predictors (Section 6.1):
    equal-weight market return, log realized market variance (trailing
    12-month), and average pairwise correlation (trailing 12-month), each
    lagged one month as the paper lags inflation."""
    market_ret = returns.mean(axis=1)
    realized_var = np.log(market_ret.rolling(12).var().clip(lower=1e-8))
    avg_corr = returns.rolling(12).corr().groupby(level=0).apply(
        lambda c: (c.values[np.triu_indices_from(c.values, k=1)]).mean() if c.shape[0] > 1 else np.nan
    )
    context = pd.DataFrame({
        "market_return": market_ret,
        "log_realized_var": realized_var,
        "avg_correlation": avg_corr,
    }).shift(1)  # Section 6.1: context lagged one month, as inflation is
    return context


def standardize_context(context: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Zero mean / unit variance standardization using training-period
    moments only (paper Section 6.1)."""
    mu = context.loc[train_mask].mean()
    sd = context.loc[train_mask].std().replace(0.0, 1.0)
    return (context - mu) / sd
