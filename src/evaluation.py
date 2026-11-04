"""Evaluation criteria, portfolios, and pairwise tests (paper Section 6.3,
Section 6.6, Appendix D.3).

Custom implementation of Equations (30)-(34) and the Holm-adjusted
Newey-West comparisons; `statsmodels` provides the long-run (HAC) variance
estimator used inside the test (Context7: statsmodels.stats.sandwich_covariance
covers OLS HAC standard errors, not a bare Bartlett long-run variance of a
difference series, so the Newey-West kernel itself is implemented here
following Newey and West (1987) and applied via scipy.stats for the normal
p-value).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def predictive_r2(mu_hat: list[np.ndarray], r_true: list[np.ndarray]) -> float:
    """Equation (30), pooled over the evaluation panel."""
    num = sum(np.sum((r - m) ** 2) for r, m in zip(r_true, mu_hat))
    den = sum(np.sum(r ** 2) for r in r_true)
    return 1.0 - num / den


def monthly_rank_correlation(mu_hat: list[np.ndarray], r_true: list[np.ndarray]) -> np.ndarray:
    """Monthly Spearman correlation between predicted and realized returns."""
    out = []
    for r, m in zip(r_true, mu_hat):
        if len(r) < 2 or np.std(m) == 0:
            out.append(np.nan)
            continue
        out.append(stats.spearmanr(m, r).statistic)
    return np.array(out)


def gaussian_log_score(mu_hat: np.ndarray, Sigma_hat: np.ndarray, r_true: np.ndarray) -> float:
    """Equation (31) for one month."""
    N = len(r_true)
    sign, logdet = np.linalg.slogdet(Sigma_hat)
    resid = r_true - mu_hat
    quad = resid @ np.linalg.solve(Sigma_hat, resid)
    return -(N * np.log(2 * np.pi) + logdet + quad) / (2 * N)


def portfolio_weights(mu_hat: np.ndarray, Sigma_hat: np.ndarray) -> dict:
    """Equation (32): GMV and tangency weights, plus an equal-weight long-short
    sorted on mu_hat (used by `long_short_return`)."""
    inv_Sigma = np.linalg.inv(Sigma_hat)
    ones = np.ones(len(mu_hat))
    w_gmv = inv_Sigma @ ones / (ones @ inv_Sigma @ ones)
    raw_tan = inv_Sigma @ mu_hat
    w_tan = raw_tan / np.sum(np.abs(raw_tan))
    return {"gmv": w_gmv, "tangency": w_tan}


def long_short_weights(mu_hat: np.ndarray, decile: float = 0.2) -> np.ndarray:
    """Full-length weight vector for the equal-weight long top decile / short
    bottom decile portfolio sorted on mu_hat (+1/k long, -1/k short, 0
    elsewhere), used both for the return and for Equation (33) turnover."""
    n = len(mu_hat)
    k = max(1, int(round(n * decile)))
    order = np.argsort(mu_hat)
    w = np.zeros(n)
    w[order[:k]] = -1.0 / k
    w[order[-k:]] = 1.0 / k
    return w


def long_short_return(mu_hat: np.ndarray, r_true: np.ndarray, decile: float = 0.2) -> float:
    """Equal-weight long top decile / short bottom decile, sorted on mu_hat."""
    return float(long_short_weights(mu_hat, decile) @ r_true)


def annualized_sharpe(monthly_returns: np.ndarray) -> float:
    mu, sd = np.mean(monthly_returns), np.std(monthly_returns, ddof=1)
    return float(mu / sd * np.sqrt(12)) if sd > 0 else float("nan")


def annualized_vol(monthly_returns: np.ndarray) -> float:
    return float(np.std(monthly_returns, ddof=1) * np.sqrt(12))


def turnover(w_prev: np.ndarray, w_curr: np.ndarray, r_assets: np.ndarray, r_f: float) -> float:
    """Equation (33): drift the previous weights forward by one month of
    realized returns before comparing with the new weights."""
    port_ret = r_f + w_prev @ r_assets
    w_drift = w_prev * (1 + r_f + r_assets) / (1 + port_ret)
    return float(np.sum(np.abs(w_curr - w_drift)))


def net_return(w: np.ndarray, r_assets: np.ndarray, turnover_value: float, cost_bps: float) -> float:
    """Equation (34): net-of-cost portfolio return, cost in basis points per
    dollar traded."""
    cost = (cost_bps / 1e4) * turnover_value
    return float(w @ r_assets - cost)


def break_even_cost_bps(gross_excess_returns: np.ndarray, turnovers: np.ndarray) -> float:
    """Proportional cost, in basis points, at which the average net excess
    return is zero (Section 6.6)."""
    avg_gross = np.mean(gross_excess_returns)
    avg_turnover = np.mean(turnovers)
    if avg_turnover == 0:
        return float("nan")
    return float(avg_gross / avg_turnover * 1e4)


def newey_west_long_run_variance(series: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of a mean-zero series (Newey and
    West, 1987), used for the pairwise comparisons of Appendix D.3."""
    series = series - series.mean()
    n = len(series)
    gamma0 = np.sum(series ** 2) / n
    lrv = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma = np.sum(series[lag:] * series[:-lag]) / n
        lrv += 2 * weight * gamma
    return max(lrv, 1e-12)


def newey_west_statistic(diff_series: np.ndarray, lags: int = 12) -> tuple[float, float]:
    """Newey-West t-statistic and two-sided p-value for H0: mean(diff)=0.

    Positive values favor the first model (see each experiment script for the
    sign convention used for GMV volatility differences)."""
    n = len(diff_series)
    mean_diff = diff_series.mean()
    lrv = newey_west_long_run_variance(diff_series, lags)
    se = np.sqrt(lrv / n)
    t_stat = mean_diff / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return float(t_stat), float(p_value)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm (1979) step-down adjustment for a family of p-values."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()
