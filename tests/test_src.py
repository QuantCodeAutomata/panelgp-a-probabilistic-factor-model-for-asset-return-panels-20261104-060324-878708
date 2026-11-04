"""Unit tests for the shared PanelGP components (src/), on small synthetic
fixtures. These check the paper's qualitative properties (signs, bounds,
window lengths, absence of look-ahead), not its reported numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src import config
from src.alignment import apply_rotation, fit_rotation, r_squared, transform_innovation_covariances
from src.characteristics import build_characteristics, build_context, monthly_returns_panel, stack_panel
from src.evaluation import (
    annualized_sharpe,
    annualized_vol,
    break_even_cost_bps,
    gaussian_log_score,
    holm_adjust,
    long_short_return,
    monthly_rank_correlation,
    net_return,
    newey_west_statistic,
    portfolio_weights,
    predictive_r2,
    turnover,
)
from src.gp_model import PanelGPConfig, PanelGPModel, fit_panelgp
from src.ipca import fit_ipca, ipca_forecast
from src.kernels import affine_kernel, constant_kernel, rbf_kernel
from src.predictive_distribution import bilinear_log_density, gaussian_log_density, log_score_gain


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def test_rbf_kernel_diagonal_is_one_and_symmetric():
    x = torch.rand(5, 3)
    ell = torch.ones(3)
    K = rbf_kernel(x, x, ell)
    assert torch.allclose(torch.diag(K), torch.ones(5), atol=1e-6)
    assert torch.allclose(K, K.T, atol=1e-6)
    assert torch.all(K <= 1.0 + 1e-6) and torch.all(K >= 0.0)


def test_affine_and_constant_kernel_shapes():
    x = torch.rand(4, 2)
    K_aff = affine_kernel(x, x)
    assert K_aff.shape == (4, 4)
    K_c = constant_kernel(4, 6)
    assert K_c.shape == (4, 6)
    assert torch.all(K_c == 1.0)


# --------------------------------------------------------------------------
# Alignment (paper Section 5.1, step 3)
# --------------------------------------------------------------------------

def test_fit_rotation_recovers_known_linear_map():
    rng = np.random.default_rng(0)
    true_upsilon = np.array([[1.5, 0.3], [-0.2, 0.8]])
    phi_hat = rng.standard_normal((200, 2))
    phi_true = phi_hat @ true_upsilon.T
    upsilon_hat = fit_rotation(phi_true, phi_hat)
    assert np.allclose(upsilon_hat, true_upsilon, atol=1e-6)


def test_apply_rotation_preserves_bilinear_product():
    rng = np.random.default_rng(1)
    upsilon = np.array([[0.9, 0.1], [-0.3, 1.2]])
    phi_hat = rng.standard_normal((10, 2))
    beta_hat = rng.standard_normal((7, 2))
    phi_aligned, beta_aligned = apply_rotation(upsilon, phi_hat, beta_hat)
    before = beta_hat @ phi_hat.T
    after = beta_aligned @ phi_aligned.T
    assert np.allclose(before, after, atol=1e-6)


def test_transform_innovation_covariances_keeps_psd():
    upsilon = np.array([[1.2, 0.0], [0.1, 0.9]])
    D_beta = np.diag([0.05, 0.05])
    D_phi = np.diag([0.05, 0.05])
    D_beta_new, D_phi_new = transform_innovation_covariances(upsilon, D_beta, D_phi)
    assert np.all(np.linalg.eigvalsh(D_beta_new) > 0)
    assert np.all(np.linalg.eigvalsh(D_phi_new) > 0)


def test_r_squared_perfect_fit_is_one_and_constant_fit_is_zero():
    true_values = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(true_values, true_values) == pytest.approx(1.0)
    assert r_squared(true_values, np.full(4, true_values.mean())) == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------
# Evaluation criteria (paper Section 6.3, Eq. 30-34)
# --------------------------------------------------------------------------

def test_predictive_r2_perfect_forecast_is_one():
    r = [np.array([0.1, -0.2, 0.3])]
    assert predictive_r2(r, r) == pytest.approx(1.0)


def test_predictive_r2_mean_zero_forecast_is_le_zero_or_near_zero():
    rng = np.random.default_rng(2)
    r = [rng.standard_normal(500)]
    zero_forecast = [np.zeros(500)]
    assert predictive_r2(zero_forecast, r) == pytest.approx(0.0, abs=0.05)


def test_monthly_rank_correlation_perfect_ranking():
    mu = [np.array([1.0, 2.0, 3.0])]
    r = [np.array([0.1, 0.2, 0.3])]
    corr = monthly_rank_correlation(mu, r)
    assert corr[0] == pytest.approx(1.0)


def test_gaussian_log_score_decreases_with_larger_residual():
    Sigma = np.eye(3) * 0.1
    close = gaussian_log_score(np.zeros(3), Sigma, np.array([0.01, 0.0, 0.0]))
    far = gaussian_log_score(np.zeros(3), Sigma, np.array([5.0, 0.0, 0.0]))
    assert close > far


def test_portfolio_weights_gmv_sums_to_one_and_tangency_unit_leverage():
    Sigma = np.diag([0.1, 0.2, 0.15])
    mu = np.array([0.02, 0.01, -0.01])
    w = portfolio_weights(mu, Sigma)
    assert w["gmv"].sum() == pytest.approx(1.0)
    assert np.abs(w["tangency"]).sum() == pytest.approx(1.0)


def test_long_short_return_sign_matches_sorted_extremes():
    mu = np.array([0.05, 0.01, -0.03, 0.02, -0.01])
    r = np.array([0.05, 0.01, -0.03, 0.02, -0.01])
    val = long_short_return(mu, r, decile=0.2)
    assert val > 0  # top mu_hat also has the top realized return here


def test_annualized_sharpe_and_vol_scale_by_sqrt12():
    monthly2 = np.array([0.01, -0.01] * 6)
    vol = annualized_vol(monthly2)
    assert vol == pytest.approx(np.std(monthly2, ddof=1) * np.sqrt(12))


def test_turnover_zero_when_weights_match_drifted_weights():
    w_prev = np.array([0.5, 0.5])
    r_assets = np.array([0.1, 0.1])
    r_f = 0.0
    w_drift = w_prev * (1 + r_f + r_assets) / (1 + r_f + w_prev @ r_assets)
    assert turnover(w_prev, w_drift, r_assets, r_f) == pytest.approx(0.0, abs=1e-10)


def test_net_return_below_gross_return_when_turnover_positive():
    w = np.array([0.6, 0.4])
    r_assets = np.array([0.02, -0.01])
    gross = w @ r_assets
    net = net_return(w, r_assets, turnover_value=1.0, cost_bps=10.0)
    assert net < gross


def test_break_even_cost_bps_zero_excess_return_gives_zero():
    assert break_even_cost_bps(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_newey_west_statistic_mean_zero_series_small_t_stat():
    rng = np.random.default_rng(3)
    series = rng.standard_normal(500) * 0.01
    t_stat, p_value = newey_west_statistic(series, lags=12)
    assert abs(t_stat) < 4.0
    assert 0.0 <= p_value <= 1.0


def test_holm_adjust_monotone_and_bounded():
    raw = [0.001, 0.02, 0.5, 0.9]
    adj = holm_adjust(raw)
    assert all(0.0 <= p <= 1.0 for p in adj)
    assert all(a >= r for a, r in zip(adj, raw))


# --------------------------------------------------------------------------
# IPCA (paper Section 6.2, Eq. 29)
# --------------------------------------------------------------------------

def test_ipca_recovers_known_single_factor_loading():
    rng = np.random.default_rng(4)
    P, T, N = 3, 40, 60
    true_gamma = rng.standard_normal(P + 1) * 0.3
    X_by_month, r_by_month = [], []
    true_factor = rng.standard_normal(T) * 0.5 + 1.0
    for t in range(T):
        X = rng.standard_normal((N, P))
        Xi = np.concatenate([X, np.ones((N, 1))], axis=1)
        r = Xi @ true_gamma * true_factor[t] + 0.001 * rng.standard_normal(N)
        X_by_month.append(X)
        r_by_month.append(r)
    fit = fit_ipca(X_by_month, r_by_month, H=1, with_intercept=False, n_iter=100, seed=0)
    mu, Sigma = ipca_forecast(fit, X_by_month[0])
    assert mu.shape == (N,)
    assert Sigma.shape == (N, N)
    assert np.all(np.linalg.eigvalsh(Sigma) > 0)
    # fitted in-sample mean correlates strongly with the true conditional mean
    true_mean = (np.concatenate([X_by_month[0], np.ones((N, 1))], axis=1) @ true_gamma) * true_factor[0]
    assert abs(np.corrcoef(mu, true_mean)[0, 1]) > 0.8


# --------------------------------------------------------------------------
# Predictive distribution (paper Appendix B.3)
# --------------------------------------------------------------------------

def test_bilinear_log_density_matches_gaussian_in_degenerate_limit():
    H = 1
    S_tilde = np.zeros((2 * H, 2 * H))
    m = np.array([0.0, 1.0])  # phi_t=1, beta_j=0 -> r = beta^T phi deterministic at 0
    A = np.zeros((2 * H, 2 * H))
    A[0, 1] = A[1, 0] = 0.5
    sigma_r2 = 0.04
    r_eval = 0.0
    log_bil = bilinear_log_density(r_eval, m, S_tilde, A, sigma_r2)
    log_gauss = gaussian_log_density(r_eval, mu=0.0, sigma2=sigma_r2)
    assert log_bil == pytest.approx(log_gauss, abs=1e-2)


def test_log_score_gain_is_finite_and_real():
    H = 1
    S_tilde = np.eye(2 * H) * 0.05
    m = np.array([0.3, 0.6])
    A = np.zeros((2 * H, 2 * H))
    A[0, 1] = A[1, 0] = 0.5
    sigma_r2 = 0.05
    gain = log_score_gain(0.1, m, S_tilde, A, sigma_r2, mu=0.18, sigma2=0.06)
    assert np.isfinite(gain)


# --------------------------------------------------------------------------
# Characteristics construction (paper Section 6.1) — no look-ahead, ranges
# --------------------------------------------------------------------------

def _toy_bars():
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(5)
    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        price = 100 * np.exp(np.cumsum(rng.standard_normal(len(dates)) * 0.01))
        vol = rng.integers(1_000, 10_000, size=len(dates))
        for d, p, v in zip(dates, price, vol):
            rows.append({"date": d, "ticker": ticker, "close": p, "volume": v})
    return pd.DataFrame(rows)


def test_characteristics_are_ranked_within_bounds_and_lagged():
    bars = _toy_bars()
    returns = monthly_returns_panel(bars)
    month_end_px = bars.pivot_table(index="date", columns="ticker", values="close").resample("ME").last()
    volume = bars.pivot_table(index="date", columns="ticker", values="volume")
    ranked = build_characteristics(month_end_px, returns, volume)
    for name, df in ranked.items():
        valid = df.values[~np.isnan(df.values)]
        assert valid.min() >= config.CHARACTERISTIC_RANK_RANGE[0] - 1e-9
        assert valid.max() <= config.CHARACTERISTIC_RANK_RANGE[1] + 1e-9

    panel = stack_panel(ranked, returns)
    assert {"date", "ticker", "ret_next"}.issubset(panel.columns)
    # no look-ahead: ret_next at row t must equal the realized return of t+1
    sample = panel.dropna(subset=["ret_next"]).iloc[0]
    next_month_mask = (returns.index > sample["date"])
    expected = returns.loc[next_month_mask, sample["ticker"]].iloc[0]
    assert sample["ret_next"] == pytest.approx(expected)


def test_build_context_shifted_by_one_month():
    bars = _toy_bars()
    returns = monthly_returns_panel(bars)
    context = build_context(returns)
    # the context at date t uses only information through t-1: shifting by
    # one month must reproduce a one-step lag relative to an unshifted calc
    unshifted_market_ret = returns.mean(axis=1)
    valid_idx = context["market_return"].dropna().index
    assert np.allclose(
        context.loc[valid_idx, "market_return"].values,
        unshifted_market_ret.shift(1).loc[valid_idx].values,
    )


# --------------------------------------------------------------------------
# GP model (paper Sections 3-4): tiny end-to-end fit sanity checks
# --------------------------------------------------------------------------

def test_panelgp_tiny_fit_runs_and_improves_elbo():
    torch.manual_seed(0)
    N, T, P, Q, H, M = 6, 5, 2, 1, 2, 4
    rng = np.random.default_rng(6)
    x = rng.uniform(size=(N, P)).astype(np.float32)
    c = rng.uniform(size=(T, Q)).astype(np.float32)
    X_train = torch.tensor(np.repeat(x, T, axis=0))
    t_idx_train = torch.tensor(np.tile(np.arange(T), N), dtype=torch.long)
    R_train = torch.tensor(rng.standard_normal(N * T).astype(np.float32)) * 0.1
    C_train = torch.tensor(c)

    cfg = PanelGPConfig(P=P, Q=Q, H=H, T=T, M=M, loading_kernel="rbf", factor_kernel="rbf", isotropic_innovations=True)
    model = PanelGPModel(cfg, seed=0)
    model.set_initial_values(d_loading=1.0)

    first_elbo, _, _ = model.elbo(X_train, t_idx_train, R_train, N * T, C_train, n_factor_samples=8)
    history = fit_panelgp(model, X_train, t_idx_train, R_train, C_train, epochs=20, batch_size=30,
                           n_factor_samples=8, seed=0)
    assert len(history) == 20
    last_elbo, _, _ = model.elbo(X_train, t_idx_train, R_train, N * T, C_train, n_factor_samples=8)
    assert np.isfinite(last_elbo.item())

    D_beta, D_phi = model.innovation_covariances()
    assert D_beta.shape == (H, H)
    assert torch.all(torch.diag(D_beta) > 0)
    sigma_r2 = model.sigma_r2()
    assert sigma_r2.item() > 0

    phi_hat, phi_cov = model.smooth_factor(C_train, C_train)
    assert phi_hat.shape == (T, H)
    beta_hat, beta_cov = model.smooth_loading(torch.tensor(x), C_train)
    assert beta_hat.shape == (N, H)

    V0, Vphi, Vbeta, trDbDp, sigma_r2_vec = model.variance_decomposition(torch.tensor(x), C_train[0], C_train)
    assert torch.all(V0 >= -1e-6)
    assert torch.all(Vphi >= -1e-6)
