"""Bilinear predictive density by one-dimensional Fourier inversion of the
cumulant generating function (paper Appendix B.3, Section 6.7).

Custom implementation (paper equation): the cumulant generating function and
saddlepoint construction of Equations (22) and (B.13)-(B.15) have no library
equivalent; `scipy.integrate.quad_vec` and `scipy.optimize.brentq` (both part
of scipy, already a pinned dependency) supply the numerical primitives the
paper specifies (Fourier inversion by quadrature, saddlepoint root-finding).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, stats


def build_A_j(omega_h: np.ndarray) -> np.ndarray:
    """Equation (B.2): Omega_H = [[0, I_H], [I_H, 0]] / 2 for the stacked
    (beta_j, phi_t) state in R^{2H}; A_j = (e_j e_j^T) kron Omega_H collapsed
    onto the 2H-dim state, i.e. the bilinear form matrix for asset j's return
    r_j = beta_j^T phi_t = z^T A z with z=(beta_j, phi_t) in R^{2H}."""
    H = omega_h
    Omega = np.zeros((2 * H, 2 * H))
    Omega[:H, H:] = 0.5 * np.eye(H)
    Omega[H:, :H] = 0.5 * np.eye(H)
    return Omega


def cgf_and_derivatives(s: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float):
    """Equation (22): K(s) for the scalar return r_j, together with K'(s) and
    K''(s) obtained by automatic differentiation via finite differences of
    the closed-form expression (stable because A, S_tilde are fixed, small
    (2H x 2H) matrices)."""
    def K(s_val):
        M = np.eye(S_tilde.shape[0]) - 2 * s_val * S_tilde @ A
        sign, logdet = np.linalg.slogdet(M)
        Minv = np.linalg.solve(M, np.eye(M.shape[0]))
        quad = m @ A @ Minv @ m
        return -0.5 * logdet + s_val * quad + 0.5 * sigma_r2 * s_val ** 2

    h = 1e-4
    K0 = K(s)
    K1 = (K(s + h) - K(s - h)) / (2 * h)
    K2 = (K(s + h) - 2 * K0 + K(s - h)) / h ** 2
    return K0, K1, K2


def solve_saddlepoint(r: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float, bracket: float = 50.0) -> float:
    """Root of K'(s) = r (Section 6.7, "saddlepoint s")."""
    def deriv(s_val):
        _, K1, _ = cgf_and_derivatives(s_val, m, S_tilde, A, sigma_r2)
        return K1 - r
    lo, hi = -bracket, bracket
    f_lo, f_hi = deriv(lo), deriv(hi)
    tries = 0
    while f_lo * f_hi > 0 and tries < 6:
        bracket *= 2
        lo, hi = -bracket, bracket
        f_lo, f_hi = deriv(lo), deriv(hi)
        tries += 1
    if f_lo * f_hi > 0:
        raise RuntimeError("saddlepoint bracket search failed")
    return optimize.brentq(deriv, lo, hi, xtol=1e-10)


def bilinear_log_density(r: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float, tol: float = 1e-9) -> float:
    """log p_bil(r) via Fourier inversion around the saddlepoint (Eq. B.14-B.15)."""
    s_star = solve_saddlepoint(r, m, S_tilde, A, sigma_r2)
    K_star, _, K2_star = cgf_and_derivatives(s_star, m, S_tilde, A, sigma_r2)
    d = np.sqrt(max(K2_star, 1e-12))

    def integrand(w):
        s_val = s_star + 1j * w / d
        M = np.eye(S_tilde.shape[0]) - 2 * s_val * S_tilde @ A
        sign, logdet = np.linalg.slogdet(M)
        Minv = np.linalg.solve(M, np.eye(M.shape[0]))
        quad = m @ A @ Minv @ m
        K_val = -0.5 * logdet + s_val * quad + 0.5 * sigma_r2 * s_val ** 2
        return np.real(np.exp(K_val - K_star - 1j * w * r / d))

    integral, _ = integrate.quad_vec(integrand, 0.0, np.inf, epsabs=tol, epsrel=tol)
    density = np.exp(K_star - s_star * r) / (np.pi * d) * integral
    return float(np.log(max(density, 1e-300)))


def gaussian_log_density(r: float, mu: float, sigma2: float) -> float:
    return float(stats.norm.logpdf(r, loc=mu, scale=np.sqrt(sigma2)))


def log_score_gain(r: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float, mu: float, sigma2: float) -> float:
    """Equation (35): log p_bil(r) - log N(r | mu, sigma2)."""
    return bilinear_log_density(r, m, S_tilde, A, sigma_r2) - gaussian_log_density(r, mu, sigma2)
