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
    """Cumulant generating function of r_j = z^T A z + sigma_r*eps, z ~
    N(m, S_tilde), eps ~ N(0,1) (Equation 22, single-asset state z=(beta_j,
    phi_t) of Equation B.13 with A = A_j of Equation B.2).

    Writing z = m + Lx with S_tilde = L L^T and x ~ N(0,I), s*z^T A z + a
    linear-in-x term reduces to the standard quadratic-plus-linear form in a
    standard normal vector, whose cumulant generating function is the
    classical result E[exp(a^T x + x^T C x)] = |I-2C|^{-1/2}
    exp(0.5 a^T (I-2C)^{-1} a) with a = 2s L^T A m, C = s L^T A L; using the
    determinant identity |I-2sL^TAL| = |I-2s S_tilde A| and
    L(I-2sL^TAL)^{-1}L^T = (I-2s S_tilde A)^{-1} S_tilde gives the closed
    form below (verified against a direct two-dimensional moment-generating-
    function quadrature for H=1 to machine precision).

    K(s), K'(s), K''(s) are obtained in closed form by matrix calculus on
    M(s) = I - 2 s B, B = S_tilde A (dM/ds = -2B), rather than by finite
    differences: close to the pole of the cumulant generating function (the
    edge of `_convergence_interval`, which the saddlepoint search of
    `solve_saddlepoint` approaches for larger |r|), a finite-difference
    derivative is numerically unstable because K(s) itself diverges there,
    while the closed-form derivative below stays accurate up to the pole.

    With Minv = M^{-1}, dMinv/ds = 2 Minv B Minv and
    d^2 Minv/ds^2 = 8 Minv B Minv B Minv (product rule applied twice).
    Writing f0(s) = -0.5 logdet M(s), b(s) = p^T Minv(s) q with p = A m,
    q = S_tilde A m, and g(s) = 2 s^2 b(s), h(s) = 0.5 sigma_r2 s^2:
    K(s) = s (m^T A m) + f0(s) + g(s) + h(s)."""
    B = S_tilde @ A
    I = np.eye(S_tilde.shape[0])
    M = I - 2 * s * B
    sign, logdet = np.linalg.slogdet(M)
    Minv = np.linalg.solve(M, I)
    a_lin = m @ A @ m
    p = A @ m
    q = S_tilde @ A @ m

    MinvB = Minv @ B
    f0 = -0.5 * logdet
    f0_p1 = np.trace(MinvB)
    f0_p2 = 2.0 * np.trace(MinvB @ MinvB)

    b0 = p @ Minv @ q
    b_p1 = 2.0 * (p @ Minv @ B @ Minv @ q)
    b_p2 = 8.0 * (p @ Minv @ B @ Minv @ B @ Minv @ q)

    g0 = 2.0 * s ** 2 * b0
    g_p1 = 4.0 * s * b0 + 2.0 * s ** 2 * b_p1
    g_p2 = 4.0 * b0 + 8.0 * s * b_p1 + 2.0 * s ** 2 * b_p2

    K0 = s * a_lin + f0 + g0 + 0.5 * sigma_r2 * s ** 2
    K1 = a_lin + f0_p1 + g_p1 + sigma_r2 * s
    K2 = f0_p2 + g_p2 + sigma_r2
    return K0, K1, K2


def _convergence_interval(S_tilde: np.ndarray, A: np.ndarray, margin: float = 1e-6) -> tuple[float, float]:
    """Open interval of s for which M(s) = I - 2 s S_tilde A stays positive
    definite, i.e. the domain on which K(s) = log E[exp(s r_j)] of Equation
    (22) actually converges.

    `np.linalg.slogdet` returns log|det(M)| regardless of the sign of
    det(M), so a root search that is allowed to leave this interval can find
    a spurious stationary point of the analytically continued (but no
    longer meaningful) log|det| beyond the true pole of the moment
    generating function. Writing S_tilde = L L^T and C = L^T A L
    (symmetric, since A = A^T), M stays positive definite exactly while
    1 - 2 s lambda_i(C) > 0 for every eigenvalue lambda_i(C) simultaneously.
    For a positive eigenvalue lambda, 1/(2 lambda) is decreasing in lambda,
    so the binding (tightest) upper bound on s comes from the LARGEST
    positive eigenvalue; symmetrically the binding lower bound comes from
    the most negative eigenvalue (smallest, i.e. largest in magnitude).
    This bounds s to (1/(2 min(lambda_i<0)), 1/(2 max(lambda_i>0))).

    S_tilde need only be positive semi-definite (the degenerate limit
    S_tilde=0, a deterministic z, is a valid input): writing S_tilde=L L^T
    (L possibly rank-deficient), B=S_tilde A=L(L^T A L)L^T is similar to the
    symmetric matrix L^T A L padded with zero eigenvalues for L's null
    space, so its spectrum is real and obtainable directly from B without
    a Cholesky factor, which requires strict positive-definiteness.
    """
    B = S_tilde @ A
    eigvals = np.linalg.eigvals(B).real
    pos = eigvals[eigvals > 1e-10]
    neg = eigvals[eigvals < -1e-10]
    s_hi = 1.0 / (2.0 * pos.max()) if pos.size else np.inf
    s_lo = 1.0 / (2.0 * neg.min()) if neg.size else -np.inf
    if not (np.isfinite(s_hi) and np.isfinite(s_lo)):
        # No pole on one or both sides (e.g. the degenerate S_tilde=0 limit,
        # where z is deterministic and K(s) is a pure quadratic in s): fall
        # back to a wide finite bracket instead of an unbounded one, since
        # the root search below needs a finite interval.
        finite = np.concatenate([pos, -neg]) if pos.size or neg.size else np.array([1.0])
        span = 1.0 / (2.0 * finite.min()) if finite.size else 1.0
        s_hi = span * 1e6 if not np.isfinite(s_hi) else s_hi
        s_lo = -span * 1e6 if not np.isfinite(s_lo) else s_lo
        return s_lo, s_hi
    width = s_hi - s_lo
    return s_lo + margin * width, s_hi - margin * width


def solve_saddlepoint(r: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float) -> float:
    """Root of K'(s) = r (Section 6.7, "saddlepoint s"), searched only inside
    the true convergence interval of the cumulant generating function (see
    `_convergence_interval`); K'(s) diverges to -inf/+inf at the interval's
    ends, so the root is always bracketed there."""
    def deriv(s_val):
        _, K1, _ = cgf_and_derivatives(s_val, m, S_tilde, A, sigma_r2)
        return K1 - r
    lo, hi = _convergence_interval(S_tilde, A)
    f_lo, f_hi = deriv(lo), deriv(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError("saddlepoint root not bracketed inside the CGF's convergence interval")
    return optimize.brentq(deriv, lo, hi, xtol=1e-10)


def bilinear_log_density(r: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float, tol: float = 1e-9) -> float:
    """log p_bil(r) via Fourier inversion around the saddlepoint (Eq. B.14-B.15)."""
    s_star = solve_saddlepoint(r, m, S_tilde, A, sigma_r2)
    K_star, _, K2_star = cgf_and_derivatives(s_star, m, S_tilde, A, sigma_r2)
    d = np.sqrt(max(K2_star, 1e-12))

    def integrand(w):
        s_val = s_star + 1j * w / d
        M = np.eye(S_tilde.shape[0]) - 2 * s_val * S_tilde @ A
        sign, logdet = np.linalg.slogdet(M.astype(np.complex128))
        # |sign| = 1 for complex slogdet; log(det M) = log(sign) + logdet,
        # so the phase arg(sign) must be kept for complex s (it vanishes for
        # the real s used elsewhere, where sign=+1 inside the convergence
        # interval, which is why the plain real-valued logdet sufficed there
        # but silently dropping it here biased every complex evaluation).
        logdet_complex = np.log(sign) + logdet
        Minv = np.linalg.solve(M, np.eye(M.shape[0]))
        quad_linear = m @ A @ m
        quad_curved = m @ A @ Minv @ S_tilde @ A @ m
        K_val = (
            s_val * quad_linear
            - 0.5 * logdet_complex
            + 2.0 * s_val ** 2 * quad_curved
            + 0.5 * sigma_r2 * s_val ** 2
        )
        return np.real(np.exp(K_val - K_star - 1j * w * r / d))

    integral, _ = integrate.quad_vec(integrand, 0.0, np.inf, epsabs=tol, epsrel=tol)
    density = np.exp(K_star - s_star * r) / (np.pi * d) * integral
    return float(np.log(max(density, 1e-300)))


def gaussian_log_density(r: float, mu: float, sigma2: float) -> float:
    return float(stats.norm.logpdf(r, loc=mu, scale=np.sqrt(sigma2)))


def log_score_gain(r: float, m: np.ndarray, S_tilde: np.ndarray, A: np.ndarray, sigma_r2: float, mu: float, sigma2: float) -> float:
    """Equation (35): log p_bil(r) - log N(r | mu, sigma2)."""
    return bilinear_log_density(r, m, S_tilde, A, sigma_r2) - gaussian_log_density(r, mu, sigma2)
