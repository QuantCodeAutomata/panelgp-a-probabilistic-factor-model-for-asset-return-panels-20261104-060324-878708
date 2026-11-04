"""PanelGP: sparse variational Gaussian-process factor model.

Custom implementation (paper Sections 3-4 and Appendices A-C). Context7 was
consulted for sparse variational GP inference (gpytorch / GPflow style
inducing-point regression); none of those libraries implement the bilinear,
two-input (asset characteristics x date context) likelihood with a jointly
dense posterior over inducing loadings and realized training factors that
the paper specifies (Eq. 7-17), so the estimator below is written from the
paper's equations rather than adapted from a GP toolbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.kernels import affine_kernel, constant_kernel, rbf_kernel

JITTER_INIT = 1e-7


def _stabilized_cholesky(mat: torch.Tensor) -> torch.Tensor:
    """Cholesky factorization with jitter increased tenfold until it succeeds
    (paper Appendix C.3, "numerical stabilization")."""
    jitter = JITTER_INIT
    n = mat.shape[-1]
    eye = torch.eye(n, dtype=mat.dtype, device=mat.device)
    for _ in range(8):
        try:
            return torch.linalg.cholesky(mat + jitter * eye)
        except RuntimeError:
            jitter *= 10.0
    raise RuntimeError("Cholesky factorization failed even with large jitter")


@dataclass
class PanelGPConfig:
    """Configuration of one PanelGP estimator instance."""

    P: int                      # number of characteristics
    Q: int                      # number of context variables
    H: int                      # number of factors
    T: int                      # number of training dates
    M: int                      # trainable inducing inputs per component
    loading_kernel: str = "rbf"     # "affine" or "rbf", Eq. C.3
    factor_kernel: str = "rbf"      # "constant" or "rbf", Eq. C.4
    isotropic_innovations: bool = False  # shared sigma_beta^2, sigma_phi^2 across h (simulation design)
    factor_kernel_has_constant: bool = True  # simulation keeps the constant term even for the RBF kernel (Appendix C.2 "Simulation")


class PanelGPModel(nn.Module):
    """Sparse variational PanelGP estimator (Eq. 7-20, Appendix C)."""

    def __init__(self, cfg: PanelGPConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(seed)
        H, M, T, P, Q = cfg.H, cfg.M, cfg.T, cfg.P, cfg.Q

        # --- kernel parameters, one set per component h (Section 3.3) ---
        if cfg.loading_kernel == "rbf":
            ell0 = math.sqrt(P / 12.0) if P > 0 else 1.0  # Equation C.7
            self.raw_ell_beta = nn.Parameter(
                torch.full((H, P), math.log(math.e ** ell0 - 1.0) if ell0 > 0 else 0.0)
            )
        self.raw_eta = nn.Parameter(torch.zeros(H))  # softplus -> eta_h, init via set_initial_values()
        if cfg.isotropic_innovations:
            self.raw_tau_beta = nn.Parameter(torch.zeros(1))
        else:
            self.raw_tau_beta = nn.Parameter(torch.zeros(H))

        if cfg.factor_kernel == "rbf":
            ell0_q = math.sqrt(Q / 12.0) if Q > 0 else 1.0
            self.raw_ell_phi = nn.Parameter(torch.full((H, Q), math.log(math.e ** ell0_q - 1.0)))
            if cfg.isotropic_innovations:
                # Appendix C.2 "Simulation": shared rho via sigmoid, two-way softmax for (a,b)
                self.raw_rho_shared = nn.Parameter(torch.zeros(1))
                self.raw_ab_logits = nn.Parameter(torch.zeros(H, 2))
            else:
                self.raw_abrho_logits = nn.Parameter(torch.zeros(H, 3))
        else:  # constant factor kernel
            if cfg.isotropic_innovations:
                self.raw_rho_shared = nn.Parameter(torch.zeros(1))
            else:
                self.raw_rho_logit = nn.Parameter(torch.zeros(H))

        self.raw_sigma_r2 = nn.Parameter(torch.zeros(1))

        # --- inducing inputs, trainable (Section 4.1) ---
        self.Z = nn.Parameter(torch.rand(H, M, P))

        # --- whitened variational state (Appendix C.3) ---
        d_v = H * M + H * T
        self.d_v = d_v
        self.m_omega = nn.Parameter(torch.randn(d_v) * 0.001)
        # raw lower-triangular Cholesky factor of S_omega; diagonal goes through softplus
        self.raw_L_omega = nn.Parameter(torch.zeros(d_v, d_v))
        with torch.no_grad():
            self.raw_L_omega.copy_(torch.eye(d_v) * math.log(math.e - 1.0))

        self._tril_mask = torch.tril(torch.ones(d_v, d_v), diagonal=-1)
        self._diag_mask = torch.eye(d_v)

    # ------------------------------------------------------------------
    # Parameter transforms
    # ------------------------------------------------------------------
    def eta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_eta)

    def tau_beta(self) -> torch.Tensor:
        tau = torch.nn.functional.softplus(self.raw_tau_beta)
        if self.cfg.isotropic_innovations:
            tau = tau.expand(self.cfg.H)
        return tau

    def sigma_r2(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_sigma_r2)[0]

    def factor_shares(self):
        """Return (a_h, b_h, rho_h) each shape (H,), Eq. 12/C.4."""
        H = self.cfg.H
        if self.cfg.factor_kernel == "rbf":
            if self.cfg.isotropic_innovations:
                rho = torch.sigmoid(self.raw_rho_shared).expand(H)
                ab = torch.softmax(self.raw_ab_logits, dim=-1) * (1 - rho).unsqueeze(-1)
                a, b = ab[:, 0], ab[:, 1]
            else:
                shares = torch.softmax(self.raw_abrho_logits, dim=-1)
                a, b, rho = shares[:, 0], shares[:, 1], shares[:, 2]
        else:  # constant kernel: a_h = 0
            if self.cfg.isotropic_innovations:
                rho = torch.sigmoid(self.raw_rho_shared).expand(H)
            else:
                rho = torch.sigmoid(self.raw_rho_logit)
            a = torch.zeros(H)
            b = 1 - rho
        return a, b, rho

    def set_initial_values(self, d_loading: float):
        """Equation C.6 initialization of sigma_r^2 and eta_h."""
        with torch.no_grad():
            self.raw_sigma_r2.fill_(_inv_softplus(torch.tensor(0.1)))
            gamma_beta = 0.1
            H = self.cfg.H
            eta0 = 0.9 / (H * (d_loading + gamma_beta))
            self.raw_eta.fill_(_inv_softplus(torch.tensor(eta0)))
            tau0 = gamma_beta * eta0
            self.raw_tau_beta.fill_(_inv_softplus(torch.tensor(tau0)))
            if self.cfg.factor_kernel == "rbf":
                # initial shares rho=0.1, b=0.1, a=0.8 (Appendix C.4)
                if self.cfg.isotropic_innovations:
                    self.raw_rho_shared.fill_(_logit(torch.tensor(0.1)))
                    self.raw_ab_logits.zero_()  # softmax -> equal split of remaining 0.9 -> a=b=0.45 (documented deviation: isotropic design shares a,b equally rather than 0.8/0.1)
                else:
                    logits = torch.log(torch.tensor([0.8, 0.1, 0.1]))
                    self.raw_abrho_logits.copy_(logits.expand(H, 3))
            else:
                if self.cfg.isotropic_innovations:
                    self.raw_rho_shared.fill_(_logit(torch.tensor(0.1)))
                else:
                    self.raw_rho_logit.fill_(_logit(torch.tensor(0.1)))

    # ------------------------------------------------------------------
    # Kernel matrices
    # ------------------------------------------------------------------
    def _k_loading(self, x1, x2, h):
        if self.cfg.loading_kernel == "affine":
            base = affine_kernel(x1, x2)
        else:
            ell = torch.nn.functional.softplus(self.raw_ell_beta[h])
            base = rbf_kernel(x1, x2, ell)
        return self.eta()[h] * base

    def _k_factor_smooth(self, c1, c2, h, a, b):
        """a_h * RBF + b_h, the smooth part of Eq. 12 without the innovation term."""
        if self.cfg.factor_kernel == "rbf":
            ell = torch.nn.functional.softplus(self.raw_ell_phi[h])
            rbf = rbf_kernel(c1, c2, ell)
            return a[h] * rbf + b[h]
        return b[h] * constant_kernel(c1.shape[0], c2.shape[0], device=c1.device, dtype=c1.dtype)

    # ------------------------------------------------------------------
    # Prior covariance / whitened state
    # ------------------------------------------------------------------
    def _prior_cholesky(self, C_train: torch.Tensor):
        """L_beta_h (M,M) and L_phi_h (T,T) per component (Eq. 10-11)."""
        a, b, rho = self.factor_shares()
        L_beta, L_phi, Kuu, Kphi = [], [], [], []
        for h in range(self.cfg.H):
            Kb = self._k_loading(self.Z[h], self.Z[h], h)
            Kuu.append(Kb)
            L_beta.append(_stabilized_cholesky(Kb))
            Kf_smooth = self._k_factor_smooth(C_train, C_train, h, a, b)
            Kf = Kf_smooth + rho[h] * torch.eye(self.cfg.T, device=C_train.device, dtype=C_train.dtype)
            Kphi.append(Kf)
            L_phi.append(_stabilized_cholesky(Kf))
        return L_beta, L_phi, Kuu, Kphi

    def variational_state(self, C_train: torch.Tensor):
        """Return m_v (d_v,), S_v (d_v,d_v), and cached prior factors."""
        H, M, T = self.cfg.H, self.cfg.M, self.cfg.T
        L_beta, L_phi, Kuu, Kphi = self._prior_cholesky(C_train)
        blocks = L_beta + L_phi  # H blocks of size M, then H blocks of size T
        Lv = torch.block_diag(*blocks)
        L_omega = torch.tril(self.raw_L_omega, diagonal=-1) + torch.diag(
            torch.nn.functional.softplus(torch.diagonal(self.raw_L_omega))
        )
        m_v = Lv @ self.m_omega
        A = Lv @ L_omega
        S_v = A @ A.T
        return m_v, S_v, L_beta, L_phi, Kuu, Kphi, L_omega

    def kl_divergence(self, L_omega: torch.Tensor) -> torch.Tensor:
        """Equation (C.5): KL(q(omega) || N(0, I))."""
        S_omega_trace = (L_omega ** 2).sum()
        quad = (self.m_omega ** 2).sum()
        log_det = 2.0 * torch.log(torch.diagonal(L_omega)).sum()
        return 0.5 * (S_omega_trace + quad - self.d_v - log_det)

    def _idx_U(self, h):
        return slice(h * self.cfg.M, (h + 1) * self.cfg.M)

    def _idx_F(self, h):
        base = self.cfg.H * self.cfg.M
        return slice(base + h * self.cfg.T, base + (h + 1) * self.cfg.T)

    # ------------------------------------------------------------------
    # GP conditionals (Eq. 19-20)
    # ------------------------------------------------------------------
    def _loading_conditional(self, X_query, h, Kuu_h, L_beta_h):
        Kqz = self._k_loading(X_query, self.Z[h], h)             # (n, M)
        G = torch.cholesky_solve(Kqz.T, L_beta_h).T               # (n, M)
        # diagonal of k(x,x): affine -> 1+||x||^2 ; rbf -> 1 (times eta)
        if self.cfg.loading_kernel == "affine":
            kqq = self.eta()[h] * (1.0 + (X_query ** 2).sum(-1))
        else:
            kqq = self.eta()[h] * torch.ones(X_query.shape[0], dtype=X_query.dtype)
        r_diag = kqq - (G * Kqz).sum(-1)
        return G, r_diag.clamp_min(1e-10)

    def _factor_conditional(self, C_query, h, a, b, Kphi_h, L_phi_h, C_train):
        Kqc = self._k_factor_smooth(C_query, C_train, h, a, b)    # (n, T), smooth part only (no rho on cross term)
        G = torch.cholesky_solve(Kqc.T, L_phi_h).T                 # (n, T)
        kqq_smooth = self._k_factor_smooth(C_query, C_query, h, a, b).diagonal()
        r_diag = kqq_smooth - (G * Kqc).sum(-1)
        return G, r_diag.clamp_min(1e-10)

    # ------------------------------------------------------------------
    # Training objective (Eq. 7-17)
    # ------------------------------------------------------------------
    def elbo(self, X_batch, t_idx_batch, R_batch, N_total, C_train, n_factor_samples=64):
        """Stochastic evidence lower bound for one minibatch (Eq. 17)."""
        H = self.cfg.H
        m_v, S_v, L_beta, L_phi, Kuu, Kphi, L_omega = self.variational_state(C_train)

        # --- smooth loading posterior at the batch characteristics ---
        m_beta = torch.zeros(X_batch.shape[0], H)
        Sigma_beta = torch.zeros(X_batch.shape[0], H, H)
        G_list = []
        for h in range(H):
            G, r_diag = self._loading_conditional(X_batch, h, Kuu[h], L_beta[h])
            G_list.append(G)
            m_beta[:, h] = G @ m_v[self._idx_U(h)]
            Sigma_beta[:, h, h] += r_diag
        for h in range(H):
            for hp in range(H):
                block = S_v[self._idx_U(h), self._idx_U(hp)]
                Sigma_beta[:, h, hp] += torch.einsum("nm,ml,nl->n", G_list[h], block, G_list[hp])

        # --- realized training factor posterior for the batch's dates ---
        unique_t, inverse = torch.unique(t_idx_batch, return_inverse=True)
        n_unique = unique_t.shape[0]
        m_phi_u = torch.zeros(n_unique, H)
        S_phi_u = torch.zeros(n_unique, H, H)
        for h in range(H):
            idx_h = self._idx_F(h)
            m_phi_u[:, h] = m_v[idx_h][unique_t]
            for hp in range(H):
                idx_hp = self._idx_F(hp)
                sub = S_v[idx_h, :][:, idx_hp]  # (T,T)
                S_phi_u[:, h, hp] = sub[unique_t, unique_t]

        L_phi_u = _stabilized_cholesky(S_phi_u)
        n_draws = n_factor_samples // 2
        z = torch.randn(n_unique, n_draws, H)
        z = torch.cat([z, -z], dim=1)  # antithetic pairs -> n_factor_samples draws
        phi_samples_u = m_phi_u.unsqueeze(1) + torch.einsum("uhk,usk->ush", L_phi_u, z)
        phi_samples = phi_samples_u[inverse]  # (batch, S, H)

        tau_beta = self.tau_beta()
        sigma_r2 = self.sigma_r2()
        sigma2_eff = sigma_r2 + torch.einsum("ush,h,ush->us", phi_samples, tau_beta, phi_samples)
        pred_mean = torch.einsum("nh,nsh->ns", m_beta, phi_samples)
        pred_var_term = torch.einsum("nsh,nhk,nsk->ns", phi_samples, Sigma_beta, phi_samples)
        resid2 = (R_batch.unsqueeze(1) - pred_mean) ** 2 + pred_var_term
        loglik = -0.5 * torch.log(2 * math.pi * sigma2_eff) - 0.5 * resid2 / sigma2_eff
        loglik = loglik.mean(dim=1)  # average over factor samples

        scale = N_total / X_batch.shape[0]
        expected_loglik = scale * loglik.sum()
        kl = self.kl_divergence(L_omega)
        return expected_loglik - kl, expected_loglik.detach(), kl.detach()

    # ------------------------------------------------------------------
    # Prediction: smooth function values (for recovery / forecasting)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def smooth_loading(self, X_query, C_train):
        """Posterior mean/variance of the smooth loading map beta_h(x), no innovation."""
        m_v, S_v, L_beta, L_phi, Kuu, Kphi, _ = self.variational_state(C_train)
        H = self.cfg.H
        mean = torch.zeros(X_query.shape[0], H)
        var = torch.zeros(X_query.shape[0], H)
        for h in range(H):
            G, r_diag = self._loading_conditional(X_query, h, Kuu[h], L_beta[h])
            mean[:, h] = G @ m_v[self._idx_U(h)]
            block = S_v[self._idx_U(h), self._idx_U(h)]
            var[:, h] = r_diag + torch.einsum("nm,ml,nl->n", G, block, G)
        return mean, var

    @torch.no_grad()
    def smooth_factor(self, C_query, C_train):
        """Posterior mean/variance of the smooth factor map phi_h(c), no innovation."""
        m_v, S_v, L_beta, L_phi, Kuu, Kphi, _ = self.variational_state(C_train)
        H = self.cfg.H
        a, b, rho = self.factor_shares()
        mean = torch.zeros(C_query.shape[0], H)
        var = torch.zeros(C_query.shape[0], H)
        for h in range(H):
            G, r_diag = self._factor_conditional(C_query, h, a, b, Kphi[h], L_phi[h], C_train)
            mean[:, h] = G @ m_v[self._idx_F(h)]
            block = S_v[self._idx_F(h), self._idx_F(h)]
            var[:, h] = r_diag + torch.einsum("nm,ml,nl->n", G, block, G)
        return mean, var

    @torch.no_grad()
    def realized_training_factors(self, C_train):
        """Posterior mean/covariance of the realized training factors F-tilde_t (part of v directly)."""
        m_v, S_v, *_ = self.variational_state(C_train)
        H, T = self.cfg.H, self.cfg.T
        mean = torch.stack([m_v[self._idx_F(h)] for h in range(H)], dim=1)  # (T,H)
        cov = torch.zeros(T, H, H)
        for h in range(H):
            for hp in range(H):
                sub = S_v[self._idx_F(h), :][:, self._idx_F(hp)]
                cov[:, h, hp] = sub.diagonal()
        return mean, cov

    @torch.no_grad()
    def innovation_covariances(self):
        """D_beta, D_phi as (H,H) diagonal matrices in the model's own factor scale."""
        D_beta = torch.diag(self.tau_beta())
        _, _, rho = self.factor_shares()
        D_phi = torch.diag(rho)
        return D_beta, D_phi

    @torch.no_grad()
    def cross_section_predictive(self, X_new, c_new_row, C_train):
        """Predictive mean vector and covariance matrix for one new cross-section that
        shares a single new context c_new_row (Eq. 24-27, B.9), used for GMV/tangency/
        log-score evaluation. Returns (mu, Sigma) in scaled-return units."""
        H = self.cfg.H
        m_v, S_v, L_beta, L_phi, Kuu, Kphi, _ = self.variational_state(C_train)
        a, b, rho = self.factor_shares()
        n = X_new.shape[0]

        m_beta = torch.zeros(n, H)
        G_beta = []
        Sbb = torch.zeros(n, n, H, H)  # [S_star]_{beta_j beta_l}, pre-innovation
        for h in range(H):
            G, r_diag = self._loading_conditional(X_new, h, Kuu[h], L_beta[h])
            G_beta.append(G)
            m_beta[:, h] = G @ m_v[self._idx_U(h)]
        for h in range(H):
            for hp in range(H):
                block = S_v[self._idx_U(h), self._idx_U(hp)]
                Sbb[:, :, h, hp] = G_beta[h] @ block @ G_beta[hp].T
        for h in range(H):
            G, r_diag = self._loading_conditional(X_new, h, Kuu[h], L_beta[h])
            Sbb[:, :, h, h] += torch.diag(r_diag)
        D_beta, D_phi = self.innovation_covariances()
        Sbb_tilde = Sbb.clone()
        diag_idx = torch.arange(n)
        Sbb_tilde[diag_idx, diag_idx, :, :] += D_beta

        c_new = c_new_row.unsqueeze(0)
        m_phi = torch.zeros(H)
        G_phi = []
        Spp = torch.zeros(H, H)
        for h in range(H):
            G, r_diag = self._factor_conditional(c_new, h, a, b, Kphi[h], L_phi[h], C_train)
            G_phi.append(G)  # (1,T)
            m_phi[h] = (G @ m_v[self._idx_F(h)])[0]
            Spp[h, h] += r_diag[0]
        for h in range(H):
            for hp in range(H):
                block = S_v[self._idx_F(h), self._idx_F(hp)]
                Spp[h, hp] += (G_phi[h] @ block @ G_phi[hp].T)[0, 0]
        Spp_tilde = Spp + D_phi

        C_cross = torch.zeros(n, H, H)  # Cov(beta_j_h, phi_hp), no innovation cross term
        for h in range(H):
            for hp in range(H):
                block = S_v[self._idx_U(h), self._idx_F(hp)]
                C_cross[:, h, hp] = (G_beta[h] @ block @ G_phi[hp].T).squeeze(-1)

        term1 = torch.einsum("jh,hk,lk->jl", m_beta, Spp_tilde, m_beta)
        term2 = torch.einsum("h,jlhk,k->jl", m_phi, Sbb_tilde, m_phi)
        term3 = torch.einsum("jh,lhk,k->jl", m_beta, C_cross, m_phi)
        term4 = torch.einsum("h,jhk,lk->jl", m_phi, C_cross, m_beta)
        term5 = torch.einsum("jlhk,hk->jl", Sbb_tilde, Spp_tilde)
        term6 = torch.einsum("lhk,jhk->jl", C_cross, C_cross)
        sigma_r2 = self.sigma_r2()
        Sigma = term1 + term2 + term3 + term4 + term5 + term6
        Sigma = Sigma + sigma_r2 * torch.eye(n)
        Sigma = 0.5 * (Sigma + Sigma.T)

        mu = torch.einsum("jh,h->j", m_beta, m_phi) + torch.einsum("jhh->j", C_cross)
        return mu, Sigma

    @torch.no_grad()
    def variance_decomposition(self, X_new, c_new_row, C_train):
        """V^0, V^phi, V^beta, tr(D_beta D_phi), sigma_r^2 per stock (Eq. 26-27)."""
        H = self.cfg.H
        m_v, S_v, L_beta, L_phi, Kuu, Kphi, _ = self.variational_state(C_train)
        a, b, rho = self.factor_shares()
        n = X_new.shape[0]
        m_beta = torch.zeros(n, H)
        Sbb_self = torch.zeros(n, H, H)
        G_beta = []
        for h in range(H):
            G, r_diag = self._loading_conditional(X_new, h, Kuu[h], L_beta[h])
            G_beta.append(G)
            m_beta[:, h] = G @ m_v[self._idx_U(h)]
            Sbb_self[:, h, h] += r_diag
        for h in range(H):
            for hp in range(H):
                block = S_v[self._idx_U(h), self._idx_U(hp)]
                Sbb_self[:, h, hp] += torch.einsum("nm,ml,nl->n", G_beta[h], block, G_beta[hp])

        c_new = c_new_row.unsqueeze(0)
        m_phi = torch.zeros(H)
        Spp = torch.zeros(H, H)
        G_phi = []
        for h in range(H):
            G, r_diag = self._factor_conditional(c_new, h, a, b, Kphi[h], L_phi[h], C_train)
            G_phi.append(G)
            m_phi[h] = (G @ m_v[self._idx_F(h)])[0]
            Spp[h, h] += r_diag[0]
        for h in range(H):
            for hp in range(H):
                block = S_v[self._idx_F(h), self._idx_F(hp)]
                Spp[h, hp] += (G_phi[h] @ block @ G_phi[hp].T)[0, 0]

        D_beta, D_phi = self.innovation_covariances()
        V0 = torch.einsum("jh,hk,jk->j", m_beta, Spp, m_beta) + torch.einsum("jhk,hk->j", Sbb_self, Spp)
        V_phi = torch.einsum("jh,hk,jk->j", m_beta, D_phi, m_beta) + torch.einsum("jhk,kh->j", Sbb_self, D_phi)
        V_beta = (m_phi @ D_beta @ m_phi) + torch.trace(D_beta @ Spp)
        tr_DbDp = torch.trace(D_beta @ D_phi)
        sigma_r2 = self.sigma_r2()
        return V0, V_phi, V_beta.expand(n), tr_DbDp.expand(n), sigma_r2.expand(n)

    @torch.no_grad()
    def bilinear_state(self, x_j: torch.Tensor, c_new_row: torch.Tensor, C_train: torch.Tensor):
        """Mean m and covariance S_tilde of the stacked state z=(beta_j, phi_t)
        in R^{2H} for one stock-month (Appendix B.3, Eq. B.13): the posterior
        covariance of (beta_j, phi_t) under q(omega), with the innovation
        variances D_beta, D_phi added back to the respective diagonal blocks
        (Eq. 25, so that S_tilde matches the generative, not the posterior,
        second moment used in the predictive density)."""
        H = self.cfg.H
        m_v, S_v, L_beta, L_phi, Kuu, Kphi, _ = self.variational_state(C_train)
        a, b, rho = self.factor_shares()
        X_new = x_j.unsqueeze(0)
        c_new = c_new_row.unsqueeze(0)

        m_beta = torch.zeros(H)
        m_phi = torch.zeros(H)
        G_beta, G_phi = [], []
        for h in range(H):
            G, _ = self._loading_conditional(X_new, h, Kuu[h], L_beta[h])
            G_beta.append(G)
            m_beta[h] = (G @ m_v[self._idx_U(h)])[0]
        for h in range(H):
            G, _ = self._factor_conditional(c_new, h, a, b, Kphi[h], L_phi[h], C_train)
            G_phi.append(G)
            m_phi[h] = (G @ m_v[self._idx_F(h)])[0]

        S_tilde = torch.zeros(2 * H, 2 * H)
        for h in range(H):
            for hp in range(H):
                S_tilde[h, hp] = (G_beta[h] @ S_v[self._idx_U(h), self._idx_U(hp)] @ G_beta[hp].T)[0, 0]
                S_tilde[H + h, H + hp] = (G_phi[h] @ S_v[self._idx_F(h), self._idx_F(hp)] @ G_phi[hp].T)[0, 0]
                cross = (G_beta[h] @ S_v[self._idx_U(h), self._idx_F(hp)] @ G_phi[hp].T)[0, 0]
                S_tilde[h, H + hp] = cross
                S_tilde[H + hp, h] = cross
        D_beta, D_phi = self.innovation_covariances()
        for h in range(H):
            S_tilde[h, h] += self._loading_conditional(X_new, h, Kuu[h], L_beta[h])[1][0] + D_beta[h, h]
            S_tilde[H + h, H + h] += self._factor_conditional(c_new, h, a, b, Kphi[h], L_phi[h], C_train)[1][0] + D_phi[h, h]
        m = torch.cat([m_beta, m_phi])
        return m, S_tilde, self.sigma_r2()


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p / (1 - p))


def fit_panelgp(
    model: PanelGPModel,
    X_train: torch.Tensor,
    t_idx_train: torch.Tensor,
    R_train: torch.Tensor,
    C_train: torch.Tensor,
    epochs: int,
    batch_size: int = 2048,
    lr_variational: float = 0.01,
    lr_model: float = 0.005,
    n_factor_samples: int = 64,
    grad_clip: float = 1.0,
    lr_halving_fractions=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
    seed: int = 0,
    verbose: bool = False,
):
    """Minibatch Adam training of the ELBO (Appendix C.4)."""
    torch.manual_seed(seed)
    variational_params = [model.m_omega, model.raw_L_omega]
    model_params = [p for n, p in model.named_parameters() if n not in ("m_omega", "raw_L_omega")]
    opt = torch.optim.Adam([
        {"params": variational_params, "lr": lr_variational},
        {"params": model_params, "lr": lr_model},
    ])
    halving_epochs = {int(f * epochs) for f in lr_halving_fractions}
    N = X_train.shape[0]
    history = []
    for epoch in range(epochs):
        if epoch in halving_epochs:
            for group in opt.param_groups:
                group["lr"] *= 0.5
        perm = torch.randperm(N)
        epoch_elbo = 0.0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            elbo, ll, kl = model.elbo(
                X_train[idx], t_idx_train[idx], R_train[idx], N, C_train,
                n_factor_samples=n_factor_samples,
            )
            loss = -elbo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_elbo += elbo.item()
        history.append(epoch_elbo)
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f"epoch {epoch}: elbo_sum={epoch_elbo:.2f}")
    return history
