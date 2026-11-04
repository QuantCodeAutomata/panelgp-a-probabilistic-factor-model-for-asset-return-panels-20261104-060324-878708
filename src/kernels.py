"""Elementary kernels of the PanelGP model (paper Section 3.3 and Appendix C.2).

Custom implementation: these three kernels and their combinations are
specified directly by the paper's equations and are not a standard library
preset, so they are written out here rather than taken from a GP toolbox.
"""

import torch


def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, lengthscales: torch.Tensor) -> torch.Tensor:
    """ARD RBF kernel, Equation (C.2): exp(-0.5 * sum_p (x1_p - x2_p)^2 / l_p^2).

    Args:
        x1: (n1, P) tensor.
        x2: (n2, P) tensor.
        lengthscales: (P,) positive tensor.
    Returns:
        (n1, n2) kernel matrix.
    """
    diff = (x1.unsqueeze(1) - x2.unsqueeze(0)) / lengthscales  # (n1, n2, P)
    sq_dist = (diff ** 2).sum(-1)
    return torch.exp(-0.5 * sq_dist)


def affine_kernel(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Affine kernel 1 + x^T x' (paper Section 3.3 and Appendix C.2)."""
    return 1.0 + x1 @ x2.T


def constant_kernel(n1: int, n2: int, device=None, dtype=None) -> torch.Tensor:
    """Constant kernel equal to 1 everywhere (paper Section 3.3)."""
    return torch.ones(n1, n2, device=device, dtype=dtype)
