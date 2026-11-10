"""ЧЕРНОВИК, ПРОБЛЕМА С quad, стоит заглушка с интегрированием трапециями и эллиптическими интегралами"""

import torch
import numpy as np
from scipy import special

Tensor = torch.Tensor


def L_parallel(
    dx: float | Tensor,
    dy: float | Tensor,
    dz: float | Tensor,
    r1: float,
    r2: float,
    n: int = 256,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Calculates mutual inductance between two parallel-oriented rings."""
    dev = torch.device(device)
    dx_t = torch.as_tensor(dx, dtype=torch.float64, device=dev)
    dy_t = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dz_t = torch.as_tensor(dz, dtype=torch.float64, device=dev)

    alpha = 2 * torch.pi * torch.arange(n, dtype=torch.float64, device=dev) / n
    db = torch.sqrt(dx_t**2 + dy_t**2)

    cos_alpha = torch.cos(alpha)
    dp = torch.sqrt(r2**2 + db**2 + 2 * r2 * db * cos_alpha)

    kappa_sq = 4 * r1 * dp / ((dp + r1)**2 + dz_t**2)

    kappa_sq_np = kappa_sq.detach().cpu().numpy()
    K_val = special.ellipk(kappa_sq_np)
    E_val = special.ellipe(kappa_sq_np)

    K_t = torch.as_tensor(K_val, dtype=torch.float64, device=dev)
    E_t = torch.as_tensor(E_val, dtype=torch.float64, device=dev)

    kappa = torch.sqrt(kappa_sq)
    kappa_safe = torch.clamp(kappa, min=torch.finfo(torch.float64).tiny)

    A = (
        1.0
        / (2 * torch.pi)
        * torch.sqrt(r1 / dp)
        * ((2.0 / kappa_safe - kappa_safe) * K_t - 2.0 * E_t / kappa_safe)
    )
    integrand = A * r2 * (r2 + db * cos_alpha) / dp

    L = integrand.sum(dim=-1) * (2 * torch.pi / n)
    return L


def L_orthogonal(
    dx: float | Tensor,
    dy: float | Tensor,
    dz: float | Tensor,
    r1: float,
    r2: float,
    n: int = 256,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Calculates mutual inductance between two orthogonal-oriented rings."""
    dev = torch.device(device)
    dx_t = torch.as_tensor(dx, dtype=torch.float64, device=dev)
    dy_t = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dz_t = torch.as_tensor(dz, dtype=torch.float64, device=dev)

    alpha = 2 * torch.pi * torch.arange(n, dtype=torch.float64, device=dev) / n
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)

    dp = torch.sqrt((dx_t - r2 * sin_alpha)**2 + dy_t**2)
    kappa_sq = 4 * r1 * dp / ((dp + r1)**2 + (dz_t - r2 * cos_alpha)**2)

    kappa_sq_np = kappa_sq.detach().cpu().numpy()
    K_val = special.ellipk(kappa_sq_np)
    E_val = special.ellipe(kappa_sq_np)

    K_t = torch.as_tensor(K_val, dtype=torch.float64, device=dev)
    E_t = torch.as_tensor(E_val, dtype=torch.float64, device=dev)

    kappa = torch.sqrt(kappa_sq)
    kappa_safe = torch.clamp(kappa, min=torch.finfo(torch.float64).tiny)

    A = (
        1.0
        / (2 * torch.pi)
        * torch.sqrt(r1 / dp)
        * ((2.0 / kappa_safe - kappa_safe) * K_t - 2.0 * E_t / kappa_safe)
    )
    integrand = A * r2 * dy_t * cos_alpha / dp

    L = integrand.sum(dim=-1) * (2 * torch.pi / n)
    return L