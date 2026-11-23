from functools import lru_cache

import torch

Tensor = torch.Tensor
C = 3e8


class BarycentricInterpolator:
    """Evaluate a polynomial interpolant using the second barycentric formula."""
    def __init__(self, xi: Tensor, yi: Tensor, dim: int = -1):
        if xi.ndim != 1 or xi.numel() == 0:
            raise ValueError("xi must be a non-empty one-dimensional tensor")
        if yi.ndim == 0:
            raise ValueError("yi must have an interpolation dimension")

        self.xi = xi.detach().to(dtype=torch.float64)
        self.yi = yi
        self.dim = dim % yi.ndim
        if yi.shape[self.dim] != xi.numel():
            raise ValueError("xi and yi lengths do not match")
        if yi.device != xi.device:
            raise ValueError("xi and yi must be on the same device")

        differences = self.xi[:, None] - self.xi[None, :]
        differences.diagonal().fill_(1.0)
        if torch.any(differences == 0):
            raise ValueError("xi values must be distinct")

        log_weights = -torch.log(torch.abs(differences)).sum(dim=1)
        signs = torch.sign(differences).prod(dim=1)
        self.wi = signs * torch.exp(log_weights - log_weights.max())

    def __call__(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=torch.float64, device=self.xi.device)
        x_flat = x.reshape(-1)
        differences = x_flat[:, None] - self.xi[None, :]
        exact_match = differences == 0
        safe_differences = torch.where(
            exact_match, torch.ones_like(differences), differences
        )
        terms = self.wi[None, :] / safe_differences

        yi_shifted = torch.movedim(self.yi, self.dim, -1)
        terms_for_values = terms.to(dtype=yi_shifted.dtype)
        numerator = torch.matmul(yi_shifted, terms_for_values.mT)
        denominator = terms.sum(dim=1).to(dtype=yi_shifted.dtype)
        result = numerator / denominator

        exact_rows = exact_match.any(dim=1)
        if self.xi.device.type == "cpu":
            if exact_rows.any():
                node_indices = exact_match.to(torch.int64).argmax(dim=1)
                exact_values = yi_shifted[..., node_indices]
                mask = exact_rows.reshape((1,) * (yi_shifted.ndim - 1) + (-1,))
                result = torch.where(mask, exact_values, result)
        else:
            node_indices = exact_match.to(torch.int64).argmax(dim=1)
            exact_values = yi_shifted[..., node_indices]
            mask = exact_rows.reshape((1,) * (yi_shifted.ndim - 1) + (-1,))
            result = torch.where(mask, exact_values, result)

        result = result.reshape((*yi_shifted.shape[:-1], *x.shape))
        return torch.movedim(result, -1, self.dim) if x.ndim else result


@lru_cache(maxsize=None)
def diff_theta(N: int, device: torch.device = torch.device("cpu")) -> tuple[Tensor, Tensor]:
    """Computes the differentiation matrix and the Chebyshev nodes (PyTorch)."""
    theta = torch.pi * torch.arange(N + 1, dtype=torch.float64, device=device) / N
    u = torch.cos(theta)
    const = torch.ones(N + 1, dtype=torch.float64, device=device)
    const[0] = 2.0
    const[N] = 2.0
    const *= (-1.0) ** torch.arange(N + 1, dtype=torch.float64, device=device)
    X = u.repeat(N + 1, 1)
    dX = X.T - X
    D = torch.outer(const, 1.0 / const) / (dX + torch.eye(N + 1, dtype=torch.float64, device=device))
    D -= torch.diag(D.sum(dim=1))
    return u, D


def compute_ln_par(
    r_1: float,
    r_2: float,
    k: float,
    N_max: int,
    N_theta: int = 200,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, dict[int, Tensor]]:
    """Computes the Ln modes for parallel-oriented rings."""
    u, D = diff_theta(N_theta, device=device)
    D2 = D @ D
    B = torch.diag(1.0 - u**2) @ D2 - 2.0 * torch.diag(u) @ D
    sinth = torch.sqrt(torch.clamp(1.0 - u**2, min=0.0))

    j1_1 = torch.special.bessel_j1(k * r_1 * sinth)
    j1_2 = torch.special.bessel_j1(k * r_2 * sinth)

    L1 = torch.pi * r_1 * r_2 * j1_1 * j1_2
    Ln_data = {1: L1.to(torch.complex128)}

    for n in range(1, N_max):
        Ln = Ln_data[n]
        BLn = B.to(torch.complex128) @ Ln
        Ln_data[n + 1] = ((n - 1) * Ln + (1.0 / n) * BLn) / (2j * k)

    return u, Ln_data


def ln_interpolators_par(u_grid: Tensor, Ln_data: dict[int, Tensor]) -> dict[int, BarycentricInterpolator]:
    """Creates interpolators for the Ln modes of parallel-oriented rings."""
    interpolators = {}
    for n, vals in Ln_data.items():
        interpolators[n] = BarycentricInterpolator(u_grid, vals, dim=-1)
    return interpolators


def L_cached_par(
    N: int,
    r: float,
    k: float,
    theta: float | Tensor,
    ln_interps: dict[int, BarycentricInterpolator],
    device: torch.device,
) -> Tensor:
    """Computes total inductance for parallel-oriented rings."""
    theta_tensor = torch.as_tensor(theta, dtype=torch.float64, device=device)
    distance = torch.as_tensor(r, dtype=torch.float64, device=device)
    total = torch.zeros_like(distance, dtype=torch.complex128)
    exp_term = torch.polar(torch.ones_like(distance), k * distance)
    u = torch.cos(theta_tensor)

    for i in range(1, N + 1):
        val = ln_interps[i](u)
        total = total + val / distance**i * exp_term
    return total

def compute_Fnm_modes(
    r_1: float,
    r_2: float,
    k: float,
    N_max: int,
    N_theta: int = 200,
    M_modes: int = 5,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, dict[int, dict[int, Tensor]]]:
    """Computes the Fnm modes for orthogonal-oriented rings (Pure PyTorch)."""
    u, D = diff_theta(N_theta, device=device)
    N = N_theta
    D2 = D @ D
    sinth = torch.sqrt(torch.clamp(1.0 - u**2, min=0.0))
    cos_th = u
    N_phi = 2 * M_modes + 3

    phi_grid = (
        2.0
        * torch.pi
        * torch.arange(N_phi, dtype=torch.float64, device=device)
        / N_phi
    )
    L1_grid = torch.zeros((N + 1, N_phi), dtype=torch.complex128, device=device)

    for j, phi in enumerate(phi_grid):
        sin_phi = torch.sin(phi)
        sin2_th = sinth**2
        under = torch.sqrt(torch.clamp(1.0 - sin2_th * sin_phi**2, min=0.0))

        j1_1 = torch.special.bessel_j1(k * r_1 * sinth)
        j1_2 = torch.special.bessel_j1(k * r_2 * under)

        L1_grid[:, j] = -torch.pi * r_1 * r_2 * (sin_phi * cos_th / under) * j1_1 * j1_2

    L1_modes = torch.fft.fft(L1_grid, dim=1) / N_phi

    F_data = {}
    for m in range(-M_modes, M_modes + 1):
        abs_m = abs(m)
        B_m = (
            torch.diag(1.0 - u**2) @ D2
            - 2.0 * torch.diag(u * (abs_m + 1)) @ D
            - abs_m * (abs_m + 1) * torch.eye(N + 1, dtype=torch.float64, device=device)
        )
        idx = m if m >= 0 else N_phi + m
        L1_m = L1_modes[:, idx]
        w = (1.0 - u**2) ** (abs_m / 2.0)

        F1_m = torch.zeros(N + 1, dtype=torch.complex128, device=device)
        F1_m[1:-1] = L1_m[1:-1] / w[1:-1]

        if abs_m > 0:
            interp = BarycentricInterpolator(u[1:-1], F1_m[1:-1], dim=0)
            F1_m[0] = interp(u[0])
            F1_m[-1] = interp(u[-1])
        else:
            F1_m = L1_m.clone()

        F_data[m] = {1: F1_m}
        for n in range(1, N_max):
            Fn = F_data[m][n]
            BFn = B_m.to(torch.complex128) @ Fn
            F_data[m][n + 1] = ((n - 1) * Fn + (1.0 / n) * BFn) / (2j * k)

    return u, F_data


def make_F_interpolators(
    u_grid: Tensor,
    F_data: dict[int, dict[int, Tensor]],
) -> dict[int, dict[int, BarycentricInterpolator]]:
    """Creates interpolators for the Fnm modes using BarycentricInterpolator."""
    interpolators = {}
    for m, n_data in F_data.items():
        interpolators[m] = {}
        for n, Fn in n_data.items():
            interpolators[m][n] = BarycentricInterpolator(u_grid, Fn, dim=0)
    return interpolators


def Ln_at_point(
    interpolators: dict[int, dict[int, BarycentricInterpolator]],
    m_modes: int,
    theta: float | Tensor,
    phi: float | Tensor,
    n: int,
    device: torch.device,
) -> Tensor:
    """Evaluates the Ln mode at a specific point (theta, phi)."""
    th_tensor = torch.as_tensor(theta, dtype=torch.float64, device=device)
    phi_tensor = torch.as_tensor(phi, dtype=torch.float64, device=device)

    u = torch.cos(th_tensor)
    result = torch.zeros_like(u, dtype=torch.complex128)

    for m in range(-m_modes, m_modes + 1):
        abs_m = abs(m)
        F_val = interpolators[m][n](u)
        w = (1.0 - u**2) ** (abs_m / 2.0)
        phase = torch.polar(torch.ones_like(phi_tensor), m * phi_tensor)
        result = result + w * F_val * phase

    return result


def L_cached_ort(
    N: int,
    r: float,
    k: float,
    theta: float | Tensor,
    phi: float | Tensor,
    interpolators: dict[int, dict[int, BarycentricInterpolator]],
    m_modes: int,
    device: torch.device,
) -> Tensor:
    """Computes total inductance for orthogonal-oriented rings."""
    distance = torch.as_tensor(r, dtype=torch.float64, device=device)
    total = torch.zeros_like(distance, dtype=torch.complex128)
    exp_term = torch.polar(torch.ones_like(distance), k * distance)

    for n in range(1, N + 1):
        Ln = Ln_at_point(interpolators, m_modes, theta, phi, n, device=device)
        total = total + Ln / distance**n * exp_term

    return total

_recurrent_cache_par = {}
_recurrent_cache_ort = {}


def _get_recurrent_interps_par(r1: float, r2: float, k: float, N: int, device: torch.device) -> dict[int, BarycentricInterpolator]:
    key = (r1, r2, k, N, str(device))
    if key not in _recurrent_cache_par:
        u_grid, ln_data = compute_ln_par(r1, r2, k, N, device=device)
        _recurrent_cache_par[key] = ln_interpolators_par(u_grid, ln_data)
    return _recurrent_cache_par[key]


def _get_recurrent_interps_ort(r1: float, r2: float, k: float, N: int, M: int, device: torch.device) -> dict[int, dict[int, BarycentricInterpolator]]:
    key = (r1, r2, k, N, M, str(device))
    if key not in _recurrent_cache_ort:
        u_grid, F_data = compute_Fnm_modes(r1, r2, k, N, M_modes=M, device=device)
        _recurrent_cache_ort[key] = make_F_interpolators(u_grid, F_data)
    return _recurrent_cache_ort[key]


def L_parallel(
    omega: float,
    dx: float,
    dy: float,
    dz: float,
    r1: float,
    r2: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Computes the recurrent inductance for parallel rings (PyTorch)."""
    dev = torch.device(device)
    k = omega / C
    dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)
    dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dz = torch.as_tensor(dz, dtype=torch.float64, device=dev)
    dist = torch.sqrt(dx**2 + dy**2 + dz**2)
    db = torch.sqrt(dx**2 + dy**2)
    theta = torch.atan2(db, dz)
    N = 2
    ln_interps = _get_recurrent_interps_par(r1, r2, k, N, device=dev)
    return L_cached_par(N, dist, k, theta, ln_interps, device=dev)


def L_orthogonal(
    omega: float,
    dx: float,
    dy: float,
    dz: float,
    r1: float,
    r2: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Computes the recurrent inductance for orthogonal rings (PyTorch)."""
    dev = torch.device(device)
    k = omega / C
    dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)
    dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
    dz = torch.as_tensor(dz, dtype=torch.float64, device=dev)
    dist = torch.sqrt(dx**2 + dy**2 + dz**2)
    db = torch.sqrt(dx**2 + dy**2)
    theta = torch.atan2(db, dz)
    phi = torch.atan2(dy, dx)
    N = 2
    M = 5
    F_interps = _get_recurrent_interps_ort(r1, r2, k, N, M, device=dev)
    return L_cached_ort(N, dist, k, theta, phi, F_interps, M, device=dev)


def clear_static_cache():
    """Clear all cached data."""
    diff_theta.cache_clear()
    _recurrent_cache_par.clear()
    _recurrent_cache_ort.clear()
