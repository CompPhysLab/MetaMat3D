"""Direct numerical integration (trapezoidal rule) for parallel and orthogonal rings."""

import torch

Tensor = torch.Tensor
c = 3e8

def L_parallel(
    omega: float | Tensor,
    dx: float,
    dy: float,
    dz: float,
    r1: float,
    r2: float,
    n: int = 128,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Calculates mutual inductance between two parallel-oriented rings using the trapezoidal rule"""
    dev = torch.device(device)
    k = omega / c if not isinstance(omega, Tensor) else omega.to(device=dev, dtype=torch.float64) / c
    
    phi = 2 * torch.pi * torch.arange(n, dtype=torch.float64, device=dev) / n
    phi1, phi2 = torch.meshgrid(phi, phi, indexing="ij")
    
    A = dz**2 + dx**2 + dy**2 + 2 * r2 * (dx * torch.cos(phi2) + dy * torch.sin(phi2)) + r1**2 + r2**2
    B = -2 * r1 * (dx + r2 * torch.cos(phi2))
    C = -2 * r1 * (dy + r2 * torch.sin(phi2))
    
    arg = A + B * torch.cos(phi1) + C * torch.sin(phi1)
    R = torch.sqrt(torch.clamp(arg, min=0.0))
    
    spr = torch.cos(phi2 - phi1)
    
    k_val = k if isinstance(k, Tensor) else torch.tensor(k, dtype=torch.float64, device=dev)
    phase_angle = k_val * R
    exp_term = torch.polar(torch.ones_like(phase_angle, dtype=torch.float64, device=dev), phase_angle)
    
    kernel = spr * exp_term / R
    L = kernel.sum() * torch.pi * r1 * r2 / (n**2)
    
    return L.to(torch.complex128)


def L_orthogonal(
    omega: float | Tensor,
    dx: float,
    dy: float,
    dz: float,
    r1: float,
    r2: float,
    n: int = 128,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Calculates mutual inductance between two orthogonal-oriented rings using the trapezoidal rule"""
    dev = torch.device(device)
    k = omega / c if not isinstance(omega, Tensor) else omega.to(device=dev, dtype=torch.float64) / c
    
    phi = 2 * torch.pi * torch.arange(n, dtype=torch.float64, device=dev) / n
    phi1, phi2 = torch.meshgrid(phi, phi, indexing="ij")
    
    A = dz**2 + dx**2 + dy**2 - 2 * r2 * (dx * torch.sin(phi2) + dz * torch.cos(phi2)) + r1**2 + r2**2
    B = -2 * dy * r1
    C = 2 * r1 * (r2 * torch.sin(phi2) - dx)
    
    arg = A + B * torch.cos(phi1) + C * torch.sin(phi1)
    R = torch.sqrt(torch.clamp(arg, min=0.0))
    
    spr = torch.cos(phi1) * torch.cos(phi2)
    
    k_val = k if isinstance(k, Tensor) else torch.tensor(k, dtype=torch.float64, device=dev)
    phase_angle = k_val * R
    exp_term = torch.polar(torch.ones_like(phase_angle, dtype=torch.float64, device=dev), phase_angle)
    
    kernel = spr * exp_term / R
    L = -kernel.sum() * torch.pi * r1 * r2 / (n**2)
    
    return L.to(torch.complex128)