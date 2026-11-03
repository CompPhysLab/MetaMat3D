"""Torch electrical response of split-ring resonators."""

import torch

Tensor = torch.Tensor


def _frequency_tensor(
    omega: float | Tensor,
    device: torch.device | str | None,
) -> Tensor:
    if isinstance(omega, Tensor):
        target_device = torch.device(device) if device is not None else omega.device
        frequencies = omega.to(device=target_device, dtype=torch.float64)
    else:
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        frequencies = torch.as_tensor(omega, dtype=torch.float64, device=target_device)
    if torch.any(frequencies == 0):
        raise ValueError("omega must be nonzero")
    return frequencies


def self_inductance(
    omega: float | Tensor,
    R: float,
    L: float,
    C: float,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the effective self-inductance as a complex128 tensor."""
    frequencies = _frequency_tensor(omega, device)
    real = -L + 1.0 / (frequencies**2 * C)
    imag = -R / frequencies
    return torch.complex(real, imag)


def impedance(
    omega: float | Tensor,
    R: float,
    L: float,
    C: float,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the complex self-impedance as a complex128 tensor."""
    frequencies = _frequency_tensor(omega, device)
    real = torch.full_like(frequencies, R)
    imag = 1.0 / (frequencies * C) - frequencies * L
    return torch.complex(real, imag)


def conductivity(
    omega: float | Tensor,
    R: float,
    L: float,
    C: float,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the complex conductivity ``1 / Z(omega)``."""
    return torch.reciprocal(impedance(omega, R, L, C, device=device))
