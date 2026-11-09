"""Build the right-hand side of the linear system."""

from typing import Literal
import torch

Tensor = torch.Tensor
InteractionModel = Literal["quasistatic", "full_retarded", "approx_retarded"]

MU_0 = 4.0 * torch.pi * 1e-7
C = 3e8


def build_rhs(
    centers: Tensor,
    spacings: tuple[float, float, float] | Tensor,
    normals: Tensor,
    radius: float,
    H_0: float,
    H_direction: tuple[float, float, float] | Tensor,
    *,
    model: InteractionModel = "quasistatic",
    k_direction: tuple[float, float, float] | Tensor | None = None,
    omega: float | Tensor | None = None,
) -> Tensor:
    """Build the complex excitation vector on the centers' device."""
    device = centers.device
    centers_float = centers.to(torch.float64)
    normal_values = normals.to(device=device, dtype=torch.float64)
    field_direction = torch.as_tensor(
        H_direction, dtype=torch.float64, device=device
    )
    projections = normal_values @ field_direction
    flux = MU_0 * torch.pi * radius**2 * H_0 * projections

    if model == "quasistatic":
        phase = torch.ones_like(flux, dtype=torch.complex128)
    else:
        scale = torch.as_tensor(spacings, dtype=torch.float64, device=device) / 2.0
        wave_direction = torch.as_tensor(
            k_direction, dtype=torch.float64, device=device
        )
        phase_coordinate = (centers_float * scale) @ wave_direction
        frequency = torch.as_tensor(omega, dtype=torch.float64, device=device)
        phase_angle = frequency / C * phase_coordinate
        phase = torch.polar(torch.ones_like(phase_angle), phase_angle)

    return phase * flux.to(torch.complex128)