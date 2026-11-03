"""Torch calculation of derived sample properties."""

import torch

Tensor = torch.Tensor
C = 3e8


def mag_moment(currents: Tensor, radius: float, normals: Tensor) -> Tensor:
    """Calculate the magnetic moment vector of the sample."""
    device = currents.device
    ring_areas = torch.pi * radius**2
    return torch.sum(
        currents.to(torch.complex128)[:, None]
        * normals.to(device=device, dtype=torch.float64).to(torch.complex128)
        * ring_areas,
        dim=0,
    )


def polarizability(
    currents: Tensor,
    radius: float,
    normals: Tensor,
    H_0: float,
    H_direction: tuple[float, float, float] | Tensor,
) -> Tensor:
    """Calculate magnetic polarizability as a complex scalar tensor."""
    if H_0 == 0:
        raise ValueError("H_0 must be nonzero to calculate polarizability")
    moment = mag_moment(currents, radius, normals)
    direction = torch.as_tensor(
        H_direction, dtype=torch.float64, device=currents.device
    ).to(torch.complex128)
    return moment @ direction / H_0


def extinction(
    currents: Tensor,
    radius: float,
    normals: Tensor,
    centers: Tensor,
    spacings: tuple[float, float, float] | Tensor,
    H_0: float,
    H_direction: tuple[float, float, float] | Tensor,
    k_direction: tuple[float, float, float] | Tensor,
    omega: float | Tensor,
) -> Tensor:
    """Calculate the extinction cross-section as a real scalar tensor."""
    if H_0 == 0:
        raise ValueError("H_0 must be nonzero to calculate extinction")

    device = currents.device
    center_positions = centers.to(device=device, dtype=torch.float64)
    scale = torch.as_tensor(spacings, dtype=torch.float64, device=device) / 2.0
    center_positions = center_positions * scale
    normal_values = normals.to(device=device, dtype=torch.float64)
    field_direction = torch.as_tensor(
        H_direction, dtype=torch.float64, device=device
    )
    wave_direction = torch.as_tensor(
        k_direction, dtype=torch.float64, device=device
    )
    frequencies = torch.as_tensor(omega, dtype=torch.float64, device=device)

    projection = normal_values @ field_direction
    phase_coordinate = center_positions @ wave_direction
    phase_angle = -frequencies / C * phase_coordinate
    phase = torch.polar(torch.ones_like(phase_angle), phase_angle)
    weighted_currents = (
        currents.to(torch.complex128)
        * (torch.pi * radius**2)
        * projection
        * phase
    )
    return (frequencies / (C * H_0) * weighted_currents.sum().imag).real
