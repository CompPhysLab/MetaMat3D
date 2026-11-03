"""Direct Torch solver for the frequency-domain system."""

import torch
from tqdm import tqdm

from src.pytorch_version.config import SimulationConfig
from src.pytorch_version.geometry import Geometry, orientation_normals
from src.pytorch_version.physics import (
    build_rhs,
    mutual_inductance_matrix,
    self_inductance,
)

Tensor = torch.Tensor


def solve_system(
    config: SimulationConfig,
    geometry: Geometry,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Solve the dense linear system on the requested device."""
    target_device = (
        torch.device(device) if device is not None else geometry.centers.device
    )
    centers = geometry.centers.to(device=target_device)
    normals = orientation_normals(geometry.orientations, device=target_device)
    frequencies = torch.tensor(
        config.omegas, dtype=torch.float64, device=target_device
    )
    spacings = (config.Dx, config.Dy, config.Dz)

    currents = torch.empty(
        (centers.shape[0], frequencies.numel()),
        dtype=torch.complex128,
        device=target_device,
    )
    mutual = None
    if config.interaction_model == "quasistatic":
        mutual = mutual_inductance_matrix(
            centers,
            spacings,
            geometry.orientations,
            radius=config.radius,
            width=config.width,
            model=config.interaction_model,
        )

    for index, omega in enumerate(tqdm(frequencies, desc="Solving frequencies")):
        if mutual is None:
            mutual = mutual_inductance_matrix(
                centers,
                spacings,
                geometry.orientations,
                radius=config.radius,
                width=config.width,
                model=config.interaction_model,
                omega=float(omega.item()),
            )
        diagonal = self_inductance(
            omega, config.R, config.L, config.C, device=target_device
        )
        rhs = build_rhs(
            centers,
            spacings,
            normals,
            config.radius,
            config.H_0,
            config.H_direction,
            model=config.interaction_model,
            k_direction=config.k_direction,
            omega=omega,
        )
        system = torch.diag(
            torch.ones(
                centers.shape[0],
                dtype=torch.complex128,
                device=target_device,
            )
            * diagonal
        ) - mutual
        currents[:, index] = torch.linalg.solve(system, rhs)
        if config.interaction_model != "quasistatic":
            mutual = None

    return currents
