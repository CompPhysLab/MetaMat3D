"""Run a Torch-backed simulation and persist its results."""

import json
import os

import numpy as np
import torch

from src.pytorch_version.config import SimulationConfig
from src.pytorch_version.geometry import build_geometry, orientation_normals
from src.pytorch_version.results import extinction, polarizability
from src.pytorch_version.solvers import fast, straight

_SOLVERS = {
    "straight": straight,
    "fast": fast,
}


def calculate(
    config: SimulationConfig,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Run the configured simulation on ``device`` and save compatible .npy files."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    if config.save_polarizability and config.H_0 == 0:
        raise ValueError("H_0 must be nonzero when saving polarizability")
    if config.save_extinction_cross_section:
        if config.H_0 == 0:
            raise ValueError(
                "H_0 must be nonzero when saving extinction cross section"
            )
        if config.k_direction is None:
            raise ValueError(
                "k_direction is required when saving extinction cross section"
            )

    print(f"Calculating system {config.name} on {target_device}")
    geometry = build_geometry(config, device=target_device)
    output_directory = os.path.join("DATA", config.name)
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file)

    solver = _SOLVERS[config.solver]
    currents = solver.solve_system(config, geometry, device=target_device)
    if config.save_currents:
        np.save(
            os.path.join(output_directory, "currents.npy"),
            currents.detach().cpu().numpy(),
        )
        print(f"Currents saved to {output_directory}/currents.npy")

    normals = orientation_normals(
        geometry.orientations, device=target_device
    )
    frequencies = torch.tensor(
        config.omegas, dtype=torch.float64, device=target_device
    )
    if config.save_polarizability:
        alpha = torch.stack(
            [
                polarizability(
                    currents[:, index],
                    config.radius,
                    normals,
                    config.H_0,
                    config.H_direction,
                )
                for index in range(frequencies.numel())
            ]
        )
        np.save(
            os.path.join(output_directory, "polarizability.npy"),
            alpha.detach().cpu().numpy(),
        )
        print(f"Polarizability saved to {output_directory}/polarizability.npy")

    if config.save_extinction_cross_section:
        cross_section = torch.stack(
            [
                extinction(
                    currents[:, index],
                    config.radius,
                    normals,
                    geometry.centers,
                    (config.Dx, config.Dy, config.Dz),
                    config.H_0,
                    config.H_direction,
                    config.k_direction,
                    frequencies[index],
                )
                for index in range(frequencies.numel())
            ]
        )
        np.save(
            os.path.join(output_directory, "extinction_cross_section.npy"),
            cross_section.detach().cpu().numpy(),
        )
        print(
            "Extinction cross section saved to "
            f"{output_directory}/extinction_cross_section.npy"
        )
