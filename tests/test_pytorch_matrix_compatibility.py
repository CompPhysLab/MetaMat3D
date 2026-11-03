"""Compare Torch matrices and excitations with the NumPy/SciPy backend."""

import numpy as np
import pytest
import torch

from src.metamaterial.config import SimulationConfig as NumpyConfig
from src.metamaterial.geometry import (
    build_geometry as build_numpy_geometry,
    orientation_normals as numpy_orientation_normals,
)
from src.metamaterial.physics import (
    build_rhs as build_numpy_rhs,
    mutual_inductance_matrix as build_numpy_matrix,
)
from src.pytorch_version.config import SimulationConfig as TorchConfig
from src.pytorch_version.geometry import (
    build_geometry as build_torch_geometry,
    orientation_normals as torch_orientation_normals,
)
from src.pytorch_version.physics import (
    build_rhs as build_torch_rhs,
    mutual_inductance_matrix as build_torch_matrix,
)


MODELS = ("quasistatic", "full_retarded")


def _config(config_type, model):
    return config_type(
        name=f"matrix_compatibility_{model}",
        R=0.2,
        L=1e-9,
        C=1e-12,
        radius=0.5e-3,
        width=0.1e-3,
        Dx=3e-3,
        Dy=3e-3,
        Dz=3e-3,
        orientations="zyx",
        type="center",
        Nx=2,
        Ny=2,
        Nz=1,
        packing="rectangle",
        interaction_model=model,
        H_0=1.3,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        omegas=np.array([2.0 * np.pi * 1e8]),
        solver="straight",
    )


@pytest.mark.parametrize("model", MODELS)
def test_torch_mutual_inductance_matrix_matches_numpy_backend(model):
    numpy_config = _config(NumpyConfig, model)
    torch_config = _config(TorchConfig, model)
    numpy_geometry = build_numpy_geometry(numpy_config)
    torch_geometry = build_torch_geometry(torch_config, device="cpu")
    omega = None if model == "quasistatic" else float(numpy_config.omegas[0])

    numpy_matrix = build_numpy_matrix(
        numpy_geometry.centers,
        np.array([numpy_config.Dx, numpy_config.Dy, numpy_config.Dz]),
        numpy_geometry.orientations,
        radius=numpy_config.radius,
        width=numpy_config.width,
        model=model,
        omega=omega,
    )
    torch_matrix = build_torch_matrix(
        torch_geometry.centers,
        (torch_config.Dx, torch_config.Dy, torch_config.Dz),
        torch_geometry.orientations,
        radius=torch_config.radius,
        width=torch_config.width,
        model=model,
        omega=omega,
    )

    assert torch_matrix.device.type == "cpu"
    assert torch_matrix.dtype == torch.complex128
    np.testing.assert_allclose(
        torch_matrix.detach().numpy(),
        numpy_matrix,
        rtol=2e-10,
        atol=1e-19,
    )


@pytest.mark.parametrize("model", MODELS)
def test_torch_excitation_matches_numpy_backend(model):
    numpy_config = _config(NumpyConfig, model)
    torch_config = _config(TorchConfig, model)
    numpy_geometry = build_numpy_geometry(numpy_config)
    torch_geometry = build_torch_geometry(torch_config, device="cpu")
    numpy_normals = numpy_orientation_normals(numpy_geometry.orientations)
    torch_normals = torch_orientation_normals(
        torch_geometry.orientations, device="cpu"
    )
    omega = None if model == "quasistatic" else float(numpy_config.omegas[0])

    numpy_rhs = build_numpy_rhs(
        numpy_geometry.centers,
        np.array([numpy_config.Dx, numpy_config.Dy, numpy_config.Dz]),
        numpy_normals,
        numpy_config.radius,
        numpy_config.H_0,
        np.array(numpy_config.H_direction),
        model=model,
        k_direction=(
            None
            if numpy_config.k_direction is None
            else np.array(numpy_config.k_direction)
        ),
        omega=omega,
    )
    torch_rhs = build_torch_rhs(
        torch_geometry.centers,
        (torch_config.Dx, torch_config.Dy, torch_config.Dz),
        torch_normals,
        torch_config.radius,
        torch_config.H_0,
        torch_config.H_direction,
        model=model,
        k_direction=torch_config.k_direction,
        omega=omega,
    )

    assert torch_rhs.device.type == "cpu"
    assert torch_rhs.dtype == torch.complex128
    np.testing.assert_allclose(
        torch_rhs.detach().numpy(),
        numpy_rhs,
        rtol=1e-13,
        atol=1e-25,
    )
