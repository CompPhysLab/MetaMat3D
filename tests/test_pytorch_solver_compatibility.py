"""Compare Torch and NumPy/SciPy solver currents on CPU."""

import numpy as np
import pytest
import torch

from src.metamaterial.config import SimulationConfig as NumpyConfig
from src.metamaterial.geometry import build_geometry as build_numpy_geometry
from src.metamaterial.solvers import fast as numpy_fast
from src.metamaterial.solvers import straight as numpy_straight
from src.pytorch_version.config import SimulationConfig as TorchConfig
from src.pytorch_version.geometry import build_geometry as build_torch_geometry
from src.pytorch_version.solvers import fast as torch_fast
from src.pytorch_version.solvers import straight as torch_straight


MODELS = ("quasistatic", "full_retarded")
SOLVERS = (
    ("straight", numpy_straight.solve_system, torch_straight.solve_system),
    ("fast", numpy_fast.solve_system, torch_fast.solve_system),
)


def _config(config_type, model, solver):
    return config_type(
        name=f"solver_compatibility_{model}_{solver}",
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
        omegas=np.array([2.0 * np.pi * 1e8, 2.0 * np.pi * 1.1e8]),
        solver=solver,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("solver_name,numpy_solver,torch_solver", SOLVERS)
def test_torch_solver_matches_numpy_backend(
    model, solver_name, numpy_solver, torch_solver
):
    numpy_config = _config(NumpyConfig, model, solver_name)
    torch_config = _config(TorchConfig, model, solver_name)
    numpy_geometry = build_numpy_geometry(numpy_config)
    torch_geometry = build_torch_geometry(torch_config, device="cpu")

    expected = numpy_solver(numpy_config, numpy_geometry)
    actual = torch_solver(torch_config, torch_geometry, device="cpu")

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.complex128
    np.testing.assert_allclose(
        actual.detach().numpy(),
        expected,
        rtol=2e-8,
        atol=1e-13,
        err_msg=(
            f"Torch {solver_name} solver differs from NumPy backend "
            f"for {model} interaction"
        ),
    )
