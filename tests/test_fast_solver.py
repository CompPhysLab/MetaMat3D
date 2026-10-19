import numpy as np
import pytest
from src.metamaterial.solvers import fast, straight
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import build_geometry
from src.metamaterial.physics import mutual_inductance_matrix
from src.metamaterial.physics.coupling.operator import build_fft_operator, _grid_spec, mutual_inductance

def make_config(model: str) -> SimulationConfig:
    """Small configuration for comparing direct and FFT-based solvers."""
    return SimulationConfig(
        name="test_solver",
        orientations="zyx",
        packing="rectangle",
        Nx=3,
        Ny=2,
        Nz=2,
        Dx=3e-3,
        Dy=3e-3,
        Dz=3e-3,
        radius=1e-3,
        width=1e-4,
        R=1.0,
        L=1e-6,
        C=1e-9,
        omegas=(1e8, 2e8),
        H_0=1.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        interaction_model=model,
        type="center",
        solver="straight",
    )

@pytest.mark.parametrize(
    "model",
    ["quasistatic", "full_retarded", "approx_retarded"],
)
def test_fast_solver_matches_straight(model):
    config = make_config(model)
    geometry = build_geometry(config)

    currents_straight = straight.solve_system(config, geometry)
    currents_fast = fast.solve_system(config, geometry)

    assert currents_fast.shape == currents_straight.shape
    assert np.all(np.isfinite(currents_fast))
    assert np.all(np.isfinite(currents_straight))

    diff = np.abs(currents_fast - currents_straight)
    scale = np.maximum(np.abs(currents_straight), 1e-30)
    relative_error = diff / scale

    idx = np.unravel_index(np.argmax(diff), diff.shape)

    print(f"\nModel: {model}")
    print(f"Max absolute error: {diff.max():.6e}")
    print(f"Max relative error: {relative_error.max():.6e}")
    print(f"Worst index: {idx}")
    print(f"fast:     {currents_fast[idx]}")
    print(f"straight: {currents_straight[idx]}")
    print(f"|fast|:   {abs(currents_fast[idx]):.6e}")
    print(f"|straight|: {abs(currents_straight[idx]):.6e}")

    np.testing.assert_allclose(
        currents_fast,
        currents_straight,
        rtol=1e-5,
        atol=1e-10,
    )