import numpy as np
import pytest
from itertools import product
from src.metamaterial.physics.coupling.operator import (
    _grid_spec,
    _build_fft_kernel,
)
from src.metamaterial.geometry import build_geometry
from src.metamaterial.physics.coupling.operator import build_fft_operator, mutual_inductance_matrix, clear_cache
from src.metamaterial.config import SimulationConfig

@pytest.fixture(autouse=True)
def clear_mutual_inductance_cache():
    clear_cache()


def make_config(*, meta_atom_type, orientations):
    return SimulationConfig(
        name="test_fft",
        R=1.0,
        L=1e-9,
        C=1e-12,
        radius=1e-3,
        width=1e-4,
        Dx=3e-3,
        Dy=3e-3,
        Dz=3e-3,
        orientations=orientations,
        type=meta_atom_type,
        Nx=3,
        Ny=2,
        Nz=2,
        packing="rectangle",
        interaction_model="quasistatic",
        H_0=1.0,
        H_direction=(0.0, 0.0, 1.0),
        omegas=np.array([2 * np.pi * 1e9]),
        solver="straight",
    )

def test_fft_operator_columns_match_dense_matrix():
    config = make_config(
        meta_atom_type="center",
        orientations="zyx",
    )
    geometry = build_geometry(config)

    fft_operator = build_fft_operator(
        geometry.grid_indices,
        geometry.orientations,
        counts=(config.Nx, config.Ny, config.Nz),
        spacing=(config.Dx, config.Dy, config.Dz),
        type=config.type,
        radius=config.radius,
        width=config.width,
        model=config.interaction_model,
    )

    dense = mutual_inductance_matrix(
        geometry.centers,
        geometry.orientations,
        radius=config.radius,
        width=config.width,
        model=config.interaction_model,
    )

    n = len(geometry.centers)

    for j in range(n):
        basis = np.zeros(n, dtype=complex)
        basis[j] = 1.0

        actual = fft_operator @ basis
        expected = dense[:, j]

        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-15):
            i = np.argmax(np.abs(actual - expected))
            print(
                f"\nMismatch at matrix[{i}, {j}]: "
                f"FFT={actual[i]:.12e}, "
                f"dense={expected[i]:.12e}, "
                f"error={abs(actual[i] - expected[i]):.3e}"
            )
            pytest.fail("FFT operator does not reproduce dense matrix")

@pytest.mark.parametrize(
    ("meta_atom_type", "orientations"),
    [
        ("center", "x"),
        ("center", "zyx"),
        ("border", "zyx"),
        ("open", "zyx"),
        ("center", "xy"),
    ],
)

def test_fft_operator_matches_dense_matrix(meta_atom_type, orientations):
    config = make_config(
        meta_atom_type=meta_atom_type,
        orientations=orientations,
    )
    geometry = build_geometry(config)

    centers = geometry.centers
    ring_orientations = geometry.orientations
    grid_indices = geometry.grid_indices

    spacing = (config.Dx, config.Dy, config.Dz)
    counts = (config.Nx, config.Ny, config.Nz)

    fft_operator = build_fft_operator(
        grid_indices,
        ring_orientations,
        counts=counts,
        spacing=spacing,
        type=config.type,
        radius=config.radius,
        width=config.width,
        model=config.interaction_model,
    )

    dense_matrix = mutual_inductance_matrix(
        centers,
        ring_orientations,
        radius=config.radius,
        width=config.width,
        model=config.interaction_model,
    )

    rng = np.random.default_rng(42)
    currents = (
        rng.standard_normal(len(centers))
        + 1j * rng.standard_normal(len(centers))
    )

    actual = fft_operator @ currents
    expected = dense_matrix @ currents

    diff = np.abs(actual - expected)
    worst = np.argsort(diff)[-10:][::-1]

    print("\nLargest discrepancies:")
    for i in worst:
        print(
            f"ring={i}, "
            f"orientation={ring_orientations[i]}, "
            f"grid_index={grid_indices[i]}, "
            f"actual={actual[i]:.12e}, "
            f"expected={expected[i]:.12e}, "
            f"abs_error={diff[i]:.3e}"
        )

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-15)

def test_fft_kernel_matches_dense_matrix_elements():
    config = make_config(
        meta_atom_type="center",
        orientations="zyx",
    )
    geometry = build_geometry(config)

    centers = geometry.centers
    orientations = geometry.orientations
    grid_indices = geometry.grid_indices

    counts = (config.Nx, config.Ny, config.Nz)
    spacing = (config.Dx, config.Dy, config.Dz)

    dense = mutual_inductance_matrix(
        centers,
        orientations,
        radius=config.radius,
        width=config.width,
        model=config.interaction_model,
    )

    axes = tuple(
        axis for axis in "xyz"
        if np.any(orientations == axis)
    )

    specs = {
        axis: _grid_spec(axis, counts, config.type)
        for axis in axes
    }

    kernels = {}
    for target, source in product(axes, repeat=2):
        built = _build_fft_kernel(
            target,
            source,
            specs[target],
            specs[source],
            spacing=spacing,
            radius=config.radius,
            width=config.width,
            model=config.interaction_model,
            omega=None,
        )
        kernels[target, source] = np.fft.ifftn(
            built.spectrum,
            s=built.fft_shape,
        )

    max_error = 0.0
    worst = None

    for i in range(len(centers)):
        target = str(orientations[i])
        target_spec = specs[target]
        target_index = (
            grid_indices[i] - np.asarray(target_spec.start)
        )

        for j in range(len(centers)):
            if i == j:
                continue

            source = str(orientations[j])
            source_spec = specs[source]
            source_index = (
                grid_indices[j] - np.asarray(source_spec.start)
            )

            kernel_index = (
                np.asarray(source_spec.shape)
                - 1
                + target_index
                - source_index
            )

            actual = kernels[target, source][tuple(kernel_index)]
            expected = dense[i, j]
            error = abs(actual - expected)

            if error > max_error:
                max_error = error
                worst = (
                    i,
                    j,
                    target,
                    source,
                    tuple(grid_indices[i]),
                    tuple(grid_indices[j]),
                    actual,
                    expected,
                )

    print(f"\nMaximum element error: {max_error:.6e}")
    print(f"Worst pair: {worst}")

    assert max_error < 1e-15, (
        f"FFT kernel differs from dense matrix: {worst}"
    )