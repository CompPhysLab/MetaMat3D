import numpy as np
import pytest
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry
from src.metamaterial.physics import build_rhs

Mu_0 = 4.0 * np.pi * 1e-7
C = 3e8

def make_config(
    *,
    interaction_model="quasistatic",
    H_0=2.0,
    H_direction=(0.0, 0.0, 1.0),
    k_direction=None,
    frequencies=None,
    radius=0.1,
):
    if frequencies is None:
        frequencies = np.array([1e8, 2e8])

    return SimulationConfig(
        name="excitation_test",
        R=1.0,
        L=1e-9,
        C=1e-12,
        radius=radius,
        width=0.01,
        orientations="zyx",
        type="center",
        Nx=1,
        Ny=1,
        Nz=1,
        Dx=2,
        Dy=2,
        Dz=2,
        packing="rectangle",
        interaction_model=interaction_model,
        H_0=H_0,
        H_direction=H_direction,
        k_direction=k_direction,
        frequencies=np.asarray(frequencies, dtype=np.float64),
        solver="straight",
    )

def make_geometry(centers, orientations):
    """Create a small geometry with a known ring ordering"""
    centers = np.asarray(centers, dtype=np.float64)
    orientations = np.asarray(orientations)

    return Geometry(
        centers=centers,
        orientations=orientations,
        grid_indices=np.zeros((len(orientations), 3), dtype=np.int64),
    )

def test_rhs_shape_and_complex_dtype():
    config = make_config()
    geometry = make_geometry(
        centers=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        orientations=["z", "x", "y"],
    )

    rhs = build_rhs(config, geometry)

    assert rhs.shape == (2, 3)
    assert rhs.dtype == np.complex128
    assert np.all(np.isfinite(rhs))

def test_quasistatic_flux_depends_on_ring_orientation():
    config = make_config(
        H_0=2.0,
        H_direction=(0.0, 0.0, 1.0),
    )
    geometry = make_geometry(
        centers=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        orientations=["z", "x", "y"],
    )

    rhs = build_rhs(config, geometry)

    expected_flux = Mu_0 * np.pi * config.radius**2 * 2.0

    np.testing.assert_allclose(
        rhs,
        np.array([
            [expected_flux, 0.0, 0.0],
            [expected_flux, 0.0, 0.0],
        ], dtype=np.complex128),
    )

def test_quasistatic_flux_uses_projection_of_H_direction():
    config = make_config(
        H_0=3.0,
        H_direction=(1.0, 1.0, 1.0),
    )
    geometry = make_geometry(
        centers=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        orientations=["x", "y", "z"],
    )

    rhs = build_rhs(config, geometry)

    # SimulationConfig normalizes H_direction.
    projection = 1.0 / np.sqrt(3.0)
    expected_flux = Mu_0 * np.pi * config.radius**2 * 3.0 * projection

    np.testing.assert_allclose(
        rhs,
        np.full((2, 3), expected_flux, dtype=np.complex128),
    )

def test_zero_field_produces_zero_rhs():
    config = make_config(H_0=0.0)
    geometry = make_geometry(
        centers=[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        orientations=["z", "x"],
    )

    rhs = build_rhs(config, geometry)

    np.testing.assert_array_equal(rhs, np.zeros((2, 2), dtype=np.complex128))

def test_non_quasistatic_mode_applies_propagation_phase():
    frequencies = np.array([1e8, 2e8])
    config = make_config(
        interaction_model="full_retarded",
        H_0=2.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        frequencies=frequencies,
    )
    geometry = make_geometry(
        centers=[
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        orientations=["z", "z"],
    )

    rhs = build_rhs(config, geometry)

    amplitude = Mu_0 * np.pi * config.radius**2 * config.H_0
    phase = np.exp(
        1j * frequencies[:, None] / C
        * np.array([[0.0, 0.25]])
    )
    expected = amplitude * phase

    np.testing.assert_allclose(rhs, expected)

def test_non_quasistatic_phase_is_one_at_origin():
    config = make_config(
        interaction_model="full_retarded",
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
    )
    geometry = make_geometry(
        centers=[[0.0, 0.0, 0.0]],
        orientations=["z"],
    )

    rhs = build_rhs(config, geometry)

    expected_flux = Mu_0 * np.pi * config.radius**2 * config.H_0

    np.testing.assert_allclose(
        rhs[:, 0],
        np.full(2, expected_flux, dtype=np.complex128),
    )

def test_ring_order_is_preserved():
    config = make_config(
        H_direction=(0.0, 0.0, 1.0),
    )
    geometry = make_geometry(
        centers=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        orientations=["x", "z", "y"],
    )

    rhs = build_rhs(config, geometry)

    # Only the second ring has a normal parallel to H.
    assert np.all(rhs[:, 0] == 0.0)
    assert np.all(np.abs(rhs[:, 1]) > 0.0)
    assert np.all(rhs[:, 2] == 0.0)

def test_invalid_orientation_raises_value_error():
    config = make_config()
    geometry = make_geometry(
        centers=[[0.0, 0.0, 0.0]],
        orientations=["invalid"],
    )

    with pytest.raises(ValueError):
        build_rhs(config, geometry)

def test_non_finite_center_raises_value_error():
    config = make_config()
    geometry = make_geometry(
        centers=[[np.nan, 0.0, 0.0]],
        orientations=["z"],
    )

    with pytest.raises(ValueError):
        build_rhs(config, geometry)