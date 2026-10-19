import numpy as np
import pytest
from src.metamaterial.physics.excitation import build_rhs

mu_0 = 4.0 * np.pi * 1e-7
c = 3e8

def make_inputs(
    *,
    model="quasistatic",
    H_0=2.0,
    H_direction=(0.0, 0.0, 1.0),
    k_direction=None,
    omega=2.0 * np.pi * 1e8,
    radius=0.1,
    centers=None,
    normals=None,
):
    if centers is None:
        centers = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

    if normals is None:
        normals = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "normals": np.asarray(normals),
        "radius": radius,
        "H_0": H_0,
        "H_direction": np.asarray(H_direction, dtype=np.float64),
        "model": model,
        "k_direction": (
            None
            if k_direction is None
            else np.asarray(k_direction, dtype=np.float64)
        ),
        "omega": omega,
    }

def test_rhs_shape_and_complex_dtype():
    inputs = make_inputs()

    rhs = build_rhs(**inputs)

    assert rhs.shape == (3,)
    assert rhs.dtype == np.complex128
    assert np.all(np.isfinite(rhs))

def test_quasistatic_flux_depends_on_ring_orientation():
    inputs = make_inputs(
        H_0=2.0,
        H_direction=(0.0, 0.0, 1.0),
        normals = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
                ],
    )

    rhs = build_rhs(**inputs)

    expected_flux = mu_0 * np.pi * inputs["radius"]**2 * 2.0

    np.testing.assert_allclose(
        rhs,
        np.array([expected_flux, 0.0, 0.0], dtype=np.complex128),
    )

def test_quasistatic_flux_uses_projection_of_H_direction():
    inputs = make_inputs(
        H_0=3.0,
        H_direction=(1.0 / np.sqrt(3.0),) * 3,
        normals = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
                ],
        centers=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )

    rhs = build_rhs(**inputs)

    expected_flux = (
        mu_0 * np.pi * inputs["radius"]**2 * 3.0 / np.sqrt(3.0)
    )

    np.testing.assert_allclose(
        rhs,
        np.full(3, expected_flux, dtype=np.complex128),
    )

def test_zero_field_produces_zero_rhs():
    inputs = make_inputs(
        H_0=0.0,
        centers=[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        normals = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
                ],
    )

    rhs = build_rhs(**inputs)

    np.testing.assert_array_equal(
        rhs,
        np.zeros(2, dtype=np.complex128),
    )

def test_non_quasistatic_mode_applies_propagation_phase():
    omega = 2.0 * np.pi * 1e8
    inputs = make_inputs(
        model="full_retarded",
        H_0=2.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        omega=omega,
        centers=[
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        normals = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
                ],
    )

    rhs = build_rhs(**inputs)

    amplitude = mu_0 * np.pi * inputs["radius"]**2 * inputs["H_0"]
    phase = np.exp(
        1j * omega / c * np.array([0.0, 0.25])
    )
    expected = amplitude * phase

    np.testing.assert_allclose(rhs, expected)

def test_non_quasistatic_phase_is_one_at_origin():
    inputs = make_inputs(
        model="full_retarded",
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        omega=2.0 * np.pi * 1e8,
        centers=[[0.0, 0.0, 0.0]],
        normals = [
            [0.0, 0.0, 1.0],
                ],
    )

    rhs = build_rhs(**inputs)

    expected_flux = mu_0 * np.pi * inputs["radius"]**2 * inputs["H_0"]

    np.testing.assert_allclose(
        rhs,
        np.array([expected_flux], dtype=np.complex128),
    )

def test_ring_order_is_preserved():
    inputs = make_inputs(
        H_direction=(0.0, 0.0, 1.0),
        centers=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        normals = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
                ],
    )

    rhs = build_rhs(**inputs)

    assert rhs[0] == 0.0
    assert abs(rhs[1]) > 0.0
    assert rhs[2] == 0.0

def test_invalid_orientation_produces_zero_projection():
    inputs = make_inputs(
        centers=[[0.0, 0.0, 0.0]],
        normals=[[0.0, 0.0, 0.0]], #invalid
    )

    rhs = build_rhs(**inputs)
    np.testing.assert_array_equal(rhs, np.zeros(1, dtype=np.complex128))

@pytest.mark.parametrize("model", ["full_retarded", "approx_retarded"])
def test_non_quasistatic_model_requires_wave_parameters(model):
    inputs = make_inputs(
        model=model,
        k_direction=None,
        omega=None,
    )

    with pytest.raises((TypeError, ValueError)):
        build_rhs(**inputs)