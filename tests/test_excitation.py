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
    spacings=(1.0, 1.0, 1.0),
    centers=None,
    normals=None,
):
    if centers is None:
        centers = [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
        ]

    if normals is None:
        normals = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]

    return {
        "centers": np.asarray(centers, dtype=np.int16),
        "spacings": np.asarray(spacings, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.int16),
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

def test_rhs_shape_dtype_and_finite_values():
    rhs = build_rhs(**make_inputs())

    assert rhs.shape == (3,)
    assert rhs.dtype == np.complex128
    assert np.all(np.isfinite(rhs))

def test_quasistatic_flux_depends_on_ring_orientation():
    inputs = make_inputs(
        H_0=2.0,
        H_direction=(0.0, 0.0, 1.0),
    )

    rhs = build_rhs(**inputs)

    expected_flux = mu_0 * np.pi * inputs["radius"] ** 2 * 2.0

    np.testing.assert_allclose(
        rhs,
        np.array([expected_flux, 0.0, 0.0], dtype=np.complex128),
    )

def test_quasistatic_flux_uses_projection_of_field_direction():
    inputs = make_inputs(
        H_0=3.0,
        H_direction=(1.0 / np.sqrt(3.0),) * 3,
        centers=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )

    rhs = build_rhs(**inputs)

    expected_flux = (
        mu_0 * np.pi * inputs["radius"] ** 2 * 3.0 / np.sqrt(3.0)
    )

    np.testing.assert_allclose(
        rhs,
        np.full(3, expected_flux, dtype=np.complex128),
    )

def test_zero_field_produces_zero_rhs():
    inputs = make_inputs(
        H_0=0.0,
        centers=[
            [0, 0, 0],
            [2, 4, 6],
        ],
        normals=[
            [0, 0, 1],
            [1, 0, 0],
        ],
    )

    rhs = build_rhs(**inputs)

    np.testing.assert_array_equal(
        rhs,
        np.zeros(2, dtype=np.complex128),
    )

def test_non_quasistatic_mode_applies_propagation_phase():
    omega = 2.0 * np.pi * 1e8
    spacings = (0.5, 1.0, 1.0)

    inputs = make_inputs(
        model="full_retarded",
        H_0=2.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        omega=omega,
        spacings=spacings,
        centers=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        normals=[
            [0, 0, 1],
            [0, 0, 1],
        ],
    )

    rhs = build_rhs(**inputs)

    amplitude = mu_0 * np.pi * inputs["radius"] ** 2 * inputs["H_0"]

    physical_x = (
        inputs["centers"][:, 0] * inputs["spacings"][0] / 2
    )
    phase = np.exp(1j * omega / c * physical_x)

    np.testing.assert_allclose(
        rhs,
        amplitude * phase,
    )

def test_non_quasistatic_phase_is_one_at_origin():
    inputs = make_inputs(
        model="full_retarded",
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        centers=[[0, 0, 0]],
        normals=[[0, 0, 1]],
    )

    rhs = build_rhs(**inputs)

    expected_flux = (
        mu_0 * np.pi * inputs["radius"] ** 2 * inputs["H_0"]
    )

    np.testing.assert_allclose(
        rhs,
        np.array([expected_flux], dtype=np.complex128),
    )

def test_ring_order_is_preserved():
    inputs = make_inputs(
        H_direction=(0.0, 0.0, 1.0),
        centers=[
            [0, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
        ],
        normals=[
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
    )

    rhs = build_rhs(**inputs)

    np.testing.assert_allclose(rhs[[0, 2]], 0.0)
    assert abs(rhs[1]) > 0.0

def test_zero_normal_produces_zero_projection():
    inputs = make_inputs(
        centers=[[0, 0, 0]],
        normals=[[0, 0, 0]],
    )

    rhs = build_rhs(**inputs)

    np.testing.assert_array_equal(
        rhs,
        np.zeros(1, dtype=np.complex128),
    )


@pytest.mark.parametrize("model", ["full_retarded", "approx_retarded"])
@pytest.mark.parametrize(
    ("k_direction", "omega"),
    [
        (None, 2.0 * np.pi * 1e8),
        ((1.0, 0.0, 0.0), None),
        (None, None),
    ],
)
def test_non_quasistatic_model_requires_wave_parameters(
    model, k_direction, omega
):
    inputs = make_inputs(
        model=model,
        k_direction=k_direction,
        omega=omega,
    )

    with pytest.raises((TypeError, ValueError)):
        build_rhs(**inputs)

def test_phase_uses_anisotropic_spacings():
    omega = 2.0 * np.pi * 1e8

    inputs = make_inputs(
        model="full_retarded",
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(0.0, 1.0, 0.0),
        omega=omega,
        spacings=(0.5, 2.0, 3.0),
        centers=[
            [0, 0, 0],
            [0, 1, 0],
        ],
        normals=[
            [0, 0, 1],
            [0, 0, 1],
        ],
    )

    rhs = build_rhs(**inputs)

    amplitude = (
        mu_0 * np.pi * inputs["radius"] ** 2 * inputs["H_0"]
    )
    physical_y = inputs["centers"][:, 1] * inputs["spacings"][1] / 2
    expected = amplitude * np.exp(1j * omega / c * physical_y)

    np.testing.assert_allclose(rhs, expected)