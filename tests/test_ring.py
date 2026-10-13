import numpy as np
import pytest

from src.metamaterial.physics.ring import (
    self_inductance,
    impedance,
    conductivity,
)

@pytest.fixture
def ring_parameters():
    return {
        "R": 2.5,
        "L": 1.2e-9,
        "C": 3.4e-12,
    }

def test_self_inductance_matches_legacy_formula(ring_parameters):
    omega = 2.0 * np.pi * 1e9
    R = ring_parameters["R"]
    L = ring_parameters["L"]
    C = ring_parameters["C"]
    expected = R / (1j * omega) - L + 1 / (omega**2 * C)
    result = self_inductance(omega, R, L, C)
    assert result == pytest.approx(expected)

def test_impedance_matches_legacy_formula(ring_parameters):
    omega = 2.0 * np.pi * 1e9
    R = ring_parameters["R"]
    L = ring_parameters["L"]
    C = ring_parameters["C"]
    expected = R - 1j * omega * L + 1j / (omega * C)
    result = impedance(omega, R, L, C)
    assert result == pytest.approx(expected)

def test_conductivity_is_inverse_impedance(ring_parameters):
    omega = 2.0 * np.pi * 1e9
    R = ring_parameters["R"]
    L = ring_parameters["L"]
    C = ring_parameters["C"]
    result = conductivity(omega, R, L, C)
    expected = 1 / impedance(omega, R, L, C)
    assert result == pytest.approx(expected)

def test_relation(ring_parameters):
    omega = 2.0 * np.pi * 1e9
    R = ring_parameters["R"]
    L = ring_parameters["L"]
    C = ring_parameters["C"]
    Z = impedance(omega, R, L, C)
    M = self_inductance(omega, R, L, C)
    assert Z == pytest.approx(1j * omega * M)

@pytest.mark.parametrize(
    "function",
    [
        self_inductance,
        impedance,
        conductivity,
    ],
)
def test_functions_support_frequency_arrays(function, ring_parameters):
    frequencies = np.array([1e9, 2e9, 3e9])
    omega = 2 * np.pi * frequencies
    result = function(
        omega,
        ring_parameters["R"],
        ring_parameters["L"],
        ring_parameters["C"],
    )
    assert result.shape == omega.shape
    assert np.all(np.isfinite(result))

@pytest.mark.parametrize(
    "function",
    [
        self_inductance,
        impedance,
        conductivity,
    ],
)
@pytest.mark.parametrize("omega", [0.0, np.array([1e9, 0.0, 2e9])])
def test_functions_reject_zero_frequency(function, ring_parameters, omega):
    with pytest.raises(ValueError, match="omega must be nonzero"):
        function(
            omega,
            ring_parameters["R"],
            ring_parameters["L"],
            ring_parameters["C"],
        )