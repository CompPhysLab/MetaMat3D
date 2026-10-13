import numpy as np
import pytest
from src.metamaterial.physics.coupling import operator

@pytest.fixture(autouse=True)
def clear_operator_cache():
    operator.clear_cache()
    yield
    operator.clear_cache()

# Coordinate transformations
@pytest.mark.parametrize(
    ("orientation1", "orientation2", "expected"),
    [
        ("z", "z", (True, 1.0, 2.0, 3.0)),
        ("y", "y", (True, 1.0, -3.0, 2.0)),
        ("x", "x", (True, -3.0, 2.0, 1.0)),
        ("z", "y", (False, 1.0, 2.0, 3.0)),
        ("z", "x", (False, 2.0, 1.0, 3.0)),
        ("y", "z", (False, 1.0, 3.0, 2.0)),
        ("y", "x", (False, -3.0, 1.0, 2.0)),
        ("x", "z", (False, 2.0, 3.0, 1.0)),
        ("x", "y", (False, 3.0, 2.0, 1.0)),
    ],
)
def test_transform_coordinates(orientation1, orientation2, expected):
    result = operator._transform_coordinates(
        1.0, 2.0, 3.0, orientation1, orientation2
    )
    assert result == expected

# Model selection and argument forwarding
@pytest.mark.parametrize(
    ("model", "expected_args"),
    [
        ("quasistatic", (1.0, 2.0, 3.0, 4.0, 5.0)),
        ("full_retarded", (10.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
        ("approx_retarded", (10.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
    ],
)
def test_mutual_inductance_selects_model(monkeypatch, model, expected_args):
    calls = []

    def fake_function(*args):
        calls.append(args)
        return 2.5 + 0.5j

    module = operator._MODELS[model]
    monkeypatch.setattr(module, "L_parallel", fake_function)

    result = operator._mutual_inductance(
        model=model,
        omega=10.0,
        dx=1.0,
        dy=2.0,
        dz=3.0,
        r1=4.0,
        r2=5.0,
        orientation1="z",
        orientation2="z",
    )

    assert result == 2.5 + 0.5j
    assert calls == [expected_args]

def test_mutual_inductance_selects_orthogonal_function(monkeypatch):
    calls = []

    def fake_parallel(*args):
        pytest.fail("Parallel function should not be called")

    def fake_orthogonal(*args):
        calls.append(args)
        return 3.0

    monkeypatch.setattr(
        operator.quasistatic, "L_parallel", fake_parallel
    )
    monkeypatch.setattr(
        operator.quasistatic, "L_orthogonal", fake_orthogonal
    )

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        dx=1.0,
        dy=2.0,
        dz=3.0,
        r1=4.0,
        r2=5.0,
        orientation1="z",
        orientation2="y",
    )

    assert result == 3.0
    assert calls == [(1.0, 2.0, 3.0, 4.0, 5.0)]

# Finite-width averaging
@pytest.mark.parametrize(
    ("model", "omega"),
    [
        ("quasistatic", None),
        ("full_retarded", 10.0),
        ("approx_retarded", 10.0),
    ],
)
def test_equal_radii_width_averaging(monkeypatch, model, omega):
    calls = []

    def fake_function(*args):
        r1, r2 = args[-2:]
        calls.append((r1, r2))
        return 10.0 * r1 + r2

    module = operator._MODELS[model]
    monkeypatch.setattr(module, "L_parallel", fake_function)

    result = operator._mutual_inductance(
        model=model,
        omega=omega,
        dx=1.0,
        dy=2.0,
        dz=3.0,
        r1=5.0,
        r2=5.0,
        orientation1="z",
        orientation2="z",
        width=2.0,
    )

    delta = 2.0 * operator.gamma / 2
    inner = 5.0 - delta
    outer = 5.0 + delta

    assert calls == [
        (inner, inner),
        (inner, outer),
        (outer, outer),
    ]

    expected = (
        (10.0 * inner + inner)
        + 2 * (10.0 * inner + outer)
        + (10.0 * outer + outer)
    ) / 4

    assert result == pytest.approx(expected)

def test_unequal_radii_width_averaging(monkeypatch):
    calls = []

    def fake_function(*args):
        r1, r2 = args[-2:]
        calls.append((r1, r2))
        return 10.0 * r1 + r2

    monkeypatch.setattr(
        operator.quasistatic, "L_parallel", fake_function
    )

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        dx=1.0,
        dy=2.0,
        dz=3.0,
        r1=3.0,
        r2=5.0,
        orientation1="z",
        orientation2="z",
        width=0.4,
    )

    assert calls == [
        (3.2, 5.0),
        (2.8, 5.0),
    ]
    assert result == pytest.approx(
        ((10.0 * 3.2 + 5.0) + (10.0 * 2.8 + 5.0)) / 2
    )


def test_zero_width_uses_original_radii(monkeypatch):
    calls = []

    def fake_function(*args):
        calls.append(args)
        return 7.0

    monkeypatch.setattr(
        operator.quasistatic, "L_parallel", fake_function
    )

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        dx=1.0,
        dy=2.0,
        dz=3.0,
        r1=4.0,
        r2=5.0,
        orientation1="z",
        orientation2="z",
        width=0.0,
    )

    assert result == 7.0
    assert calls == [(1.0, 2.0, 3.0, 4.0, 5.0)]

# Public API: mu_0, coincident centers and cache
def test_public_function_applies_mu0_once(monkeypatch):
    monkeypatch.setattr(
        operator,
        "_cached_mutual_inductance",
        lambda *args: 2.0 + 3.0j,
    )

    result = operator.mutual_inductance(
        displacement=(1.0, 2.0, 3.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="z",
        orientation2="z",
    )

    assert result == pytest.approx(operator.mu_0 * (2.0 + 3.0j))

def test_coincident_centers_return_zero_without_model_call(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Model must not be called for coincident centers")

    monkeypatch.setattr(
        operator, "_cached_mutual_inductance", fail_if_called
    )

    result = operator.mutual_inductance(
        displacement=(0.0, 0.0, 0.0),
        r1=4.0,
        r2=5.0,
        width=0.2,
        orientation1="z",
        orientation2="y",
    )

    assert result == 0.0 + 0.0j

def test_cached_function_reuses_identical_inputs(monkeypatch):
    calls = []

    def fake_function(*args):
        calls.append(args)
        return 2.0

    monkeypatch.setattr(
        operator.quasistatic, "L_parallel", fake_function
    )

    kwargs = dict(
        displacement=(1.0, 2.0, 3.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="z",
        orientation2="z",
    )

    first = operator.mutual_inductance(**kwargs)
    second = operator.mutual_inductance(**kwargs)

    assert first == second
    assert len(calls) == 1

# Matrix assembly
def test_matrix_is_symmetric_and_has_zero_diagonal(monkeypatch):
    def fake_mutual_inductance(*, displacement, **kwargs):
        return complex(sum(displacement), 1.0)

    monkeypatch.setattr(
        operator, "mutual_inductance", fake_mutual_inductance
    )

    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    orientations = np.array(["z", "y", "x"])

    matrix = operator.mutual_inductance_matrix(
        centers,
        orientations,
        radius=1.0,
    )

    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(matrix), 0.0)
    np.testing.assert_allclose(matrix, matrix.T)

def test_matrix_empty_input():
    matrix = operator.mutual_inductance_matrix(
        np.empty((0, 3)),
        np.empty((0,), dtype=str),
        radius=1.0,
    )

    assert matrix.shape == (0, 0)

@pytest.mark.parametrize(
    ("model", "omega"),
    [
        ("quasistatic", None),
        ("full_retarded", 1.0e8),
        ("approx_retarded", 1.0e8),
    ],
)
def test_real_model_returns_finite_mutual_inductance(model, omega):
    result = operator.mutual_inductance(
        displacement=(0.0, 0.0, 0.02),
        r1=0.005,
        r2=0.005,
        width=0.0,
        orientation1="z",
        orientation2="z",
        model=model,
        omega=omega,
    )

    assert np.isfinite(result.real)
    assert np.isfinite(result.imag)
