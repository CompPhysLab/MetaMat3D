import numpy as np
import pytest
from src.metamaterial.physics.coupling import operator

@pytest.fixture(autouse=True)
def clear_cache():
    """Keep cache tests independent"""
    operator._cached_mutual_inductance.cache_clear()
    yield
    operator._cached_mutual_inductance.cache_clear()

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

    assert result[0] is expected[0]
    assert result[1:] == pytest.approx(expected[1:])

# Model dispatch and local coordinates
@pytest.mark.parametrize(
    ("model", "omega", "expected_args"),
    [
        ("quasistatic", None, (1.0, 2.0, 3.0, 4.0, 5.0)),
        (
            "full_retarded",
            10.0,
            (10.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        ),
        (
            "approx_retarded",
            10.0,
            (10.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        ),
    ],
)
def test_mutual_inductance_selects_model(
    monkeypatch, model, omega, expected_args
):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(("parallel", args))
            return 2.0

        @staticmethod
        def L_orthogonal(*args):
            calls.append(("orthogonal", args))
            return 3.0

    monkeypatch.setitem(operator._MODELS, model, FakeModel)

    result = operator._mutual_inductance(
        model=model,
        omega=omega,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=4.0,
        r2=5.0,
        parallel=True,
        width=0.0,
    )

    assert result == pytest.approx(2.0)
    assert calls == [("parallel", expected_args)]

def test_mutual_inductance_selects_orthogonal_function(monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append("parallel")
            return 2.0

        @staticmethod
        def L_orthogonal(*args):
            calls.append("orthogonal")
            return 3.0

    monkeypatch.setitem(operator._MODELS, "quasistatic", FakeModel)

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=4.0,
        r2=5.0,
        parallel=False,
        width=0.0,
    )

    assert result == pytest.approx(3.0)
    assert calls == ["orthogonal"]

def test_public_function_passes_transformed_coordinates(monkeypatch):
    captured = {}

    def fake_cached(
        model, omega, x, y, z, r1, r2, width, parallel
    ):
        captured.update(
            model=model,
            omega=omega,
            x=x,
            y=y,
            z=z,
            r1=r1,
            r2=r2,
            width=width,
            parallel=parallel,
        )
        return 1.0

    monkeypatch.setattr(
        operator, "_cached_mutual_inductance", fake_cached
    )

    operator.mutual_inductance(
        displacement=(1.0, 2.0, 3.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="y",
        orientation2="y",
    )

    assert captured == {
        "model": "quasistatic",
        "omega": None,
        "x": 1.0,
        "y": -3.0,
        "z": 2.0,
        "r1": 4.0,
        "r2": 5.0,
        "width": 0.0,
        "parallel": True,
    }

# Strip-width
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

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(args)
            return 10.0 * args[-2] + args[-1]

        @staticmethod
        def L_orthogonal(*args):
            raise AssertionError("Unexpected orthogonal call")

    monkeypatch.setitem(operator._MODELS, model, FakeModel)

    result = operator._mutual_inductance(
        model=model,
        omega=omega,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=4.0,
        r2=4.0,
        parallel=True,
        width=0.2,
    )

    delta = operator.gamma * 0.2 / 2.0
    r = 4.0 - delta
    R = 4.0 + delta

    expected = (
        (10.0 * r + r)
        + 2.0 * (10.0 * r + R)
        + (10.0 * R + R)
    ) / 4.0

    assert result == pytest.approx(expected)
    assert len(calls) == 3

def test_unequal_radii_width_averaging(monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(args)
            return 10.0 * args[-2] + args[-1]

        @staticmethod
        def L_orthogonal(*args):
            raise AssertionError("Unexpected orthogonal call")

    monkeypatch.setitem(operator._MODELS, "quasistatic", FakeModel)

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=3.0,
        r2=5.0,
        parallel=True,
        width=0.2,
    )

    delta = operator.gamma * 0.2 / 2.0

    expected = (
        (10.0 * (3.0 + delta) + 5.0)
        + (10.0 * (3.0 - delta) + 5.0)
    ) / 2.0

    assert result == pytest.approx(expected)
    assert len(calls) == 2

def test_unequal_radii_width_averaging_when_r2_is_smaller(monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(args)
            return 10.0 * args[-2] + args[-1]

        @staticmethod
        def L_orthogonal(*args):
            raise AssertionError("Unexpected orthogonal call")

    monkeypatch.setitem(operator._MODELS, "quasistatic", FakeModel)

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=5.0,
        r2=3.0,
        parallel=True,
        width=0.2,
    )

    delta = operator.gamma * 0.2 / 2.0

    expected = (
        (10.0 * 5.0 + (3.0 + delta))
        + (10.0 * 5.0 + (3.0 - delta))
    ) / 2.0

    assert result == pytest.approx(expected)
    assert len(calls) == 2

def test_zero_width_uses_original_radii(monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(args)
            return 7.0

        @staticmethod
        def L_orthogonal(*args):
            raise AssertionError("Unexpected orthogonal call")

    monkeypatch.setitem(operator._MODELS, "quasistatic", FakeModel)

    result = operator._mutual_inductance(
        model="quasistatic",
        omega=None,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=4.0,
        r2=5.0,
        parallel=True,
        width=0.0,
    )

    assert result == pytest.approx(7.0)
    assert len(calls) == 1
    assert calls[0][-2:] == (4.0, 5.0)

# Public function: mu_0, zero displacement and cache key
def test_public_function_applies_mu0_once(monkeypatch):
    captured = {}

    def fake_cached(
        model, omega, x, y, z, r1, r2, width, parallel
    ):
        captured.update(
            model=model,
            omega=omega,
            x=x,
            y=y,
            z=z,
            r1=r1,
            r2=r2,
            width=width,
            parallel=parallel,
        )
        return 2.0+3.0j

    monkeypatch.setattr(
        operator, "_cached_mutual_inductance", fake_cached
    )

    result = operator.mutual_inductance(
        displacement=(1.0, 2.0, 3.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="z",
        orientation2="z",
    )

    assert result == pytest.approx(
        operator.mu_0 * (2.0 + 3.0j)
    )
    assert captured["x"] == 1.0
    assert captured["y"] == 2.0
    assert captured["z"] == 3.0

def test_coincident_centers_return_zero_without_model_call(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Cache should not be called")

    monkeypatch.setattr(
        operator, "_cached_mutual_inductance", fail_if_called
    )

    result = operator.mutual_inductance(
        displacement=(0.0, 0.0, 0.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="z",
        orientation2="z",
    )

    assert result == 0.0 + 0.0j

def test_cached_function_reuses_identical_inputs(monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(args)
            return 2.0

        @staticmethod
        def L_orthogonal(*args):
            calls.append(args)
            return 3.0

    monkeypatch.setitem(operator._MODELS, "quasistatic", FakeModel)

    kwargs = dict(
        model="quasistatic",
        omega=None,
        x=1.0,
        y=2.0,
        z=3.0,
        r1=4.0,
        r2=5.0,
        width=0.0,
        parallel=True,
    )

    first = operator._cached_mutual_inductance(**kwargs)
    second = operator._cached_mutual_inductance(**kwargs)

    assert first == second
    assert len(calls) == 1

    info = operator._cached_mutual_inductance.cache_info()
    assert info.misses == 1
    assert info.hits == 1

def test_cache_reuses_equivalent_local_coordinates(monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def L_parallel(*args):
            calls.append(args)
            return 2.0

        @staticmethod
        def L_orthogonal(*args):
            raise AssertionError("Unexpected orthogonal call")

    monkeypatch.setitem(operator._MODELS, "quasistatic", FakeModel)

    first = operator.mutual_inductance(
        displacement=(1.0, 2.0, 3.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="z",
        orientation2="z",
    )

    second = operator.mutual_inductance(
        displacement=(1.0, 3.0, -2.0),
        r1=4.0,
        r2=5.0,
        width=0.0,
        orientation1="y",
        orientation2="y",
    )

    assert first == second
    assert len(calls) == 1

    info = operator._cached_mutual_inductance.cache_info()
    assert info.misses == 1
    assert info.hits == 1

# Mutual inductance matrix
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
        width=0.0,
    )

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)

def test_matrix_empty_input():
    matrix = operator.mutual_inductance_matrix(
        np.empty((0, 3)),
        np.empty((0,), dtype=str),
        radius=1.0,
        width=0.0,
    )

    assert matrix.shape == (0, 0)