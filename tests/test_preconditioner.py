import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
import src.metamaterial.solvers.preconditioners as preconditioners

def make_geometry():
    centers = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [4, 0, 0],
        ],
        dtype=np.int64,
    )
    spacings = np.array([1.0, 1.0, 1.0])
    orientations = np.array(["x", "y", "z"])
    diagonal = np.array([10.0, 20.0, 30.0], dtype=np.complex128)
    return centers, spacings, orientations, diagonal

def test_matrix_preserves_diagonal(monkeypatch):
    centers, spacings, orientations, diagonal = make_geometry()

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        lambda **kwargs: 2.0 + 1.0j,
    )

    matrix = preconditioners.build_nearest_neighbor_matrix(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=1.5,
    )

    np.testing.assert_allclose(matrix.diagonal(), diagonal)

def test_matrix_includes_only_pairs_inside_cutoff(monkeypatch):
    centers, spacings, orientations, diagonal = make_geometry()
    calls = []

    def fake_mutual_inductance(**kwargs):
        calls.append(kwargs)
        return 1.0 + 0.0j

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        fake_mutual_inductance,
    )

    matrix = preconditioners.build_nearest_neighbor_matrix(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=1.5,
    )

    assert len(calls) == 1
    assert matrix[0, 1] == pytest.approx(-1.0)
    assert matrix[1, 0] == pytest.approx(-1.0)
    assert matrix[0, 2] == pytest.approx(0.0)
    assert matrix[1, 2] == pytest.approx(0.0)

def test_pair_exactly_at_cutoff_is_excluded(monkeypatch):
    centers = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    spacings = np.ones(3)
    orientations = np.array(["x", "y"])
    diagonal = np.array([10.0, 20.0], dtype=np.complex128)

    def unexpected_call(**kwargs):
        pytest.fail("mutual_inductance must not be called for a pair at the cutoff")

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        unexpected_call,
    )

    matrix = preconditioners.build_nearest_neighbor_matrix(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=1.0,
    )

    np.testing.assert_allclose(matrix.toarray(), np.diag(diagonal))

def test_interactions_between_different_orientations_are_included(monkeypatch):
    centers = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64)
    spacings = np.ones(3)
    orientations = np.array(["x", "z"])
    diagonal = np.array([10.0, 20.0], dtype=np.complex128)

    calls = []

    def fake_mutual_inductance(**kwargs):
        calls.append(kwargs)
        assert kwargs["orientation1"] == "x"
        assert kwargs["orientation2"] == "z"
        return 3.0 + 2.0j

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        fake_mutual_inductance,
    )

    matrix = preconditioners.build_nearest_neighbor_matrix(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=1.5,
    )

    assert len(calls) == 1
    assert matrix[0, 1] == pytest.approx(-3.0 - 2.0j)
    assert matrix[1, 0] == pytest.approx(-3.0 - 2.0j)

def test_preconditioner_returns_linear_operator(monkeypatch):
    centers, spacings, orientations, diagonal = make_geometry()

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        lambda **kwargs: 0.0,
    )

    P = preconditioners.build_preconditioner(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=0.5,
    )

    assert isinstance(P, LinearOperator)
    assert P.shape == (3, 3)
    assert P.dtype == np.dtype(np.complex128)

def test_preconditioner_inverts_diagonal_matrix(monkeypatch):
    centers, spacings, orientations, diagonal = make_geometry()

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        lambda **kwargs: 0.0,
    )

    P = preconditioners.build_preconditioner(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=0.5,
    )

    vector = np.array([1.0 + 2.0j, 3.0 - 1.0j, -2.0j])
    expected = vector / diagonal

    np.testing.assert_allclose(P @ vector, expected)

def test_preconditioner_solves_approximate_matrix(monkeypatch):
    centers, spacings, orientations, diagonal = make_geometry()

    monkeypatch.setattr(
        preconditioners,
        "mutual_inductance",
        lambda **kwargs: 1.0 + 0.0j,
    )

    cutoff = 1.5
    matrix = preconditioners.build_nearest_neighbor_matrix(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=cutoff,
    )
    P = preconditioners.build_preconditioner(
        centers,
        spacings,
        orientations,
        diagonal,
        radius=0.1,
        width=0.01,
        cutoff_distance=cutoff,
    )

    vector = np.array([1.0 + 1.0j, 2.0 - 1.0j, 0.5 + 3.0j])

    np.testing.assert_allclose(P @ (matrix @ vector), vector, rtol=1e-12, atol=1e-12)

def test_singular_preconditioner_matrix_raises(monkeypatch):
    centers = np.array([[0, 0, 0]], dtype=np.int64)
    spacings = np.ones(3)
    orientations = np.array(["x"])
    diagonal = np.array([0.0 + 0.0j])

    with pytest.raises(RuntimeError):
        preconditioners.build_preconditioner(
            centers,
            spacings,
            orientations,
            diagonal,
            radius=0.1,
            width=0.01,
            cutoff_distance=1.0,
        )