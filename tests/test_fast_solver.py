import numpy as np
from scipy.sparse.linalg import LinearOperator

import src.metamaterial.solvers.fast as fast


def test_solve_system_returns_physical_solution(monkeypatch):
    """The solver must return x satisfying A @ x = rhs."""
    n = 2

    # A = diag(self_inductance), because the mutual operator is zero.
    diagonal = np.array([2.0 + 1.0j, 4.0 - 1.0j])
    rhs = np.array([1.0 + 2.0j, 3.0 - 1.0j])

    class FakeConfig:
        omegas = [1.0]
        Dx = Dy = Dz = 1.0
        Nx = Ny = Nz = 2
        type = "test"
        radius = 0.1
        width = 0.01
        interaction_model = "quasistatic"
        R = 1.0
        L = 1.0
        C = 1.0
        H_0 = 1.0
        H_direction = "z"
        k_direction = None

    class FakeGeometry:
        centers = np.array([[0, 0, 0], [1, 0, 0]])
        orientations = np.array(["x", "y"])

    monkeypatch.setattr(
        fast,
        "orientation_normals",
        lambda orientations: np.zeros((n, 3)),
    )

    monkeypatch.setattr(
        fast,
        "self_inductance",
        lambda *args: diagonal[0],
    )

    # In solve_system, M_0 is computed as a scalar times ones.
    # Make it the same diagonal entry for this test.
    monkeypatch.setattr(
        fast,
        "build_fft_operator",
        lambda *args, **kwargs: LinearOperator(
            shape=(n, n),
            matvec=lambda x: np.zeros(n, dtype=np.complex128),
            dtype=np.complex128,
        ),
    )

    monkeypatch.setattr(
        fast,
        "build_rhs",
        lambda *args, **kwargs: rhs.copy(),
    )

    # Use a nontrivial but known left preconditioner.
    P_diagonal = np.array([0.5 + 0.0j, 0.25 + 0.0j])

    monkeypatch.setattr(
        fast,
        "build_preconditioner",
        lambda *args, **kwargs: LinearOperator(
            shape=(n, n),
            matvec=lambda x: P_diagonal * x,
            dtype=np.complex128,
        ),
    )

    result = fast.solve_system(FakeConfig(), FakeGeometry())

    expected = rhs / diagonal[0]

    assert result.shape == (n, 1)
    np.testing.assert_allclose(result[:, 0], expected, rtol=1e-9, atol=1e-11)