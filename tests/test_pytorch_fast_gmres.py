from types import SimpleNamespace

import pytest
import torch

from src.pytorch_version.solvers import fast


class _DenseOperator:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape
        self.device = matrix.device
        self.dtype = matrix.dtype
        self.calls = 0

    def __matmul__(self, vector):
        self.calls += 1
        return self.matrix @ vector


def test_gmres_uses_givens_rotations_for_complex_system(monkeypatch):
    matrix = torch.tensor(
        [
            [3.0 + 1.0j, 1.0 - 2.0j, 0.0, 0.5j],
            [0.0 + 1.0j, 4.0 - 1.0j, 1.0, 0.0],
            [1.0, 0.0 - 0.5j, 3.0 + 2.0j, 1.0],
            [0.0, 1.0 + 1.0j, 0.0 - 1.0j, 2.0],
        ],
        dtype=torch.complex128,
    )
    expected = torch.tensor(
        [1.0 + 2.0j, -0.5 + 0.25j, 2.0 - 1.0j, 0.75 + 0.5j],
        dtype=torch.complex128,
    )
    operator = _DenseOperator(matrix)

    def fail_lstsq(*args, **kwargs):
        pytest.fail("GMRES should not call torch.linalg.lstsq")

    monkeypatch.setattr(torch.linalg, "lstsq", fail_lstsq)
    actual = fast._gmres(
        operator,
        matrix @ expected,
        rtol=1e-12,
        restart=2,
        max_iterations=40,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    assert torch.linalg.vector_norm(matrix @ actual - matrix @ expected) < 1e-11


@pytest.mark.parametrize("model", ("quasistatic", "full_retarded"))
def test_solver_warm_starts_each_frequency(monkeypatch, model):
    class _ZeroOperator:
        def __init__(self, size, device):
            self.shape = (size, size)
            self.device = device
            self.dtype = torch.complex128

        def __matmul__(self, vector):
            return torch.zeros_like(vector)

    geometry = SimpleNamespace(
        centers=torch.zeros((2, 3), dtype=torch.int64),
        orientations=("z", "z"),
    )
    config = SimpleNamespace(
        omegas=(1.0, 2.0),
        Dx=1.0,
        Dy=1.0,
        Dz=1.0,
        Nx=1,
        Ny=1,
        Nz=1,
        type="center",
        radius=0.1,
        width=0.0,
        interaction_model=model,
        R=1.0,
        L=1.0,
        C=1.0,
        H_0=1.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
    )

    monkeypatch.setattr(
        fast, "orientation_normals", lambda orientations, device: torch.zeros(
            (2, 3), dtype=torch.float64, device=device
        )
    )
    monkeypatch.setattr(
        fast,
        "build_fft_operator",
        lambda centers, *args, **kwargs: _ZeroOperator(
            centers.shape[0], centers.device
        ),
    )
    monkeypatch.setattr(
        fast,
        "self_inductance",
        lambda *args, device: torch.tensor(
            2.0, dtype=torch.complex128, device=device
        ),
    )
    monkeypatch.setattr(
        fast,
        "build_rhs",
        lambda centers, *args, **kwargs: torch.ones(
            2, dtype=torch.complex128, device=centers.device
        ),
    )
    initial_guesses = []

    def fake_gmres(matrix, rhs, *, initial_guess=None, **kwargs):
        initial_guesses.append(
            None if initial_guess is None else initial_guess.clone()
        )
        return torch.full_like(rhs, len(initial_guesses))

    monkeypatch.setattr(fast, "_gmres", fake_gmres)

    result = fast.solve_system(config, geometry, device="cpu")

    assert initial_guesses[0] is None
    torch.testing.assert_close(initial_guesses[1], result[:, 0])
    torch.testing.assert_close(
        result,
        torch.tensor(
            [[1.0, 2.0], [1.0, 2.0]], dtype=torch.complex128
        ),
    )


def test_gmres_error_reports_frequency_and_relative_residual():
    operator = _DenseOperator(
        torch.tensor(
            [[2.0, 1.0], [1.0, 3.0]], dtype=torch.complex128
        )
    )
    rhs = torch.ones(2, dtype=torch.complex128)

    with pytest.raises(
        RuntimeError,
        match=r"omega=123.*1/1 iterations.*relative=",
    ):
        fast._gmres(
            operator,
            rhs,
            restart=1,
            max_iterations=1,
            omega=123.0,
        )
