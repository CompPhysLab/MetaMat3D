
import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.metamaterial.physics.coupling.operator import (
    build_fft_operator,
    mutual_inductance_matrix,
)

@pytest.mark.parametrize(
    ("ring_type", "counts", "centers"),
    [
        # Кольца в центрах ячеек
        (
            "center",
            (3, 3, 3),
            np.array([
                [1, 1, 1],
                [3, 1, 1],
                [1, 3, 1],
                [1, 1, 3],
                [3, 3, 1],
                [3, 1, 3],
            ]),
        ),
        # Кольца на границах по нормали к ориентации
        (
            "border",
            (3, 3, 3),
            np.array([
                [0, 1, 1],  # x
                [1, 0, 1],  # y
                [1, 1, 0],  # z
                [2, 3, 1],  # x
                [3, 2, 1],  # y
                [1, 3, 2],  # z
            ])
        ),
        # Открытая сетка по нормали к ориентации
        (
            "open",
            (4, 4, 4),
            np.array([
                [2, 1, 1],  # x
                [1, 2, 3],  # y
                [1, 3, 2],  # z
                [4, 3, 1],  # x
                [3, 4, 5],  # y
                [5, 1, 4],  # z
            ])
        ),
    ],
)
def test_fft_operator_matches_dense_matrix(
    ring_type: str,
    counts: tuple[int, int, int],
    centers: np.ndarray,
) -> None:
    orientations = np.array(["x", "y", "z", "x", "y", "z"])

    spacings = (1.0, 1.3, 0.8)
    radius = 0.2
    width = 0.05

    dense = mutual_inductance_matrix(
        centers=centers,
        spacings=np.asarray(spacings),
        orientations=orientations,
        radius=radius,
        width=width,
        model="quasistatic",
    )

    fft_operator = build_fft_operator(
        centers=centers,
        orientations=orientations,
        counts=counts,
        spacings=spacings,
        type=ring_type,
        radius=radius,
        width=width,
        model="quasistatic",
    )

    rng = np.random.default_rng(12345)

    for _ in range(3):
        vector = (
            rng.standard_normal(len(centers))
            + 1j * rng.standard_normal(len(centers))
        )

        expected = dense @ vector
        actual = fft_operator @ vector

        assert_allclose(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-12,
        )