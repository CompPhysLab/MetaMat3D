"""Preconditioners for iterative metamaterial solvers."""
from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import LinearOperator, spilu
from scipy.spatial import cKDTree
from src.metamaterial.physics.coupling.operator import mutual_inductance

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
StrArray = NDArray[np.str_]
IntArray = NDArray[np.int16]
InteractionModel = Literal["quasistatic", "full_retarded", "approx_retarded"]

def build_nearest_neighbor_matrix(centers: IntArray, spacings: FloatArray,
                                  orientations: StrArray, diagonal: ComplexArray, *,
                                  radius: float, width: float, cutoff_distance: float,
                                  model: InteractionModel = "quasistatic", omega: float | None = None
                                  ) -> csc_matrix:
    centers = np.asarray(centers, dtype=np.int16)
    spacings = np.asarray(spacings, dtype=float)
    orientations = np.asarray(orientations)
    diagonal = np.asarray(diagonal, dtype=np.complex128).reshape(-1)
    n = len(centers)
    pairs = cKDTree(centers).query_pairs(cutoff_distance, output_type="ndarray")
    rows: list[int] = list(range(n))
    cols: list[int] = list(range(n))
    values: list[complex] = diagonal.astype(np.complex128).tolist()

    for i, j in pairs:
        displacement = centers[j] - centers[i]
        distance = float(np.linalg.norm(displacement))
        if distance < cutoff_distance:
            value = mutual_inductance(
                displacement=displacement,
                spacings=spacings,
                r1=float(radius),
                r2=float(radius),
                width=float(width),
                orientation1=str(orientations[i]),
                orientation2=str(orientations[j]),
                model=model,
                omega=omega,
            )
            rows.extend((int(i), int(j)))
            cols.extend((int(j), int(i)))
            values.extend((-value, -value))

    matrix = coo_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, cols)),
        shape=(n, n),dtype=np.complex128).tocsc()
    matrix.sum_duplicates()
    return matrix

def build_preconditioner(centers: IntArray, spacings: FloatArray,
                                          orientations: StrArray, M_0: complex, *,
                                          radius: float, width: float, cutoff_distance: float,
                                          model: InteractionModel = "quasistatic", omega: float | None = None
                                          ) -> LinearOperator:
    diagonal = np.full(len(centers), M_0, dtype=np.complex128)
    matrix = build_nearest_neighbor_matrix(centers, spacings,
                                           orientations, diagonal,
                                           radius=radius, width=width, cutoff_distance=cutoff_distance,
                                           model=model, omega=omega)
    factor = spilu(matrix, fill_factor = 2, drop_tol = 1e-1)

    def matvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(factor.solve(np.asarray(vector, dtype=np.complex128)))

    preconditioner = LinearOperator(
        shape=matrix.shape,
        matvec=matvec,
        dtype=np.complex128,
    )
    return preconditioner
