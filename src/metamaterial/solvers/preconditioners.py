from __future__ import annotations

from typing import Literal
import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import LinearOperator, splu
from src.metamaterial.physics.coupling.operator import interaction_model, mutual_inductance, mutual_inductance_matrix
ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]
Orientation = Literal["x", "y", "z"]

def orientation_groups(orientations: StrArray) -> dict[Orientation, IntArray]:
    """Returns global indices for each orientation, preserving input order"""
    values = np.asarray(orientations).astype("U1")
    return {axis: np.flatnonzero(values == axis).astype(np.int64) for axis in "xyz" if np.any(values == axis)}

def build_dense_orientation_blocks(centers: FloatArray, orientations: StrArray, *, diagonal: ComplexArray, radius: float, width: float, model: interaction_model = "quasistatic", omega: float | None = None) -> dict[Orientation, tuple[IntArray, ComplexArray]]:
    """Builds exact dense diagonal blocks

    Parameters
    ----------
    centers : FloatArray
        Ring centers, shape (N, 3)
    orientations : StrArray
        Ring orientations, shape (N,)
    diagonal : ComplexArray
        Self-impedance diagonal, shape (N,)
    radius : float
        Radius of the rings
    width : float
        Width of the strip
    model : interaction_model
        Interaction model
    omega : float | None
        Angular frequency; required by non-quasistatic models

    Returns
    -------
    dict[Orientation, tuple[IntArray, ComplexArray]]
        for each orientation, a tuple of global indices and the corresponding
        dense block matrix
    """
    centers_array = np.asarray(centers, dtype=np.float64)
    orientations_array = np.asarray(orientations).astype("U1")
    diagonal_array = np.asarray(diagonal, dtype=np.complex128).reshape(-1)
    n = len(centers_array)

    groups = orientation_groups(orientations_array)
    mutual = mutual_inductance_matrix(centers_array, orientations_array, radius=radius, width=width, model=model, omega=omega)
    result: dict[Orientation, tuple[IntArray, ComplexArray]] = {}
    for axis, indices in groups.items():
        block = -mutual[np.ix_(indices, indices)].copy()
        block[np.diag_indices_from(block)] = diagonal_array[indices]
        result[axis] = (indices, block)
    return result

def build_orientation_block_preconditioner(centers: FloatArray, orientations: StrArray, *, diagonal: ComplexArray, radius: float, width: float, model: interaction_model = "quasistatic", omega: float | None = None) -> LinearOperator:
    """Builds a sparse orientation-block Jacobi inverse preconditioner

    Parameters
    ----------
    centers : FloatArray
        Ring centers, shape (N, 3)
    orientations : StrArray
        Ring orientations, shape (N,)
    diagonal : ComplexArray
        Self-impedance diagonal, shape (N,)
    radius : float
        Radius of the rings
    width : float
        Width of the strip
    model : interaction_model
        Interaction model
    omega : float | None
        Angular frequency, required for retarded models

    Returns
    -------
    LinearOperator
        Inverse preconditioner passed to GMRES
    """
    centers_array = np.asarray(centers, dtype=np.float64)
    orientations_array = np.asarray(orientations).astype("U1")
    diagonal_array = np.asarray(diagonal, dtype=np.complex128).reshape(-1)
    n = len(centers_array)
    groups = orientation_groups(orientations_array)
    factorizations = []

    for axis, indices in groups.items():
        block = lil_matrix((len(indices), len(indices)), dtype=np.complex128)
        block.setdiag(diagonal_array[indices])
        for local_i in range(len(indices)):
            global_i = int(indices[local_i])
            for local_j in range(local_i + 1, len(indices)):
                global_j = int(indices[local_j])
                displacement_array = centers_array[global_j] - centers_array[global_i]
                value = mutual_inductance(displacement=tuple(float(v) for v in displacement_array), r1=radius, r2=radius, width=width, orientation1=axis, orientation2=axis, model=model, omega=omega)
                block[local_i, local_j] = -value
                block[local_j, local_i] = -value

        factor = splu(csc_matrix(block))
        factorizations.append((indices, factor))

    def matvec(vector: npt.ArrayLike) -> ComplexArray:
        rhs = np.asarray(vector, dtype=np.complex128).reshape(-1)
        result = np.zeros(n, dtype=np.complex128)
        for indices, factor in factorizations:
            result[indices] = factor.solve(rhs[indices])
        return result

    return LinearOperator((n, n), matvec=matvec, dtype=np.complex128)
