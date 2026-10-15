"""Build the right-hand side of the linear system."""

import numpy as np
import numpy.typing as npt
from typing import Literal
CouplingModel = Literal["quasistatic", "full_retarded", "approx_retarded"]
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

mu_0 = 4.0 * np.pi * 1e-7
c = 3e8

def build_rhs(centers: FloatArray, normals: FloatArray, radius: float, H_0: float, H_direction: FloatArray, *, model: CouplingModel = "quasistatic", k_direction: FloatArray | None = None, omega: float | None = None) -> ComplexArray:
    """Build the excitation vector for every configured frequency

    Parameters
    ----------
    centers: FloatArray
        ring centers, shape (N, 3)
    normals: FloatArray
        ring normals, shape (N, 3)
    radius: float
        common mean ring radius
    H_0: float
        amplitude of the incident magnetic field
    H_direction: FloatArray
        direction of the incident magnetic field, shape (3,)
    model: CouplingModel
        coupling model to use
    k_direction: FloatArray | None
        direction of the wave vector, shape (3,), required for non-quasistatic models
    omega: float | None
        angular frequency, required for non-quasistatic models

    Returns
    -------
    ComplexArray
        Array with shape (N,), where N is the number of rings
    """
    centers = np.asarray(centers, dtype=np.float64)
    H_direction = np.asarray(H_direction, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    projections = normals @ H_direction
    n_rings = centers.shape[0]

    flux_amplitude = (mu_0* np.pi * radius**2 * H_0 * projections)
    if model != "quasistatic":
        k_direction = np.asarray(k_direction, dtype=np.float64)
        phase_coordinate = centers @ k_direction # <r_n,k_direction>
        phase = np.exp(1j * omega / c * phase_coordinate)
    else:
        phase = np.ones(n_rings, dtype=np.complex128)
    rhs = phase * flux_amplitude

    return np.asarray(rhs, dtype=np.complex128)