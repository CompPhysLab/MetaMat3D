"""Build the right-hand side of the linear system."""

import numpy as np
import numpy.typing as npt
from typing import Literal
interaction_model = Literal["quasistatic", "full_retarded", "approx_retarded"]
IntArray = npt.NDArray[np.int16]
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

mu_0 = 4.0 * np.pi * 1e-7
c = 3e8

def build_rhs(centers: IntArray, spacings: FloatArray, normals: IntArray,
              radius: float, H_0: float, H_direction: FloatArray, *,
              model: interaction_model = "quasistatic",
              k_direction: FloatArray | None = None, omega: float | None = None) -> ComplexArray:
    """Builds the excitation vector for given frequency

    Parameters
    ----------
    centers: IntArray
        Ring centers, shape (N, 3)
    spacings: FloatArray
        Lattice diameters, shape (3), containing Dx, Dy, Dz
    normals: FloatArray
        Ring normals, shape (N, 3)
    radius: float
        Common mean ring radius
    H_0: float
        Amplitude of the incident magnetic field
    H_direction: FloatArray
        Direction of the incident magnetic field, shape (3,)
    model: interaction_model
        Interaction model to use
    k_direction: FloatArray | None
        Direction of the wave vector, shape (3,), required for non-quasistatic models
    omega: float | None
        Angular frequency, required for non-quasistatic models

    Returns
    -------
    ComplexArray
        Array with shape (N,), where N is the number of rings
    """
    centers = np.asarray(centers, dtype=np.int16)
    spacings = np.asarray(spacings, dtype=float)
    H_direction = np.asarray(H_direction, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.int16)
    projections = normals @ H_direction
    n_rings = centers.shape[0]

    flux_amplitude = (mu_0* np.pi * radius**2 * H_0 * projections)
    if model != "quasistatic":
        k_direction = np.asarray(k_direction, dtype=np.float64)
        phase_coordinate = (centers*spacings/2) @ k_direction # <r_n,k_direction>
        phase = np.exp(1j * omega / c * phase_coordinate)
    else:
        phase = np.ones(n_rings, dtype=np.complex128)
    rhs = phase * flux_amplitude

    return np.asarray(rhs, dtype=np.complex128)