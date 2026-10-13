"""Build the right-hand side of the linear system"""

import numpy as np
import numpy.typing as npt
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

mu_0 = 4.0 * np.pi * 1e-7
c = 3e8

def _orientation_normals(orientations: npt.NDArray[np.str_]) -> FloatArray:
    """Convert ring orientations to Cartesian unit normals"""
    normals = np.zeros((orientations.size, 3), dtype=np.float64)

    for axis, column in (("x", 0), ("y", 1), ("z", 2)):
        normals[orientations == axis, column] = 1.0

    if not np.all(np.any(normals != 0.0, axis=1)):
        raise ValueError("Geometry contains an invalid ring orientation")

    return normals

def build_rhs(config: SimulationConfig, geometry: Geometry) -> ComplexArray:
    """Build the excitation vector for every configured frequency

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object
    geometry : Geometry
        Geometry object containing ring centers and orientations

    Returns
    -------
    ComplexArray
        Array with shape (Nf, N), where Nf is the number of
        frequencies and N is the number of rings
        rhs[f, n] is the magnetic flux through ring n at frequency f
    """
    centers = np.asarray(geometry.centers, dtype=np.float64)
    orientations = np.asarray(geometry.orientations)
    frequencies = np.asarray(config.frequencies, dtype=np.float64)

    n_rings = centers.shape[0]
    n_frequencies = frequencies.size

    if centers.shape != (n_rings, 3):
        raise ValueError("geometry.centers must have shape (N, 3)")

    if orientations.shape != (n_rings,):
        raise ValueError("geometry.orientations must have shape (N,)")

    if not np.all(np.isfinite(centers)):
        raise ValueError("Ring centers must be finite")

    normals = _orientation_normals(orientations)
    H_direction = np.asarray(config.H_direction, dtype=np.float64)
    projections = normals @ H_direction
    flux_amplitude = (mu_0* np.pi * config.radius**2 * config.H_0 * projections)
    if config.is_non_quasistatic:
        if config.k_direction is None:
            raise ValueError(
                "k_direction is required for non-quasistatic excitation"
            )
        k_direction = np.asarray(config.k_direction, dtype=np.float64)
        phase_coordinate = centers @ k_direction # <r_n,k_direction>
        phase = np.exp(1j * frequencies[:, None] / c * phase_coordinate[None, :])
    else:
        phase = np.ones(
            (n_frequencies, n_rings),
            dtype=np.complex128,
        )
    rhs = phase * flux_amplitude[None, :]

    return np.asarray(rhs, dtype=np.complex128)