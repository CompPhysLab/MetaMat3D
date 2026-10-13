from functools import lru_cache
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from . import approx_retarded, full_retarded, quasistatic

mu_0 = 4 * np.pi * 1e-7
gamma = 0.7 #gamma-factor
CouplingModel = Literal["quasistatic", "full_retarded", "approx_retarded"]
Orientation = Literal["x", "y", "z"]
FloatArray = NDArray[np.float64]

_MODELS = {
    "quasistatic": quasistatic,
    "full_retarded": full_retarded,
    "approx_retarded": approx_retarded,
}

def _transform_coordinates(dx: float, dy: float, dz: float, orientation1: Orientation, orientation2: Orientation) -> tuple[bool, float, float, float]:
    """Transform displacement into coordinates expected by the model

    Parameters
    ----------
    dx, dy, dz : float
        displacement coordinates in the global frame
    orientation1, orientation2 : Orientation
        orientations of the two rings, each one of 'x', 'y', 'z'
    
    Returns
    -------
        parallel : bool
            whether the rings have the same orientation
        x, y, z : float
            transformed displacement coordinates

    Notes
    -----
        1. Parallel rings' coordinates convert to the coordinates of pair of rings with orientations 'z'
        2. Orthogonal rings' coordinates convert to the coordinates of pair of rings with orientations 'z' and 'y'
    """
    if orientation1 == orientation2:
        if orientation1 == "z":
            return True, dx, dy, dz
        if orientation1 == "y":
            return True, dx, -dz, dy
        return True, -dz, dy, dx

    if orientation1 == "z":
        if orientation2 == "y":
            return False, dx, dy, dz
        return False, dy, dx, dz

    if orientation1 == "y":
        if orientation2 == "z":
            return False, dx, dz, dy
        return False, -dz, dx, dy

    if orientation2 == "z":
        return False, dy, dz, dx
    return False, dz, dy, dx

def _mutual_inductance(model: CouplingModel, omega: float | None, dx: float, dy: float, dz: float, r1: float, r2: float, orientation1: Orientation, orientation2: Orientation, width: float = 0.0) -> complex:
    """Calculate the mutual inductance

    Parameters
    ----------
    model : CouplingModel
        coupling model to use for the calculation
    omega : float | None
        angular frequency of the electromagnetic wave (required for non-quasistatic models)
    dx, dy, dz : float
        displacement coordinates between the centers of the two rings
    r1, r2 : float
        radii of the two rings
    orientation1, orientation2 : Orientation
        orientations of the two rings
    width : float, optional
        width of the strip, by default 0

    Returns
    -------
    complex
        mutual inductance between the two rings
    """
    parallel, x, y, z = _transform_coordinates(dx, dy, dz, orientation1, orientation2)
    module = _MODELS[model]
    function_name = "L_parallel" if parallel else "L_orthogonal"
    function = getattr(module, function_name)
    if model == "quasistatic":
        if r1 == r2 and width:
            w = width * gamma
            R = r1 + w / 2
            r = r1 - w / 2
            L_1 = function(x, y, z, r, r)
            L_2 = function(x, y, z, r, R)
            L_3 = function(x, y, z, R, R)
            return (L_1 + 2 * L_2 + L_3) / 4
        elif width:
            id_r1 = r1 == min(r1, r2)
            id_r2 = r2 == min(r1, r2)
            L_1 = function(x, y, z, r1 + id_r1 * width/2, r2 + id_r2*width/2)
            L_2 = function(x, y, z, r1 - id_r1 * width/2, r2 - id_r2*width/2)
            return (L_1 + L_2)/2
        return complex(function(x, y, z, r1, r2))
    if r1 == r2 and width:
        w = width * gamma
        R = r1 + w / 2
        r = r1 - w / 2
        L_1 = function(omega, x, y, z, r, r)
        L_2 = function(omega, x, y, z, r, R)
        L_3 = function(omega, x, y, z, R, R)
        return (L_1 + 2 * L_2 + L_3) / 4
    elif width:
        id_r1 = r1 == min(r1, r2)
        id_r2 = r2 == min(r1, r2)
        L_1 = function(omega, x, y, z, r1 + id_r1 * width/2, r2 + id_r2*width/2)
        L_2 = function(omega, x, y, z, r1 - id_r1 * width/2, r2 - id_r2*width/2)
        return (L_1 + L_2)/2
    return complex(function(omega, x, y, z, r1, r2))

@lru_cache(maxsize=100_000)
def _cached_mutual_inductance(model: CouplingModel, omega: float | None, dx: float, dy: float, dz: float, r1: float, r2: float, width: float, orientation1: Orientation, orientation2: Orientation) -> complex:
    """Cache immutable scalar inputs and the resulting mutual inductance for performance"""
    return _mutual_inductance(model, omega, dx, dy, dz, r1, r2, orientation1, orientation2, width)

def mutual_inductance(*, displacement: tuple[float, float, float], r1: float, r2: float, width: float, orientation1: Orientation, orientation2: Orientation, model: CouplingModel = "quasistatic", omega: float | None = None) -> complex:
    """Calculate mutual inductance M12

    Parameters
    ----------
    displacement : tuple[float, float, float]
        vector from the center of ring 1 to the center of ring 2
    r1, r2 : float
        radii of the two rings
    width : float
        width of the strip
    orientation1, orientation2 : Orientation
        orientations of the two rings, each one of 'x', 'y', 'z'
    model : CouplingModel, optional
        coupling model to use for the calculation (default: 'quasistatic')
    omega : float | None, optional
        angular frequency of the electromagnetic wave (required for non-quasistatic models)

    Returns
    -------
    complex
        mutual inductance between the two rings
    """
    
    dx, dy, dz = map(float, displacement)
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return 0.0 + 0.0j

    mutual_inductance = _cached_mutual_inductance(model, None if omega is None else float(omega), dx, dy, dz, r1, r2, width, orientation1, orientation2)
    return complex(mu_0 * mutual_inductance)

def mutual_inductance_matrix(centers: FloatArray, orientations: NDArray, *, radius: float, model: CouplingModel = "quasistatic", omega: float | None = None) -> NDArray[np.complex128]:
    """Build the symmetric mutual-inductance matrix

    Parameters
    ----------
    centers: FloatArray
        ring centers, shape (N, 3)
    orientations: NDArray
        ring orientations, shape (N,), containing 'x', 'y', 'z'
    radius: float
        common mean ring radius
    model: CouplingModel
        coupling model
    omega: float | None
        angular frequency; required by non-quasistatic models

    Returns:
        Complex matrix of shape (N, N), with a zero diagonal
    """

    centers = np.asarray(centers, dtype=float)
    orientations = np.asarray(orientations)

    n_rings = len(centers)
    matrix = np.zeros((n_rings, n_rings), dtype=np.complex128)

    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            displacement = tuple(centers[j] - centers[i])
            value = mutual_inductance(displacement=displacement, r1=radius, r2=radius, orientation1=str(orientations[i]), orientation2=str(orientations[j]), model=model, omega=omega)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix

def clear_cache() -> None:
    """Clear the operator's mutual-inductance cache"""
    _cached_mutual_inductance.cache_clear()
