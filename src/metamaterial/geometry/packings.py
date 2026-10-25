"""Geometric inclusion rules for ring centers."""
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int16]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

def rectangle(points: IntArray, counts: tuple[int, int, int]) -> BoolArray:
    """Keeps all candidate points in the rectangular bounding box

    Parameters
    ----------
    points : IntArray
        Array of shape (N, 3)
    counts : tuple[int, int, int]
        Number of meta-atoms along each axis (Nx, Ny, Nz)

    Returns
    -------
    BoolArray
        Boolean array of shape (N,) indicating which points are inside the box.
    """
    points = np.asarray(points, dtype=int)
    return np.ones(points.shape[0], dtype=bool)

def ellipse(points: IntArray, counts: tuple[int, int, int], *, tol: float = 1e-12,) -> BoolArray:
    """Keeps points inside an ellipsoid centered at (Nx, Ny, Nz)

    Parameters
    ----------
    points : IntArray
        Array of shape (N, 3)
    counts : tuple[int, int, int]
        Number of meta-atoms along each axis (Nx, Ny, Nz)
    tol : float, optional
        Numerical tolerance for inclusion, by default 1e-12
    
    Returns
    -------
    BoolArray
        Boolean array of shape (N,) indicating which points are inside the ellipsoid
    """
    points = np.asarray(points, dtype=int)
    center = np.asarray(counts, dtype=int) 
    normalized = (points - center) / center
    return np.sum(normalized**2, axis=1) <= 1.0 + tol

def cylinder_z(points: IntArray, counts: tuple[int, int, int], *, tol: float = 1e-12,) -> BoolArray:
    """Keeps points inside a cylinder along the z-axis centered at (Lx/2, Ly/2) with height Lz

    Parameters
    ----------
    points : IntArray
        Array of shape (N, 3)
    counts : tuple[int, int, int]
        Number of meta-atoms along each axis (Nx, Ny, Nz)
    tol : float, optional
        Numerical tolerance for inclusion, by default 1e-12
    
    Returns
    -------
    BoolArray
        Boolean array of shape (N,) indicating which points are inside the cylinder
    """
    xy_points = np.asarray(points, dtype=int)[:, :2]
    center_xy = np.asarray(counts, dtype=int)[:2]
    normalized_xy = (xy_points - center_xy) / center_xy
    return np.sum(normalized_xy**2, axis=1) <= 1.0 + tol

PACKING_RULES: dict[str, Callable[..., BoolArray]] = {
    "rectangle": rectangle,
    "ellipse": ellipse,
    "cylinder_z": cylinder_z,
}


def get_packing_rule(name: str) -> Callable[..., BoolArray]:
    """Returns a packing rule by name"""
    try:
        return PACKING_RULES[name]
    except (KeyError, AttributeError) as exc:
        supported = ", ".join(PACKING_RULES)
        raise ValueError(
            f"Unsupported packing {name!r}. Supported packings: {supported}"
        ) from exc
