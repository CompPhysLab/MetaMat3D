"""Geometric inclusion rules for ring centers."""
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

def rectangle(points: FloatArray, counts: tuple[float, float, float]) -> BoolArray:
    """Keeps all candidate points in the rectangular bounding box

    Parameters
    ----------
    points : FloatArray
        Array of shape (N, 3)
    counts : tuple[float, float, float]
        Number of meta-atoms along each axis (Nx, Ny, Nz)

    Returns
    -------
    BoolArray
        Boolean array of shape (N,) indicating which points are inside the box.
    """
    points = np.asarray(points, dtype=float)
    return np.ones(points.shape[0], dtype=bool)

def ellipse(points: FloatArray, counts: tuple[float, float, float], *, tol: float = 1e-12,) -> BoolArray:
    """Keeps points inside an ellipsoid centered at (Lx/2, Ly/2, Lz/2)

    Parameters
    ----------
    points : FloatArray
        Array of shape (N, 3)
    counts : tuple[float, float, float]
        Number of meta-atoms along each axis (Nx, Ny, Nz)
    tol : float, optional
        Numerical tolerance for inclusion, by default 1e-12
    
    Returns
    -------
    BoolArray
        Boolean array of shape (N,) indicating which points are inside the ellipsoid
    """
    points = np.asarray(points, dtype=float)
    lengths = np.asarray(counts, dtype=float)

    center = lengths / 2.0
    normalized = (points - center) / center
    return np.sum(normalized**2, axis=1) <= 1.0 + tol

def cylinder_z(points: FloatArray, counts: tuple[float, float, float], *, tol: float = 1e-12,) -> BoolArray:
    """Keeps points inside a cylinder along the z-axis centered at (Lx/2, Ly/2) with height Lz

    Parameters
    ----------
    points : FloatArray
        Array of shape (N, 3)
    counts : tuple[float, float, float]
        Number of meta-atoms along each axis (Nx, Ny, Nz)
    tol : float, optional
        Numerical tolerance for inclusion, by default 1e-12
    
    Returns
    -------
    BoolArray
        Boolean array of shape (N,) indicating which points are inside the cylinder
    """
    points = np.asarray(points, dtype=float)
    lengths = np.asarray(counts, dtype=float)

    center_xy = lengths[:2] / 2.0
    xy_points = points[:, :2]
    normalized_xy = (xy_points - center_xy) / center_xy
    z_points = points[:, 2]

    inside_cylinder_xy = np.sum(normalized_xy**2, axis=1) <= 1.0 + tol
    inside_cylinder_z = (z_points >= -tol) & (z_points <= lengths[2] + tol)

    return inside_cylinder_xy & inside_cylinder_z

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
