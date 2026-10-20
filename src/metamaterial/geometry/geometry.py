"""Generation of ordered ring-center geometry."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from .packings import get_packing_rule

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StrArray = NDArray[np.str_]

_ALLOWED_TYPES = {"border", "open", "center"}
_ALLOWED_ORIENTATIONS = {"z", "y", "x"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

@dataclass(frozen=True)
class Geometry:
    """Ordered geometry consumed by the solvers

    centers[n] is the (x, y, z) center of ring n
    orientations[n] is the normal direction of that ring ('z', 'y', or 'x')
    grid_indices[n] stores (i, j, k) in physical x/y/z axis order
    """
    centers: FloatArray
    orientations: StrArray
    grid_indices: IntArray

    def __post_init__(self) -> None:
        n = self.centers.shape[0]
        if self.centers.shape != (n, 3):
            raise ValueError("centers must have shape (N, 3)")
        if self.orientations.shape != (n,):
            raise ValueError("orientations must have shape (N,)")
        if self.grid_indices.shape != (n, 3):
            raise ValueError("grid_indices must have shape (N, 3)")

def _coordinates_for_orientation(orientation: str, type: str, counts: tuple[int, int, int],) -> tuple[tuple[NDArray[np.float64], ...], tuple[NDArray[np.int64], ...]]:
    """Builds x/y/z coordinate and index axes for one ring orientation
    
    Parameters
    ----------
    orientation : str
        Ring orientation ('z', 'y', or 'x')
    type : str
        Meta-atom type ('border', 'open', or 'center')
    counts : tuple[int, int, int]
        Number of meta-atoms along each axis (Nx, Ny, Nz)

    Returns
    -------
    tuple[tuple[NDArray[np.float64], ...], tuple[NDArray[np.int64], ...]]
        X/Dx, Y/Dy, and Z/Dz coordinate arrays and index arrays
    """
    normal_axis = _AXIS_INDEX[orientation]
    coord_axes = []
    index_axes = []

    for axis in range(3):
        count = counts[axis]

        if axis != normal_axis or type == "center":
            indices = np.arange(count, dtype=np.int64)
            coords = (indices.astype(float) + 0.5)
        elif type == "border":
            indices = np.arange(count + 1, dtype=np.int64)
            coords = indices.astype(float)
        elif type == "open":
            indices = np.arange(1, count, dtype=np.int64)
            coords = indices.astype(float)
        else:
            raise ValueError(f"Unsupported type {type!r}")

        coord_axes.append(coords)
        index_axes.append(indices)

    return tuple(coord_axes), tuple(index_axes)

def build_geometry(config) -> Geometry:
    """Builds geometry from a SimulationConfig-like object

    Parameters
    ----------
    config : object
        Configuration object with attributes Dx, Dy, Dz, orientations, type, Nx, Ny, Nz, packing

    Returns
    -------
    Geometry
        Ordered geometry with centers, orientations, and grid indices

    Notes
    -----
        1. Orientations follow config.orientations (e.g. 'zyx')
        2. Within each orientation, k-j-i order
        3. Shape filtering preserves the order of surviving candidates
    """
    dx, dy, dz = float(config.Dx), float(config.Dy), float(config.Dz)
    nx, ny, nz = int(config.Nx), int(config.Ny), int(config.Nz)
    type = str(config.type)
    packing = str(config.packing)
    orientations = str(config.orientations)

    counts = (nx, ny, nz)
    spacings = (dx, dy, dz)
    packing_rule = get_packing_rule(packing)

    center_blocks = []
    orientation_blocks = []
    index_blocks = []

    for orientation in orientations:
        coord_axes, index_axes = _coordinates_for_orientation(orientation, type, counts)
        # meshgrid(indexing='ij') stores axes as (x, y, z).
        # Fortran-order flattening makes x/i vary fastest, then y/j, then z/k.
        x, y, z = np.meshgrid(*coord_axes, indexing="ij")
        ix, iy, iz = np.meshgrid(*index_axes, indexing="ij")

        centers = np.column_stack((x.ravel(order="F")*dx,
                                   y.ravel(order="F")*dy,
                                   z.ravel(order="F")*dz))
        points  = np.column_stack((x.ravel(order="F"),
                                   y.ravel(order="F"),
                                   z.ravel(order="F")))
        indices = np.column_stack((ix.ravel(order="F"),
                                   iy.ravel(order="F"),
                                   iz.ravel(order="F"))).astype(np.int64)

        active = packing_rule(points, counts)
        center_blocks.append(centers[active])
        index_blocks.append(indices[active])
        orientation_blocks.append(
            np.full(np.count_nonzero(active), orientation, dtype="<U1")
        )

    if center_blocks:
        centers = np.concatenate(center_blocks, axis=0)
        grid_indices = np.concatenate(index_blocks, axis=0)
        ring_orientations = np.concatenate(orientation_blocks, axis=0)
    else:
        centers = np.empty((0, 3), dtype=float)
        grid_indices = np.empty((0, 3), dtype=np.int64)
        ring_orientations = np.empty((0,), dtype="<U1")

    return Geometry(
        centers=centers,
        orientations=ring_orientations,
        grid_indices=grid_indices,
    )

def orientation_normals(orientations: StrArray) -> FloatArray:
    """Converts ring orientations to Cartesian unit normals"""
    normals = np.zeros((orientations.size, 3), dtype=np.float64)
    for axis, column in (("x", 0), ("y", 1), ("z", 2)):
        normals[orientations == axis, column] = 1.0
    return normals