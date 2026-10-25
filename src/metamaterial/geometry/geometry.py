"""Generation of ordered ring-center geometry."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from .packings import get_packing_rule

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int16]
StrArray = NDArray[np.str_]

_ALLOWED_TYPES = {"border", "open", "center"}
_ALLOWED_ORIENTATIONS = {"z", "y", "x"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

@dataclass(frozen=True)
class Geometry:
    """Ordered geometry consumed by the solvers

    centers[n] is the (2x/Dx, 2y/Dy, 2z/Dz) center of ring n
    orientations[n] is the normal direction of that ring ('z', 'y', or 'x')
    """
    centers: IntArray
    orientations: StrArray

    def __post_init__(self) -> None:
        n = self.centers.shape[0]
        if self.centers.shape != (n, 3):
            raise ValueError("centers must have shape (N, 3)")
        if self.orientations.shape != (n,):
            raise ValueError("orientations must have shape (N,)")

def _coordinates_for_orientation(orientation: str, type: str, counts: tuple[int, int, int],) -> tuple[IntArray, ...]:
    """Builds i/j/k coordinate axes for one ring orientation
    
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
    tuple[NDArray[np.float64], ...]
        2x/Dx, 2y/Dy, and 2z/Dz coordinate arrays
    """
    normal_axis = _AXIS_INDEX[orientation]
    coord_axes = []
    index_axes = []

    for axis in range(3):
        count = counts[axis]

        if axis != normal_axis or type == "center":
            coords = 2 * np.arange(count, dtype=np.int16)+1
        elif type == "border":
            coords = 2 * np.arange(count + 1, dtype=np.int16)
        elif type == "open":
            coords = 2 * np.arange(1, count, dtype=np.int16)

        coord_axes.append(coords)

    return tuple(coord_axes)

def build_geometry(config) -> Geometry:
    """Builds geometry from a SimulationConfig-like object

    Parameters
    ----------
    config : object
        Configuration object with attributes Nx, Ny, Nz, orientations, type, packing

    Returns
    -------
    Geometry
        Ordered geometry with centers, orientations

    Notes
    -----
        1. Orientations follow config.orientations (e.g. 'zyx')
        2. Within each orientation, k-j-i order
        3. Shape filtering preserves the order of surviving candidates
    """
    counts = (int(config.Nx), int(config.Ny), int(config.Nz))
    type = str(config.type)
    packing = str(config.packing)
    orientations = str(config.orientations)
    packing_rule = get_packing_rule(packing)

    center_blocks = []
    orientation_blocks = []

    for orientation in orientations:
        coord_axes = _coordinates_for_orientation(orientation, type, counts)
        i, j, k = np.meshgrid(*coord_axes, indexing="ij")

        centers = np.column_stack((i.ravel(order="F"),
                                   j.ravel(order="F"),
                                   k.ravel(order="F")))

        active = packing_rule(centers, counts)
        center_blocks.append(centers[active])
        orientation_blocks.append(
            np.full(np.count_nonzero(active), orientation, dtype="<U1")
        )

    centers = np.concatenate(center_blocks, axis=0)
    ring_orientations = np.concatenate(orientation_blocks, axis=0)

    return Geometry(
        centers=centers,
        orientations=ring_orientations
    )

def orientation_normals(orientations: StrArray) -> IntArray:
    """Converts ring orientations to Cartesian unit normals"""
    normals = np.zeros((orientations.size, 3), dtype=np.int16)
    for axis, column in (("x", 0), ("y", 1), ("z", 2)):
        normals[orientations == axis, column] = 1.0
    return normals