"""Legacy retarded-inductance calculation with corrected orientation mapping."""

import numpy as np
from tqdm import tqdm

from src.metamaterial.physics.coupling import full_retarded_old

mu_0 = 4.0 * np.pi * 1e-7


def _transform_coordinates(dx, dy, dz, orientation1, orientation2):
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


def Mnm(omega, First_ring, Second_ring, Data=None):
    """Calculate retarded mutual inductance with canonical ring geometry."""
    if Data is None:
        Data = {}

    dx = Second_ring.x - First_ring.x
    dy = Second_ring.y - First_ring.y
    dz = Second_ring.z - First_ring.z
    r1 = First_ring.r
    r2 = Second_ring.r
    width = First_ring.w * 0.7

    key = (
        f"{dx} {dy} {dz} {r1} {r2} "
        f"{First_ring.pos}{Second_ring.pos} {omega}"
    )
    reverse_key = (
        f"{-dx} {-dy} {-dz} {r2} {r1} "
        f"{Second_ring.pos}{First_ring.pos} {omega}"
    )
    if key in Data:
        return Data[key]
    if reverse_key in Data:
        return Data[reverse_key]
    if dx == 0 and dy == 0 and dz == 0:
        Data[key] = 0.0
        return 0.0

    parallel, x, y, z = _transform_coordinates(
        dx, dy, dz, First_ring.pos, Second_ring.pos
    )
    if parallel:
        function = full_retarded_old.L_parallel
    else:
        function = full_retarded_old.L_orthogonal

    if width:
        effective_width = width
        radii1 = (r1 - effective_width / 2, r1 + effective_width / 2)
        radii2 = (r2 - effective_width / 2, r2 + effective_width / 2)
        value = sum(
            function(omega, x, y, z, radius1, radius2)
            for radius1 in radii1
            for radius2 in radii2
        ) / 4
    else:
        value = function(omega, x, y, z, r1, r2)
    value *= mu_0
    Data[key] = value
    Data[reverse_key] = value
    return value


def Matrix(omega, rings, Data=None):
    """Build the symmetric legacy retarded mutual-inductance matrix."""
    if Data is None:
        Data = {}

    matrix = np.zeros((len(rings), len(rings)), dtype=np.complex128)
    for row in tqdm(range(len(rings))):
        for column in range(row, len(rings)):
            value = Mnm(omega, rings[row], rings[column], Data)
            matrix[row, column] = value
            matrix[column, row] = value
    return matrix
