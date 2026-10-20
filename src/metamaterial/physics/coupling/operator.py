from functools import lru_cache
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from itertools import product
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.sparse.linalg import LinearOperator
from . import approx_retarded, full_retarded, quasistatic

mu_0 = 4 * np.pi * 1e-7
gamma = 0.7 #gamma-factor
interaction_model = Literal["quasistatic", "full_retarded", "approx_retarded"]
Orientation = Literal["x", "y", "z"]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
StrArray = NDArray[np.str_]
IntArray = NDArray[np.int64]

_MODELS = {
    "quasistatic": quasistatic,
    "full_retarded": full_retarded,
    "approx_retarded": approx_retarded,
}

def _transform_coordinates(dx: float, dy: float, dz: float, orientation1: Orientation, orientation2: Orientation) -> tuple[bool, float, float, float]:
    """Transforms displacement into coordinates expected by the model

    Parameters
    ----------
    dx, dy, dz : float
        Displacement coordinates in the global frame
    orientation1, orientation2 : Orientation
        Orientations of the two rings, each one of 'x', 'y', 'z'
    
    Returns
    -------
        parallel : bool
            Whether the rings have the same orientation
        x, y, z : float
            Transformed displacement coordinates

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

def _mutual_inductance(model: interaction_model, omega: float | None, x: float, y: float, z: float, r1: float, r2: float, parallel: bool, width: float = 0.0) -> complex:
    """Calculates the mutual inductance

    Parameters
    ----------
    model : interaction_model
        Interaction model to use for the calculation
    omega : float | None
        Angular frequency (required for non-quasistatic models)
    x, y, z : float
        Local coordinates
    r1, r2 : float
        Radii of the two rings
    parallel : bool
        Whether the rings are parallel
    width : float, optional
        Width of the strip, by default 0

    Returns
    -------
    complex
        Mutual inductance between the two rings
    """
    module = _MODELS[model]
    function_name = "L_parallel" if parallel else "L_orthogonal"
    function = getattr(module, function_name)
    if model == "quasistatic":
        args = (x, y, z)
    else:
        args = (omega, x, y, z)
    if width:
        w = width * gamma
        r1_values = (r1 - w / 2, r1 + w / 2)
        r2_values = (r2 - w / 2, r2 + w / 2)
        values = [
            function(*args, radius1, radius2)
            for radius1 in r1_values
            for radius2 in r2_values
        ]

        return complex(sum(values) / 4)
    return complex(function(*args, r1, r2))

@lru_cache(maxsize=100_000)
def _cached_mutual_inductance(model: interaction_model, omega: float | None, x: float, y: float, z: float, r1: float, r2: float, parallel: bool, width: float) -> complex:
    """Cache the resulting mutual inductance for performance with local coordinates"""
    return _mutual_inductance(model, omega, x, y, z, r1, r2, parallel, width)

def mutual_inductance(*, displacement: tuple[float, float, float], r1: float, r2: float, width: float, orientation1: Orientation, orientation2: Orientation, model: interaction_model = "quasistatic", omega: float | None = None) -> complex:
    """Calculates mutual inductance M12

    Parameters
    ----------
    displacement : tuple[float, float, float]
        Vector from the center of ring 1 to the center of ring 2
    r1, r2 : float
        Radii of the two rings
    width : float
        Width of the strip
    orientation1, orientation2 : Orientation
        Orientations of the two rings, each one of 'x', 'y', 'z'
    model : interaction_model, optional
        Interaction model to use for the calculation (default: 'quasistatic')
    omega : float | None, optional
        Angular frequency; required for non-quasistatic models

    Returns
    -------
    complex
        Mutual inductance between the two rings
    """
    
    dx, dy, dz = map(float, displacement)
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return 0.0 + 0.0j

    parallel, x, y, z = _transform_coordinates(dx, dy, dz, orientation1, orientation2)
    mutual_inductance = _cached_mutual_inductance(model, None if omega is None else float(omega), x, y, z, r1, r2, parallel, width)
    return complex(mu_0 * mutual_inductance)

def mutual_inductance_matrix(centers: FloatArray, orientations: StrArray, *, radius: float, width: float, model: interaction_model = "quasistatic", omega: float | None = None) -> ComplexArray:
    """Builds the symmetric mutual-inductance matrix

    Parameters
    ----------
    centers: FloatArray
        Ring centers, shape (N, 3)
    orientations: NDArray
        Ring orientations, shape (N,), containing 'x', 'y', 'z'
    radius: float
        Common mean ring radius
    width: float
        Common width of the strip
    model: interaction_model
        Interaction model
    omega: float | None
        Angular frequency; required by non-quasistatic models

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
            value = mutual_inductance(displacement=displacement, r1=radius, r2=radius, width=width, orientation1=str(orientations[i]), orientation2=str(orientations[j]), model=model, omega=omega)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix

def clear_cache() -> None:
    """Clear the operator's mutual-inductance cache"""
    _cached_mutual_inductance.cache_clear()

@dataclass(frozen=True, slots=True)
class _GridSpec:
    shape: tuple[int, int, int]
    start: tuple[int, int, int]
    offset: tuple[float, float, float]

@dataclass(frozen=True, slots=True)
class _FFTKernel:
    target: Orientation
    source: Orientation
    target_shape: tuple[int, int, int]
    source_shape: tuple[int, int, int]
    fft_shape: tuple[int, int, int]
    spectrum: np.ndarray

def _grid_spec(orientation: Orientation, counts: tuple[int, int, int], type: str) -> _GridSpec:
    """Shape and start for 1 orientation
    
    Parameters
    ----------
    orientation: Orientation
        'x', 'y' or 'z'
    counts: tuple[int, int, int]
        Number of cells along each axis
    type: str
        'center', 'border' or 'open'

    Returns
    -------
    _GridSpec
        shape, start and offset for the orientation
    """
    normal_axis = "xyz".index(orientation)
    shape = []
    start = []
    offset = []

    for axis in range(3):
        count = counts[axis]

        if axis != normal_axis or type == "center":
            shape.append(count)
            start.append(0)
            offset.append(0.5)

        elif type == "border":
            shape.append(count + 1)
            start.append(0)
            offset.append(0.0)

        elif type == "open":
            shape.append(max(count - 1, 0))
            start.append(1)
            offset.append(0.0)

    return _GridSpec(shape=tuple(shape), start=tuple(start), offset=tuple(offset))

def _build_fft_kernel(target: Orientation, source: Orientation, target_spec: _GridSpec, source_spec: _GridSpec, *, spacing: tuple[float, float, float], radius: float, width: float, model: interaction_model, omega: float | None) -> _FFTKernel:
    """Builds FFT kernel for a pair of orientations
    
    Parameters
    ----------
    target: Orientation
        'x', 'y' or 'z'
    source: Orientation
        'x', 'y' or 'z'
    target_spec: _GridSpec
        grid spec for target orientation
    source_spec: _GridSpec
        grid spec for source orientation
    spacing: tuple[float, float, float]
        Dx, Dy, Dz
    radius: float
        Radius of the ring
    width: float
        Width of the strip
    model: interaction_model
        Interaction model
    omega: float | None
        Angular frequency; required by non-quasistatic models

    Returns
    -------
    _FFTKernel
        FFT kernel for the pair of orientations
    """
    target_shape = target_spec.shape
    source_shape = source_spec.shape

    kernel_shape = tuple(
        nt + ns - 1
        for nt, ns in zip(target_shape, source_shape)
    )
    fft_shape = tuple(next_fast_len(n) for n in kernel_shape)

    #kernel index S - 1 + t - s
    #physical displacement from source to target is:
    #source_origin - target_origin + (S - 1 - kernel_index) * spacing
    delta_axes = [
        source_shape[axis] - 1 - np.arange(kernel_shape[axis])
        for axis in range(3)
    ]

    displacement = np.stack(
        np.meshgrid(*[
            delta_axes[axis] * spacing[axis]
            + (
                source_spec.start[axis]
                + source_spec.offset[axis]
                - target_spec.start[axis]
                - target_spec.offset[axis]
            ) * spacing[axis]
            for axis in range(3)
        ], indexing="ij"),
        axis=-1,
    )

    kernel = np.empty(kernel_shape, dtype=np.complex128)

    for index in np.ndindex(kernel_shape):
        kernel[index] = mutual_inductance(
            displacement=displacement[index],
            r1=radius,
            r2=radius,
            width=width,
            orientation1=target,
            orientation2=source,
            model=model,
            omega=omega,
        )

    spectrum = fftn(kernel, s=fft_shape)

    return _FFTKernel(target=target, source=source, target_shape=target_shape, source_shape=source_shape, fft_shape=fft_shape, spectrum=spectrum)

def build_fft_operator(grid_indices: IntArray, orientations: StrArray, *, counts: tuple[int, int, int], spacing: tuple[float, float, float], type: str, radius: float, width: float, model: interaction_model = "quasistatic", omega: float | None = None) -> LinearOperator:
    """Builds a linear operator for the mutual-inductance matrix using FFT-based convolution

    Parameters
    ----------
    grid_indices: IntArray
        Indices of the active rings in the full grid, shape (N, 3)
    orientations: StrArray
        Orientations of the active rings, shape (N,)
    counts: tuple[int, int, int]
        Number of cells along each axis (Nx, Ny, Nz)
    spacing: tuple[float, float, float]
        Grid spacing along each axis (Dx, Dy, Dz)
    type: str
        Type of the grid ('center', 'border', or 'open')
    radius: float
        Radius of the ring
    width: float
        Width of the strip
    model: interaction_model
        Interaction model
    omega: float | None
        Angular frequency; required by non-quasistatic models
    """
    grid_indices = np.asarray(grid_indices, dtype=int)
    orientations = np.asarray(orientations)

    if model == "quasistatic" and omega is not None:
        omega = None

    orientations = orientations.astype("U1")
    unique_orientations = tuple(
        axis for axis in "xyz"
        if np.any(orientations == axis)
    )

    specs = {
        axis: _grid_spec(axis, counts, type)
        for axis in unique_orientations
    }

    active_indices = {}
    for axis in unique_orientations:
        mask = orientations == axis
        spec = specs[axis]
        indices = grid_indices[mask] - np.asarray(spec.start)
        active_indices[axis] = (
            np.flatnonzero(mask),
            tuple(indices[:, dim] for dim in range(3)),
        )

    kernels = {}
    for target, source in product(
        unique_orientations, repeat=2
    ):
        kernels[target, source] = _build_fft_kernel(
            target,
            source,
            specs[target],
            specs[source],
            spacing=spacing,
            radius=radius,
            width=width,
            model=model,
            omega=omega,
        )

    def matvec(vector: np.ndarray) -> np.ndarray:
        '''Defines the matrix-vector product for the linear operator'''
        vector = np.asarray(vector).reshape(-1)
        result = np.zeros(len(vector), dtype=np.complex128)
        source_grids = {}

        for axis in unique_orientations:
            active, indices = active_indices[axis]
            grid = np.zeros(specs[axis].shape, dtype=np.complex128)
            grid[indices] = vector[active]
            source_grids[axis] = grid

        for target in unique_orientations:
            target_spec = specs[target]
            accumulated = np.zeros(
                target_spec.shape, dtype=np.complex128
            )

            for source in unique_orientations:
                kernel = kernels[target, source]
                source_grid = source_grids[source]

                convolution = ifftn(
                    fftn(source_grid, s=kernel.fft_shape)
                    * kernel.spectrum
                )

                start = tuple(n - 1 for n in kernel.source_shape)
                stop = tuple(
                    start[d] + kernel.target_shape[d]
                    for d in range(3)
                )

                crop = tuple(
                    slice(start[d], stop[d])
                    for d in range(3)
                )

                accumulated += convolution[crop]

            active, indices = active_indices[target]
            result[active] = accumulated[indices]

        return result
    
    n = len(grid_indices)

    return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.complex128)
