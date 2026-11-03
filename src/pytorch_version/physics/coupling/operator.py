"""Torch implementations of dense and FFT mutual-inductance operators."""

from dataclasses import dataclass
from itertools import product
from typing import Literal

import torch

Tensor = torch.Tensor
InteractionModel = Literal["quasistatic", "full_retarded", "approx_retarded"]
Orientation = Literal["x", "y", "z"]

MU_0 = 4.0 * torch.pi * 1e-7
C = 3e8
GAMMA = 0.7
_QUADRATURE_POINTS = 128
_PAIR_BATCH_SIZE = 64


def _transform_displacement(
    displacement: Tensor,
    orientation1: Orientation,
    orientation2: Orientation,
) -> tuple[bool, Tensor]:
    dx, dy, dz = displacement.unbind(dim=-1)
    if orientation1 == orientation2:
        if orientation1 == "z":
            coordinates = (dx, dy, dz)
        elif orientation1 == "y":
            coordinates = (dx, -dz, dy)
        else:
            coordinates = (-dz, dy, dx)
        return True, torch.stack(coordinates, dim=-1)

    transforms = {
        ("z", "y"): (dx, dy, dz),
        ("z", "x"): (dy, dx, dz),
        ("y", "z"): (dx, dz, dy),
        ("y", "x"): (-dz, dx, dy),
        ("x", "z"): (dy, dz, dx),
        ("x", "y"): (dz, dy, dx),
    }
    return False, torch.stack(transforms[orientation1, orientation2], dim=-1)


def _pair_integral(
    displacement: Tensor,
    radius1: float,
    radius2: float,
    *,
    parallel: bool,
    quasistatic: bool,
    omega: float,
) -> Tensor:
    """Vectorized periodic trapezoid integration for a batch of ring pairs."""
    device = displacement.device
    dtype = torch.float64
    dx, dy, dz = displacement.to(dtype=dtype).unbind(dim=-1)
    phi = (
        2.0
        * torch.pi
        * torch.arange(_QUADRATURE_POINTS, dtype=dtype, device=device)
        / _QUADRATURE_POINTS
    )
    phi1 = phi[None, :, None]
    phi2 = phi[None, None, :]
    dx, dy, dz = (value[:, None, None] for value in (dx, dy, dz))
    r1 = torch.as_tensor(radius1, dtype=dtype, device=device)
    r2 = torch.as_tensor(radius2, dtype=dtype, device=device)

    if parallel:
        a = (
            dz**2 + dx**2 + dy**2
            + 2.0 * r2 * (dx * torch.cos(phi2) + dy * torch.sin(phi2))
            + r1**2 + r2**2
        )
        b = -2.0 * r1 * (dx + r2 * torch.cos(phi2))
        c = -2.0 * r1 * (dy + r2 * torch.sin(phi2))
        distance_squared = a + b * torch.cos(phi1) + c * torch.sin(phi1)
        projection = torch.cos(phi2 - phi1)
        sign = 1.0
    else:
        a = (
            dz**2 + dx**2 + dy**2
            - 2.0 * r2 * (dx * torch.sin(phi2) + dz * torch.cos(phi2))
            + r1**2 + r2**2
        )
        b = -2.0 * dy * r1
        c = 2.0 * r1 * (r2 * torch.sin(phi2) - dx)
        distance_squared = a + b * torch.cos(phi1) + c * torch.sin(phi1)
        projection = torch.cos(phi1) * torch.cos(phi2)
        sign = 1.0 if quasistatic else -1.0

    distance = torch.sqrt(
        torch.clamp(distance_squared, min=torch.finfo(dtype).tiny)
    )
    phase = omega / C * distance
    kernel = projection * torch.polar(torch.ones_like(phase), phase) / distance
    return (
        sign
        * kernel.sum(dim=(-2, -1))
        * torch.pi
        * r1
        * r2
        / (_QUADRATURE_POINTS**2)
    )


def _mutual_for_pairs(
    displacement: Tensor,
    orientation1: Orientation,
    orientation2: Orientation,
    *,
    radius1: float,
    radius2: float,
    width: float,
    model: InteractionModel,
    omega: float | None,
) -> Tensor:
    if model == "approx_retarded":
        raise NotImplementedError(
            "The approx_retarded model has not yet been ported to the Torch backend."
        )
    if model not in ("quasistatic", "full_retarded"):
        raise ValueError(f"Unsupported interaction model: {model!r}")
    if model == "full_retarded" and omega is None:
        raise ValueError("omega is required for the full_retarded model")

    zero_displacement = torch.all(displacement == 0, dim=-1)
    safe_displacement = torch.where(
        zero_displacement[:, None],
        torch.tensor(
            [4.0 * max(radius1, radius2), 0.0, 0.0],
            dtype=torch.float64,
            device=displacement.device,
        ),
        displacement,
    )
    parallel, local_displacement = _transform_displacement(
        safe_displacement, orientation1, orientation2
    )
    angular_frequency = 0.0 if model == "quasistatic" else float(omega)
    if width:
        half_width = width * GAMMA / 2.0
        radii1 = (radius1 - half_width, radius1 + half_width)
        radii2 = (radius2 - half_width, radius2 + half_width)
        results = [
            _pair_integral(
                local_displacement,
                current_radius1,
                current_radius2,
                parallel=parallel,
                quasistatic=model == "quasistatic",
                omega=angular_frequency,
            )
            for current_radius1, current_radius2 in product(radii1, radii2)
        ]
        value = torch.stack(results).mean(dim=0)
    else:
        value = _pair_integral(
            local_displacement,
            radius1,
            radius2,
            parallel=parallel,
            quasistatic=model == "quasistatic",
            omega=angular_frequency,
        )
    value = value.to(torch.complex128) * MU_0
    return torch.where(zero_displacement, torch.zeros_like(value), value)


def _validate_model(model: InteractionModel, omega: float | None) -> None:
    if model == "approx_retarded":
        raise NotImplementedError(
            "The approx_retarded model has not yet been ported to the Torch backend."
        )
    if model not in ("quasistatic", "full_retarded"):
        raise ValueError(f"Unsupported interaction model: {model!r}")
    if model == "full_retarded" and omega is None:
        raise ValueError("omega is required for the full_retarded model")


def mutual_inductance(
    *,
    displacement: tuple[float, float, float] | Tensor,
    r1: float,
    r2: float,
    width: float,
    orientation1: Orientation,
    orientation2: Orientation,
    model: InteractionModel = "quasistatic",
    omega: float | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Calculate mutual inductance for one pair of rings."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    delta = torch.as_tensor(displacement, dtype=torch.float64, device=target_device)
    if delta.shape != (3,):
        raise ValueError("displacement must have shape (3,)")
    _validate_model(model, omega)
    if not torch.count_nonzero(delta).item():
        return torch.zeros((), dtype=torch.complex128, device=target_device)

    return _mutual_for_pairs(
        delta.unsqueeze(0),
        orientation1,
        orientation2,
        radius1=r1,
        radius2=r2,
        width=width,
        model=model,
        omega=omega,
    )[0]


def mutual_inductance_matrix(
    centers: Tensor,
    spacings: tuple[float, float, float] | Tensor,
    orientations: tuple[Orientation, ...] | list[Orientation],
    *,
    radius: float,
    width: float,
    model: InteractionModel = "quasistatic",
    omega: float | None = None,
) -> Tensor:
    """Build the symmetric dense mutual-inductance matrix."""
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    if len(orientations) != centers.shape[0]:
        raise ValueError("orientations must have shape (N,)")
    _validate_model(model, omega)

    device = centers.device
    n_rings = centers.shape[0]
    matrix = torch.zeros(
        (n_rings, n_rings), dtype=torch.complex128, device=device
    )
    if n_rings < 2:
        return matrix

    scales = torch.as_tensor(spacings, dtype=torch.float64, device=device) / 2.0
    physical_centers = centers.to(torch.float64) * scales
    rows, cols = torch.triu_indices(n_rings, n_rings, offset=1, device=device)
    displacements = physical_centers[cols] - physical_centers[rows]

    orientation_ids = torch.tensor(
        [{"x": 0, "y": 1, "z": 2}[axis] for axis in orientations],
        dtype=torch.int64,
        device=device,
    )
    axis_ids = {"x": 0, "y": 1, "z": 2}
    pairs = sorted(set(product(set(orientations), repeat=2)))
    for orientation1, orientation2 in pairs:
        mask = (
            (orientation_ids[rows] == axis_ids[orientation1])
            & (orientation_ids[cols] == axis_ids[orientation2])
        )
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        if not selected.numel():
            continue
        values = torch.cat(
            [
                _mutual_for_pairs(
                    displacements[batch],
                    orientation1,
                    orientation2,
                    radius1=radius,
                    radius2=radius,
                    width=width,
                    model=model,
                    omega=omega,
                )
                for batch in selected.split(_PAIR_BATCH_SIZE)
            ]
        )
        pair_rows, pair_cols = rows[selected], cols[selected]
        matrix[pair_rows, pair_cols] = values
        matrix[pair_cols, pair_rows] = values

    return matrix


@dataclass(frozen=True)
class _GridSpec:
    shape: tuple[int, int, int]
    start: tuple[int, int, int]


@dataclass(frozen=True)
class _FFTKernel:
    target_shape: tuple[int, int, int]
    source_shape: tuple[int, int, int]
    fft_shape: tuple[int, int, int]
    spectrum: Tensor


class TorchLinearOperator:
    """Matrix-free Torch operator interface for iterative solvers."""

    def __init__(self, size: int, device: torch.device):
        self.shape = (size, size)
        self.device = device
        self.dtype = torch.complex128
        self._matvec = None

    def set_matvec(self, matvec) -> None:
        self._matvec = matvec

    def __matmul__(self, vector: Tensor) -> Tensor:
        if self._matvec is None:
            raise RuntimeError("operator matvec has not been initialized")
        return self._matvec(vector)


def _grid_spec(
    orientation: Orientation,
    counts: tuple[int, int, int],
    meta_atom_type: str,
) -> _GridSpec:
    normal_axis = "xyz".index(orientation)
    shape = []
    start = []
    for axis, count in enumerate(counts):
        if axis != normal_axis or meta_atom_type == "center":
            shape.append(count)
            start.append(1)
        elif meta_atom_type == "border":
            shape.append(count + 1)
            start.append(0)
        elif meta_atom_type == "open":
            shape.append(max(count - 1, 0))
            start.append(2)
        else:
            raise ValueError(f"Unsupported meta-atom type {meta_atom_type!r}")
    return _GridSpec(tuple(shape), tuple(start))


def _fft_size(length: int) -> int:
    return 1 << (length - 1).bit_length()


def _build_fft_kernel(
    target: Orientation,
    source: Orientation,
    target_spec: _GridSpec,
    source_spec: _GridSpec,
    *,
    spacings: tuple[float, float, float],
    radius: float,
    width: float,
    model: InteractionModel,
    omega: float | None,
    device: torch.device,
) -> _FFTKernel:
    target_shape = target_spec.shape
    source_shape = source_spec.shape
    kernel_shape = tuple(
        target_shape[axis] + source_shape[axis] - 1 for axis in range(3)
    )
    fft_shape = tuple(_fft_size(length) for length in kernel_shape)
    delta_axes = [
        2 * (source_shape[axis] - 1 - torch.arange(kernel_shape[axis], device=device))
        + source_spec.start[axis]
        - target_spec.start[axis]
        for axis in range(3)
    ]
    delta_grid = torch.stack(torch.meshgrid(*delta_axes, indexing="ij"), dim=-1)
    scale = torch.tensor(spacings, dtype=torch.float64, device=device) / 2.0
    displacements = delta_grid.reshape(-1, 3).to(torch.float64) * scale

    kernel_parts = [
        _mutual_for_pairs(
            batch,
            target,
            source,
            radius1=radius,
            radius2=radius,
            width=width,
            model=model,
            omega=omega,
        )
        for batch in displacements.split(_PAIR_BATCH_SIZE)
    ]
    kernel = torch.cat(kernel_parts).reshape(kernel_shape)
    spectrum = torch.fft.fftn(kernel, s=fft_shape)
    return _FFTKernel(target_shape, source_shape, fft_shape, spectrum)


def build_fft_operator(
    centers: Tensor,
    orientations: tuple[Orientation, ...] | list[Orientation],
    *,
    counts: tuple[int, int, int],
    spacings: tuple[float, float, float],
    type: str,
    radius: float,
    width: float,
    model: InteractionModel = "quasistatic",
    omega: float | None = None,
) -> TorchLinearOperator:
    """Build FFT convolution for mutual induction on the centers' device."""
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    if len(orientations) != centers.shape[0]:
        raise ValueError("orientations must have shape (N,)")
    _validate_model(model, omega)
    device = centers.device
    n_rings = centers.shape[0]
    active_orientations = tuple(axis for axis in "xyz" if axis in orientations)
    specs = {
        axis: _grid_spec(axis, counts, type) for axis in active_orientations
    }
    active_indices = {}
    for axis in active_orientations:
        active = torch.tensor(
            [i for i, current in enumerate(orientations) if current == axis],
            dtype=torch.int64,
            device=device,
        )
        origin = torch.tensor(specs[axis].start, dtype=torch.int64, device=device)
        indices = (centers[active].to(torch.int64) - origin) // 2
        active_indices[axis] = (
            active,
            tuple(indices[:, dimension] for dimension in range(3)),
        )

    kernels = {}
    for target, source in product(active_orientations, repeat=2):
        kernels[target, source] = _build_fft_kernel(
            target,
            source,
            specs[target],
            specs[source],
            spacings=spacings,
            radius=radius,
            width=width,
            model=model,
            omega=omega,
            device=device,
        )

    def matvec(vector: Tensor) -> Tensor:
        vector = vector.to(device=device, dtype=torch.complex128).reshape(-1)
        if vector.numel() != n_rings:
            raise ValueError(f"expected a vector of length {n_rings}")
        source_grids = {}
        for axis in active_orientations:
            active, indices = active_indices[axis]
            grid = torch.zeros(specs[axis].shape, dtype=torch.complex128, device=device)
            grid[indices] = vector[active]
            source_grids[axis] = grid

        result = torch.zeros(n_rings, dtype=torch.complex128, device=device)
        for target in active_orientations:
            accumulated = torch.zeros(
                specs[target].shape, dtype=torch.complex128, device=device
            )
            for source in active_orientations:
                kernel = kernels[target, source]
                convolution = torch.fft.ifftn(
                    torch.fft.fftn(source_grids[source], s=kernel.fft_shape)
                    * kernel.spectrum
                )
                start = tuple(size - 1 for size in kernel.source_shape)
                crop = tuple(
                    slice(start[axis], start[axis] + kernel.target_shape[axis])
                    for axis in range(3)
                )
                accumulated += convolution[crop]
            active, indices = active_indices[target]
            result[active] = accumulated[indices]
        return result

    operator = TorchLinearOperator(n_rings, device)
    operator.set_matvec(matvec)
    return operator
