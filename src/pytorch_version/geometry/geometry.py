"""Generation of ordered ring-center geometry."""

from dataclasses import dataclass

import torch
from .packings import get_packing_rule

Tensor = torch.Tensor
Orientation = str

_ALLOWED_TYPES = {"border", "open", "center"}
_ALLOWED_ORIENTATIONS = {"z", "y", "x"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Geometry:
    """Ordered geometry consumed by the solvers.

    ``centers`` stores doubled lattice coordinates in physical x/y/z order.
    Orientation labels remain strings because PyTorch tensors do not support
    string values; numeric geometry and masks are tensors on the selected device.
    """

    centers: Tensor
    orientations: tuple[Orientation, ...]

    def __post_init__(self) -> None:
        n = self.centers.shape[0]
        if self.centers.ndim != 2 or self.centers.shape != (n, 3):
            raise ValueError("centers must have shape (N, 3)")
        if self.centers.dtype != torch.int64:
            raise TypeError("centers must have dtype torch.int64")
        if len(self.orientations) != n:
            raise ValueError("orientations must have shape (N,)")
        if any(orientation not in _ALLOWED_ORIENTATIONS for orientation in self.orientations):
            raise ValueError("orientations must contain only 'x', 'y', or 'z'")


def _coordinates_for_orientation(
    orientation: str,
    type: str,
    counts: tuple[int, int, int],
    *,
    device: torch.device,
) -> tuple[Tensor, ...]:
    """Build doubled lattice-coordinate axes for one ring orientation."""
    if orientation not in _ALLOWED_ORIENTATIONS:
        raise ValueError(f"Unsupported orientation {orientation!r}")
    if type not in _ALLOWED_TYPES:
        raise ValueError(f"Unsupported type {type!r}")

    normal_axis = _AXIS_INDEX[orientation]
    coord_axes = []

    for axis, count in enumerate(counts):
        if axis != normal_axis or type == "center":
            coordinates = 2 * torch.arange(count, dtype=torch.int64, device=device) + 1
        elif type == "border":
            coordinates = 2 * torch.arange(count + 1, dtype=torch.int64, device=device)
        else:
            coordinates = 2 * torch.arange(1, count, dtype=torch.int64, device=device)

        coord_axes.append(coordinates)

    return tuple(coord_axes)


def _flatten_fortran_order(grid: Tensor) -> Tensor:
    """Flatten a 3D x/y/z grid with the x axis varying fastest."""
    return grid.permute(2, 1, 0).reshape(-1)


def build_geometry(
    config,
    *,
    device: torch.device | str | None = None,
) -> Geometry:
    """Build ordered geometry tensors on ``device`` (CPU by default)."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    counts = (int(config.Nx), int(config.Ny), int(config.Nz))
    type = str(config.type)
    orientations = str(config.orientations)
    packing_rule = get_packing_rule(str(config.packing))

    center_blocks: list[Tensor] = []
    orientation_blocks: list[tuple[str, ...]] = []

    for orientation in orientations:
        coord_axes = _coordinates_for_orientation(
            orientation, type, counts, device=target_device
        )
        x, y, z = torch.meshgrid(*coord_axes, indexing="ij")
        centers = torch.stack(
            (
                _flatten_fortran_order(x),
                _flatten_fortran_order(y),
                _flatten_fortran_order(z),
            ),
            dim=1,
        )

        active = packing_rule(centers, counts)
        center_blocks.append(centers[active])
        orientation_blocks.append(
            (orientation,) * int(torch.count_nonzero(active).item())
        )

    if center_blocks:
        centers = torch.cat(center_blocks, dim=0)
        ring_orientations = tuple(
            orientation
            for block in orientation_blocks
            for orientation in block
        )
    else:
        centers = torch.empty((0, 3), dtype=torch.int64, device=target_device)
        ring_orientations = ()

    return Geometry(centers=centers, orientations=ring_orientations)


def orientation_normals(
    orientations: tuple[Orientation, ...] | list[Orientation],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Convert orientation labels to Cartesian unit-normal tensors."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    axis_indices = {"x": 0, "y": 1, "z": 2}
    try:
        indices = [axis_indices[orientation] for orientation in orientations]
    except KeyError as exc:
        raise ValueError(f"Unsupported orientation {exc.args[0]!r}") from exc

    normals = torch.zeros(
        (len(indices), 3), dtype=torch.int64, device=target_device
    )
    if indices:
        rows = torch.arange(len(indices), device=target_device)
        columns = torch.tensor(indices, dtype=torch.int64, device=target_device)
        normals[rows, columns] = 1
    return normals
