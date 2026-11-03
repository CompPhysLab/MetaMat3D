"""Local geometry of a single meta-atom."""

from dataclasses import dataclass, field
import math
import numbers

import torch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class MetaAtom:
    """Geometric representation of a single meta-atom."""

    Dx: float
    Dy: float
    Dz: float
    orientations: str
    type: str
    device: torch.device | str = "cpu"

    centers: Tensor = field(init=False, repr=False, compare=False)
    normals: Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_orientations()
        self._validate_type()

        device = torch.device(self.device)
        centers, normals = self._build_geometry(device)

        object.__setattr__(self, "device", device)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "normals", normals)

    def _validate_dimensions(self) -> None:
        dimensions = {
            "Dx": self.Dx,
            "Dy": self.Dy,
            "Dz": self.Dz,
        }

        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a number")

            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def _validate_orientations(self) -> None:
        if not isinstance(self.orientations, str):
            raise TypeError("orientations must be a string")

        if (
            not self.orientations
            or any(axis not in "xyz" for axis in self.orientations)
            or len(set(self.orientations)) != len(self.orientations)
        ):
            raise ValueError(
                "orientations must contain unique axes from 'x', 'y', 'z'"
            )

    def _validate_type(self) -> None:
        if self.type not in {"border", "open", "center"}:
            raise ValueError("type must be one of 'border', 'open', or 'center'")

    def _build_geometry(self, device: torch.device) -> tuple[Tensor, Tensor]:
        centers: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []

        half_dimensions = {
            "x": self.Dx / 2,
            "y": self.Dy / 2,
            "z": self.Dz / 2,
        }

        axis_vectors = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }

        for axis in self.orientations:
            normal = axis_vectors[axis]

            if self.type == "center":
                centers.append((0.0, 0.0, 0.0))
                normals.append(normal)
                continue

            for sign in (-1.0, 1.0):
                center = [0.0, 0.0, 0.0]
                center["xyz".index(axis)] = sign * half_dimensions[axis]
                centers.append(tuple(center))
                normals.append(normal)

        return (
            torch.tensor(centers, dtype=torch.float64, device=device).reshape(-1, 3),
            torch.tensor(normals, dtype=torch.float64, device=device).reshape(-1, 3),
        )

    @property
    def n_rings(self) -> int:
        """Number of rings in the local meta-atom geometry."""
        return self.centers.shape[0]
