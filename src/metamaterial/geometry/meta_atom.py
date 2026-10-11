"""Local geometry of a single meta-atom."""

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class MetaAtom:
    """Geometric representation of a single meta-atom

    Parameters
    ----------
    Dx, Dy, Dz:
        Cell dimensions along the x, y, and z axes
    orientations:
        String containing unique allowed ring orientations: z, y, x
    type:
        Meta-atom geometry: 'border', 'open', or 'center'
    """

    Dx: float
    Dy: float
    Dz: float
    orientations: str
    type: str

    centers: FloatArray = field(init=False, repr=False, compare=False)
    normals: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_orientations()
        self._validate_type()

        centers, normals = self._build_geometry()

        # Store independent arrays and prevent accidental modification.
        centers.setflags(write=False)
        normals.setflags(write=False)

        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "normals", normals)

    def _validate_dimensions(self) -> None:
        dimensions = {
            "Dx": self.Dx,
            "Dy": self.Dy,
            "Dz": self.Dz,
        }

        for name, value in dimensions.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a number")

            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

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
            raise ValueError(
                "type must be one of 'border', 'open', or 'center'"
            )

    def _build_geometry(self) -> tuple[FloatArray, FloatArray]:
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

            # 'border' and 'open' share the same local geometry
            for sign in (-1.0, 1.0):
                center = [0.0, 0.0, 0.0]
                center["xyz".index(axis)] = (
                    sign * half_dimensions[axis]
                )

                centers.append(tuple(center))
                normals.append(
                    tuple(component for component in normal)
                )

        return (
            np.asarray(centers, dtype=np.float64).reshape(-1, 3),
            np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        )

    @property
    def n_rings(self) -> int:
        """Number of rings in the local meta-atom geometry"""
        return self.centers.shape[0]
