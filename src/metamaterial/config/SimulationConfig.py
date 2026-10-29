"""Configuration for a single metamaterial simulation."""

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
FloatArray = npt.NDArray[np.float64]

Vector3 = tuple[float, float, float]

@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters describing one simulation run
    This class validates configuration parameters
    """

    # Simulation identification
    name: str

    # Ring parameters
    R: float
    L: float
    C: float
    radius: float
    width: float

    # Meta-atom configuration
    Dx: float
    Dy: float
    Dz: float
    orientations: str
    type: str

    # Sample geometry
    Nx: np.int16
    Ny: np.int16
    Nz: np.int16
    packing: str

    # Interaction model
    interaction_model: str

    # Excitation
    H_0: float
    H_direction: Vector3
    k_direction: Vector3 | None = None

    # Frequency grid
    omegas: FloatArray | None = None

    # Linear-system solver
    solver: str = ""

    # Output options
    save_currents: bool = True
    save_polarizability: bool = True
    save_extinction_cross_section: bool = True

    def __post_init__(self) -> None:
        """Validate parameters and normalize direction vectors"""

        self._validate_name()
        self._validate_ring_parameters()
        self._validate_meta_atom_dimensions()
        self._validate_orientations()
        self._validate_type()
        self._validate_geometry()
        self._validate_interaction_model()
        self._validate_excitation()
        self._validate_omegas()
        self._validate_solver()
        self._validate_output_options()

    # Validation
    def _validate_name(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

    def _validate_ring_parameters(self) -> None:
        parameters = {
            "R": self.R,
            "L": self.L,
            "C": self.C,
            "radius": self.radius,
            "width": self.width,
        }

        for name, value in parameters.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a number")

            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and positive")

        # radius is the mean radius of the ring.
        if self.width >= 2 * self.radius:
            raise ValueError(
                "width must be smaller than twice the mean radius"
            )

    def _validate_meta_atom_dimensions(self) -> None:
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
                raise ValueError(f"{name} must be finite and positive")

            if value <= 2 * (self.radius + self.width / 2):
                raise ValueError(
                    f"{name} must be bigger than twice the outer radius of the ring"
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
        allowed = {"border", "open", "center"}

        if self.type not in allowed:
            raise ValueError(
                f"type must be one of {allowed}"
            )

    def _validate_geometry(self) -> None:
        for name, value in (
            ("Nx", self.Nx),
            ("Ny", self.Ny),
            ("Nz", self.Nz),
        ):
            if isinstance(value, bool) or not isinstance(value, np.int16):
                raise TypeError(f"{name} must be an integer")

            if value <= 0:
                raise ValueError(f"{name} must be positive")

        allowed_packings = {"rectangle", "ellipse", "cylinder_z"}

        if self.packing not in allowed_packings:
            raise ValueError(
                f"packing must be one of {allowed_packings}"
            )

    def _validate_interaction_model(self) -> None:
        if (
            not isinstance(self.interaction_model, str)
            or not self.interaction_model
        ):
            raise ValueError("interaction_model must be a non-empty string")

        allowed_models = {"quasistatic", "full_retarded", "approx_retarded"}

        if self.interaction_model not in allowed_models:
            raise ValueError(
                f"interaction_model must be one of {allowed_models}"
            )

    @staticmethod
    def _normalize_vector(
        vector: Vector3,
        name: str,
    ) -> Vector3:
        if not isinstance(vector, (tuple, list)) or len(vector) != 3:
            raise ValueError(f"{name} must contain exactly three components")

        for component in vector:
            if isinstance(component, (bool, np.bool_)) or not isinstance(
                component, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} components must be numbers")

        values = tuple(float(component) for component in vector)

        if not all(np.isfinite(component) for component in values):
            raise ValueError(f"{name} components must be finite")

        norm = np.sqrt(sum(component**2 for component in values))

        if norm == 0:
            raise ValueError(f"{name} must not be a zero vector")

        return tuple(component / norm for component in values)

    @staticmethod
    def _skalar_product(v1: Vector3, v2: Vector3) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    def _validate_excitation(self) -> None:
        if isinstance(self.H_0, (bool, np.bool_)) or not isinstance(
            self.H_0, (int, float, np.integer, np.floating)
        ):
            raise TypeError("H_0 must be a number")

        if not np.isfinite(self.H_0) or self.H_0 < 0:
            raise ValueError(
                "H_0 must be finite and non-negative"
            )

        object.__setattr__(
            self,
            "H_direction",
            self._normalize_vector(self.H_direction, "H_direction"),
        )

        if self.k_direction is not None:
            object.__setattr__(
                self,
                "k_direction",
                self._normalize_vector(self.k_direction, "k_direction"),
            )

        if self.is_non_quasistatic and self.k_direction is None:
            raise ValueError(
                "k_direction is required for a non-quasistatic model"
            )

        if self.k_direction is not None and self._skalar_product(self.H_direction, self.k_direction) != 0:
            raise ValueError(
                "H_direction must be perpendicular to k_direction"
            )

    @property
    def is_non_quasistatic(self) -> bool:
        """Whether the selected model is non-quasistatic

        """
        model = self.interaction_model
        return model != "quasistatic"

    def _validate_omegas(self) -> None:
        if self.omegas is None:
            raise ValueError("frequencies must be provided")

        omegas = np.asarray(self.omegas, dtype=np.float64)

        if omegas.ndim != 1:
            raise ValueError("frequencies must be a one-dimensional array")

        if omegas.size == 0:
            raise ValueError("frequencies must not be empty")

        if not np.all(np.isfinite(omegas)) or np.any(omegas <= 0):
            raise ValueError(
                "All frequencies must be finite and positive"
            )

        if np.any(np.diff(omegas) <= 0):
            raise ValueError(
                "frequencies must be strictly increasing"
            )

        # Copy to avoid retaining a reference to the caller's array.
        omegas = omegas.copy()
        omegas.setflags(write=False)
        object.__setattr__(self, "omegas", omegas)

    def _validate_solver(self) -> None:
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise ValueError("solver must be a non-empty string")

        solvers = {"straight", "fast"}
        if self.solver not in solvers:
            raise ValueError(f"solver must be one of {solvers}")

    def _validate_output_options(self) -> None:
        for name in (
            "save_currents",
            "save_polarizability",
            "save_extinction_cross_section",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary"""
        data = asdict(self)

        if self.omegas is not None:
            data["omegas"] = self.omegas.tolist()

        return data
