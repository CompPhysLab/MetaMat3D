import torch
from tqdm import tqdm

from src.pytorch_version.config import SimulationConfig
from src.pytorch_version.geometry import Geometry, orientation_normals
from src.pytorch_version.physics import (
    build_rhs,
    self_inductance,
)
from src.pytorch_version.physics.coupling.operator import (
    TorchLinearOperator,
    build_fft_operator,
)
from src.pytorch_version.solvers.preconditioners import (
    build_jacobi_preconditioner,
)

Tensor = torch.Tensor


def _complex_givens(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Return a unitary rotation that maps ``(a, b)`` to ``(r, 0)``."""
    abs_a = torch.abs(a)
    norm = torch.hypot(abs_a, torch.abs(b))
    safe_norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    safe_abs_a = torch.where(abs_a == 0, torch.ones_like(abs_a), abs_a)
    phase = torch.where(abs_a == 0, torch.ones_like(a), a / safe_abs_a)
    cosine = abs_a / safe_norm
    sine = phase * torch.conj(b) / safe_norm
    cosine = torch.where(norm == 0, torch.ones_like(cosine), cosine)
    sine = torch.where(norm == 0, torch.zeros_like(sine), sine)
    return cosine, sine


def _gmres(
    matrix,
    rhs: Tensor,
    *,
    rtol: float = 1e-10,
    restart: int | None = None,
    max_iterations: int | None = None,
    initial_guess: Tensor | None = None,
    omega: Tensor | float | None = None,
) -> Tensor:
    """Solve with restarted GMRES using SciPy-compatible default limits."""
    n = rhs.numel()
    if n == 0:
        return torch.zeros_like(rhs)
    if restart is None:
        restart = min(20, n)
    if restart < 1:
        raise ValueError("restart must be positive")
    if max_iterations is None:
        max_iterations = 10 * n * restart
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    solution = (
        torch.zeros_like(rhs) if initial_guess is None else initial_guess.clone()
    )
    rhs_norm = torch.linalg.vector_norm(rhs).item()
    tolerance = rtol * rhs_norm
    iterations = 0

    while iterations < max_iterations:
        residual = rhs - matrix @ solution
        beta = torch.linalg.vector_norm(residual)
        if beta.item() <= tolerance:
            return solution

        basis: list[Tensor] = [residual / beta]
        hessenberg = torch.zeros(
            (restart + 1, restart),
            dtype=matrix.dtype,
            device=matrix.device,
        )
        residual_coefficients = torch.zeros(
            restart + 1, dtype=matrix.dtype, device=matrix.device
        )
        residual_coefficients[0] = beta
        rotations: list[tuple[Tensor, Tensor]] = []
        base_solution = solution
        steps = min(restart, max_iterations - iterations, n)

        for column in range(steps):
            vector = matrix @ basis[column]
            arnoldi_norm = torch.linalg.vector_norm(vector)
            for row in range(column + 1):
                coefficient = torch.vdot(basis[row], vector)
                hessenberg[row, column] = coefficient
                vector = vector - coefficient * basis[row]

            next_norm = torch.linalg.vector_norm(vector)
            hessenberg[column + 1, column] = next_norm

            for row, (cosine, sine) in enumerate(rotations):
                top = hessenberg[row, column].clone()
                bottom = hessenberg[row + 1, column].clone()
                hessenberg[row, column] = cosine * top + sine * bottom
                hessenberg[row + 1, column] = (
                    -torch.conj(sine) * top + cosine * bottom
                )

            cosine, sine = _complex_givens(
                hessenberg[column, column],
                hessenberg[column + 1, column],
            )
            rotations.append((cosine, sine))
            top = hessenberg[column, column].clone()
            bottom = hessenberg[column + 1, column].clone()
            hessenberg[column, column] = cosine * top + sine * bottom
            hessenberg[column + 1, column] = 0
            top = residual_coefficients[column].clone()
            bottom = residual_coefficients[column + 1].clone()
            residual_coefficients[column] = cosine * top + sine * bottom
            residual_coefficients[column + 1] = (
                -torch.conj(sine) * top + cosine * bottom
            )

            used = column + 1
            iterations += 1

            next_norm_value, estimated_residual, arnoldi_norm_value = (
                torch.stack(
                    (
                        next_norm,
                        torch.abs(residual_coefficients[column + 1]),
                        arnoldi_norm,
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            breakdown = next_norm_value <= (
                torch.finfo(torch.float64).eps * arnoldi_norm_value
            )
            if estimated_residual <= tolerance or breakdown:
                triangular = hessenberg[:used, :used]
                coefficients = torch.linalg.solve_triangular(
                    triangular,
                    residual_coefficients[:used, None],
                    upper=True,
                )[:, 0]
                vectors = torch.stack(basis[:used], dim=1)
                candidate = base_solution + vectors @ coefficients
                residual = rhs - matrix @ candidate
                residual_norm = torch.linalg.vector_norm(residual).item()
                if residual_norm <= tolerance:
                    return candidate
                relative_residual = residual_norm / max(
                    rhs_norm, torch.finfo(torch.float64).tiny
                )
                frequency = (
                    ""
                    if omega is None
                    else f" for omega={float(omega):.9g}"
                )
                if breakdown:
                    raise RuntimeError(
                        f"GMRES stagnated{frequency} after {iterations} "
                        f"iterations; residual={residual_norm:.3e} "
                        f"(relative={relative_residual:.3e})."
                    )
                solution = candidate
                break
            basis.append(vector / next_norm)

        else:
            triangular = hessenberg[:steps, :steps]
            coefficients = torch.linalg.solve_triangular(
                triangular,
                residual_coefficients[:steps, None],
                upper=True,
            )[:, 0]
            vectors = torch.stack(basis[:steps], dim=1)
            solution = base_solution + vectors @ coefficients
            continue

    residual_norm = torch.linalg.vector_norm(rhs - matrix @ solution).item()
    relative_residual = residual_norm / max(
        rhs_norm, torch.finfo(torch.float64).tiny
    )
    frequency = (
        "" if omega is None else f" for omega={float(omega):.9g}"
    )
    raise RuntimeError(
        f"GMRES did not converge{frequency} after {iterations}/"
        f"{max_iterations} iterations; residual={residual_norm:.3e} "
        f"(relative={relative_residual:.3e}, tolerance={rtol:.3e})."
    )


class _SystemOperator:
    def __init__(
        self,
        mutual_operator: TorchLinearOperator,
        diagonal: Tensor,
        *,
        scale: bool,
    ):
        self.mutual_operator = mutual_operator
        self.diagonal = diagonal
        self.scale = scale
        self.shape = mutual_operator.shape
        self.device = mutual_operator.device
        self.dtype = torch.complex128

    def __matmul__(self, vector: Tensor) -> Tensor:
        result = self.diagonal * vector - self.mutual_operator @ vector
        return result / self.diagonal if self.scale else result


def solve_system(
    config: SimulationConfig,
    geometry: Geometry,
    *,
    device: torch.device | str | None = None,
    use_preconditioner: bool = True,
) -> Tensor:
    """Solve with Torch GMRES; ``use_preconditioner`` applies Jacobi scaling."""
    target_device = (
        torch.device(device) if device is not None else geometry.centers.device
    )
    centers = geometry.centers.to(device=target_device)
    normals = orientation_normals(geometry.orientations, device=target_device)
    frequencies = torch.tensor(
        config.omegas, dtype=torch.float64, device=target_device
    )
    spacings = (config.Dx, config.Dy, config.Dz)
    currents = torch.empty(
        (centers.shape[0], frequencies.numel()),
        dtype=torch.complex128,
        device=target_device,
    )
    mutual_operator = None
    if config.interaction_model == "quasistatic":
        mutual_operator = build_fft_operator(
            centers,
            geometry.orientations,
            counts=(config.Nx, config.Ny, config.Nz),
            spacings=spacings,
            type=config.type,
            radius=config.radius,
            width=config.width,
            model=config.interaction_model,
        )

    previous_solution = None
    for index, omega in enumerate(tqdm(frequencies, desc="Solving frequencies")):
        if mutual_operator is None:
            mutual_operator = build_fft_operator(
                centers,
                geometry.orientations,
                counts=(config.Nx, config.Ny, config.Nz),
                spacings=spacings,
                type=config.type,
                radius=config.radius,
                width=config.width,
                model=config.interaction_model,
                omega=float(omega.item()),
            )
        diagonal_value = self_inductance(
            omega, config.R, config.L, config.C, device=target_device
        )
        rhs = build_rhs(
            centers,
            spacings,
            normals,
            config.radius,
            config.H_0,
            config.H_direction,
            model=config.interaction_model,
            k_direction=config.k_direction,
            omega=omega,
        )

        inverse_diagonal = (
            build_jacobi_preconditioner(diagonal_value)
            if use_preconditioner
            else None
        )
        system = _SystemOperator(
            mutual_operator,
            diagonal_value,
            scale=inverse_diagonal is not None,
        )
        scaled_rhs = rhs * inverse_diagonal if inverse_diagonal is not None else rhs
        solution = _gmres(
            system,
            scaled_rhs,
            initial_guess=previous_solution,
            omega=omega,
        )
        currents[:, index] = solution
        previous_solution = solution
        if config.interaction_model != "quasistatic":
            mutual_operator = None

    return currents
