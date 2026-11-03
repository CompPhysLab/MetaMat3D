"""Restarted Torch GMRES solver."""

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


def _gmres(
    matrix,
    rhs: Tensor,
    *,
    rtol: float = 1e-10,
    restart: int = 30,
    max_iterations: int = 500,
) -> Tensor:
    """Solve a dense system with restarted Arnoldi/GMRES iterations."""
    n = rhs.numel()
    solution = torch.zeros_like(rhs)
    tolerance = rtol * torch.linalg.vector_norm(rhs).item()
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
        base_solution = solution
        steps = min(restart, max_iterations - iterations, n)

        for column in range(steps):
            vector = matrix @ basis[column]
            for row in range(column + 1):
                coefficient = torch.vdot(basis[row], vector)
                hessenberg[row, column] = coefficient
                vector = vector - coefficient * basis[row]

            next_norm = torch.linalg.vector_norm(vector)
            hessenberg[column + 1, column] = next_norm
            if next_norm.item() > torch.finfo(torch.float64).eps:
                basis.append(vector / next_norm)

            used = column + 1
            target = torch.zeros(
                (used + 1, 1), dtype=matrix.dtype, device=matrix.device
            )
            target[0, 0] = beta
            coefficients = torch.linalg.lstsq(
                hessenberg[: used + 1, :used], target
            ).solution[:, 0]
            vectors = torch.stack(basis[:used], dim=1)
            candidate = base_solution + vectors @ coefficients
            iterations += 1

            residual_norm = torch.linalg.vector_norm(rhs - matrix @ candidate)
            if residual_norm.item() <= tolerance:
                return candidate
            if next_norm.item() <= torch.finfo(torch.float64).eps:
                solution = candidate
                break

        else:
            solution = candidate
            continue

        solution = candidate
        if torch.linalg.vector_norm(rhs - matrix @ solution).item() > tolerance:
            raise RuntimeError(
                f"GMRES stagnated after {iterations} iterations; "
                f"residual={torch.linalg.vector_norm(rhs - matrix @ solution).item():.3e}"
            )

    residual_norm = torch.linalg.vector_norm(rhs - matrix @ solution).item()
    raise RuntimeError(
        f"GMRES did not converge after {max_iterations} iterations; "
        f"residual={residual_norm:.3e}"
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
        solution = _gmres(system, scaled_rhs)
        currents[:, index] = solution
        if config.interaction_model != "quasistatic":
            mutual_operator = None

    return currents
