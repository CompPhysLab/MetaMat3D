from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from numpy.typing import NDArray
import src.metamaterial.solvers.fast as fast
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry, build_geometry
ComplexArray = NDArray[np.complex128]

@dataclass
class BenchmarkResult:
    use_preconditioner: bool
    elapsed_seconds: list[float]
    iterations: list[int]
    currents: ComplexArray

    @property
    def min_seconds(self) -> float:
        return float(np.min(self.elapsed_seconds))

    @property
    def min_iterations(self) -> float:
        return float(np.min(self.iterations))

def benchmark_solver(
    config: SimulationConfig,
    geometry: Geometry,
    *,
    repeats: int = 3,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Compare GMRES with and without the nearest-neighbor preconditioner."""

    results: list[BenchmarkResult] = []

    original_gmres = fast.gmres

    for use_preconditioner in (False, True):
        times: list[float] = []
        iteration_counts: list[int] = []
        last_currents = None

        try:
            for run_index in range(repeats):
                iteration_count = 0

                def counted_gmres(*args, **kwargs):
                    nonlocal iteration_count

                    user_callback = kwargs.pop("callback", None)
                    kwargs["callback_type"] = "pr_norm"

                    def callback(residual_norm):
                        nonlocal iteration_count
                        iteration_count += 1

                        if user_callback is not None:
                            user_callback(residual_norm)

                    kwargs["callback"] = callback
                    return original_gmres(*args, **kwargs)

                fast.gmres = counted_gmres

                start = perf_counter()
                last_currents = fast.solve_system(
                    config,
                    geometry,
                    use_preconditioner=use_preconditioner,
                )
                elapsed = perf_counter() - start

                # The first run can include one-time library initialization.
                # Keep it for correctness, but exclude it from timing statistics.
                if run_index > 0:
                    times.append(elapsed)
                    iteration_counts.append(iteration_count)

        finally:
            fast.gmres = original_gmres

        if not times:
            raise ValueError("Use repeats >= 2 to measure performance.")

        results.append(
            BenchmarkResult(
                use_preconditioner=use_preconditioner,
                elapsed_seconds=times,
                iterations=iteration_counts,
                currents=last_currents,
            )
        )

    return results[0], results[1]

def print_comparison(
    plain: BenchmarkResult,
    preconditioned: BenchmarkResult,
) -> None:
    """Print a compact comparison of the two solver configurations."""

    print("\n=== GMRES benchmark ===")
    print(f"{'Metric':<32}{'Without P':>15}{'With P':>15}")
    print("-" * 62)

    print(
        f"{'min total time, s':<32}"
        f"{plain.min_seconds:>15.4f}"
        f"{preconditioned.min_seconds:>15.4f}"
    )
    print(
        f"{'min GMRES iterations':<32}"
        f"{plain.min_iterations:>15.1f}"
        f"{preconditioned.min_iterations:>15.1f}"
    )

    speedup = plain.min_seconds / preconditioned.min_seconds
    print(f"\nSpeedup: {speedup:.3f}x")

    norm_plain = np.linalg.norm(plain.currents)
    norm_difference = np.linalg.norm(
        plain.currents - preconditioned.currents
    )

    relative_difference = norm_difference / max(norm_plain, np.finfo(float).tiny)
    print(f"Relative difference in currents: {relative_difference:.3e}")

name='test retarded sphere'
R = 0.002
L = 13.459 * 10 ** -4
C = 470 * 10 ** -5
radius = 4.935 * 10 ** -3
width = 0
Dx = Dy = Dz = 15 * 10 ** -3
orientations = 'zyx'
type = 'border'
Nx, Ny, Nz = 5, 5, 5
packing = 'rectangle'
interaction_model = 'full_retarded'
H_0 = 1
H_direction = (0, 0, 1)
k_direction = (1, 0, 0)
omega_0 = 1 / np.sqrt(L * C)
omegas = np.linspace(omega_0*0.9, omega_0*1.1, 20)
solver = 'fast'
save_currents = True
save_polarizability = False
save_extinction_cross_section = True

config = SimulationConfig(name=name, R=R, L=L, C=C, radius=radius, width=width, Dx=Dx, Dy=Dy, Dz=Dz,
                          orientations=orientations, type=type, Nx=Nx, Ny=Ny, Nz=Nz, packing=packing, interaction_model=interaction_model,
                          H_0=H_0, H_direction=H_direction, k_direction=k_direction, omegas=omegas, solver=solver, save_currents=save_currents, save_polarizability=save_polarizability, save_extinction_cross_section=save_extinction_cross_section)
geometry = build_geometry(config)

plain, preconditioned = benchmark_solver(config, geometry, repeats=4)
print_comparison(plain, preconditioned)