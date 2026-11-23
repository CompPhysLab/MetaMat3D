"""Бенчмарк для разных солверов, сравнение numpy/pytorch cpu/pytorch gpu"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.metamaterial.config import SimulationConfig as NumpyConfig
from src.metamaterial.geometry import build_geometry as build_numpy_geometry
from src.metamaterial.physics.coupling import operator as numpy_coupling_operator
from src.metamaterial.solvers import fast as numpy_fast
from src.metamaterial.solvers import straight as numpy_straight
from src.pytorch_version.config import SimulationConfig as TorchConfig
from src.pytorch_version.geometry import build_geometry as build_torch_geometry
from src.pytorch_version.physics.coupling import (
    approx_retarded as torch_approx_retarded,
)
from src.pytorch_version.solvers import fast as torch_fast
from src.pytorch_version.solvers import straight as torch_straight


_CSV_FIELDS = (
    "model",
    "size",
    "rings",
    "frequencies",
    "solver",
    "backend",
    "device",
    "status",
    "seconds",
    "numpy_over_backend_speedup",
    "relative_error_vs_numpy",
    "geometry_seconds",
)


def _make_config(
    config_type: type,
    model: str,
    size: int,
    omegas: np.ndarray,
    solver: str,
):
    return config_type(
        name=f"benchmark_{model}_{size}_{solver}",
        R=0.2,
        L=1e-9,
        C=1e-12,
        radius=0.5e-3,
        width=0.1e-3,
        Dx=3e-3,
        Dy=3e-3,
        Dz=3e-3,
        orientations="zyx",
        type="center",
        Nx=size,
        Ny=size,
        Nz=size,
        packing="rectangle",
        interaction_model=model,
        H_0=1.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        omegas=omegas,
        solver=solver,
        save_currents=False,
        save_polarizability=False,
        save_extinction_cross_section=False,
    )


def _clear_numpy_cache() -> None:
    cache = getattr(
        numpy_coupling_operator, "_cached_mutual_inductance", None
    )
    if cache is not None:
        cache.cache_clear()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _solve(backend: str, solver: str, config: Any, geometry: Any,
           device: torch.device):
    if backend == "NumPy CPU":
        if solver == "fast":
            return numpy_fast.solve_system(
                config, geometry, use_preconditioner=True
            )
        return numpy_straight.solve_system(config, geometry)

    if solver == "fast":
        return torch_fast.solve_system(
            config,
            geometry,
            device=device,
            use_preconditioner=True,
        )
    return torch_straight.solve_system(config, geometry, device=device)


def _to_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(reference)), np.finfo(np.float64).tiny
    )
    return float(np.linalg.norm(actual - reference) / denominator)


def _append_row(writer: csv.DictWriter, file, row: dict[str, Any]) -> None:
    writer.writerow(row)
    file.flush()


def _result_row(
    *,
    model: str,
    size: int,
    rings: int,
    frequencies: int,
    solver: str,
    backend: str,
    device: str,
    status: str,
    seconds: float | str,
    speedup: float | str,
    error: float | str,
    geometry_seconds: float | str,
) -> dict[str, Any]:
    return {
        "model": model,
        "size": size,
        "rings": rings,
        "frequencies": frequencies,
        "solver": solver,
        "backend": backend,
        "device": device,
        "status": status,
        "seconds": seconds,
        "numpy_over_backend_speedup": speedup,
        "relative_error_vs_numpy": error,
        "geometry_seconds": geometry_seconds,
    }


def run_benchmark(model: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Compare NumPy and PyTorch solvers for {model}."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="cubic lattice sizes (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--frequencies",
        type=int,
        default=10,
        help="number of angular-frequency samples (default: 10)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="timed runs per backend and solver (default: 1)",
    )
    parser.add_argument(
        "--solvers",
        choices=("fast", "straight", "both"),
        default="both",
        help="solvers to run (default: both)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name(f"benchmark_{model}_results.csv"),
        help="output CSV (defaults to this script's folder)",
    )
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("all sizes must be positive")
    if args.frequencies < 1:
        parser.error("--frequencies must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    sizes = tuple(args.sizes)
    solver_names = (
        ("fast", "straight")
        if args.solvers == "both"
        else (args.solvers,)
    )
    cuda_available = torch.cuda.is_available()
    torch_devices = [torch.device("cpu")]
    if cuda_available:
        torch_devices.append(torch.device("cuda"))

    frequencies_hz = np.linspace(1e8, 1.9e8, args.frequencies)
    omegas = 2.0 * np.pi * frequencies_hz
    results: dict[tuple[int, str, str], np.ndarray] = {}
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.ones(1, device="cuda")
        torch.cuda.synchronize()
    print(
        f"Sizes: {', '.join(map(str, sizes))}; "
        f"frequencies: {args.frequencies}; "
        f"solvers: {', '.join(solver_names)}; repeats: {args.repeats}"
    )
    print(f"CSV: {args.csv.resolve()}")

    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        file.flush()

        for size in sizes:
            numpy_base = _make_config(
                NumpyConfig, model, size, omegas, "fast"
            )
            torch_base = _make_config(
                TorchConfig, model, size, omegas, "fast"
            )

            start = time.perf_counter()
            numpy_geometry = build_numpy_geometry(numpy_base)
            numpy_geometry_seconds = time.perf_counter() - start

            torch_geometries = {}
            torch_geometry_seconds = {}
            for device in torch_devices:
                start = time.perf_counter()
                geometry = build_torch_geometry(
                    torch_base, device=device
                )
                _synchronize(device)
                torch_geometries[str(device)] = geometry
                torch_geometry_seconds[str(device)] = (
                    time.perf_counter() - start
                )
                np.testing.assert_array_equal(
                    _to_numpy(geometry.centers), numpy_geometry.centers
                )
                if tuple(geometry.orientations) != tuple(
                    numpy_geometry.orientations
                ):
                    raise AssertionError(
                        f"Geometry orientations differ on {device}"
                    )

            rings = len(numpy_geometry.orientations)
            print(f"\n{size}x{size}x{size}: {rings} rings")

            for solver in solver_names:
                numpy_config = _make_config(
                    NumpyConfig, model, size, omegas, solver
                )
                numpy_timings = []
                numpy_solution = None
                for _ in range(args.repeats):
                    _clear_numpy_cache()
                    start = time.perf_counter()
                    numpy_solution = (
                        numpy_straight.solve_system(
                            numpy_config, numpy_geometry
                        )
                        if solver == "straight"
                        else numpy_fast.solve_system(
                            numpy_config,
                            numpy_geometry,
                            use_preconditioner=True,
                        )
                    )
                    numpy_timings.append(time.perf_counter() - start)
                assert numpy_solution is not None
                numpy_seconds = float(np.median(numpy_timings))
                numpy_solution = _to_numpy(numpy_solution)
                results[size, "NumPy CPU", solver] = numpy_solution
                _append_row(
                    writer,
                    file,
                    _result_row(
                        model=model,
                        size=size,
                        rings=rings,
                        frequencies=args.frequencies,
                        solver=solver,
                        backend="NumPy CPU",
                        device="cpu",
                        status="ok",
                        seconds=numpy_seconds,
                        speedup=1.0,
                        error=0.0,
                        geometry_seconds=numpy_geometry_seconds,
                    ),
                )
                print(
                    f"  {solver:8s} NumPy CPU   "
                    f"{numpy_seconds:9.3f}s"
                )

                for device in torch_devices:
                    backend = (
                        "PyTorch CUDA"
                        if device.type == "cuda"
                        else "PyTorch CPU"
                    )
                    config = _make_config(
                        TorchConfig, model, size, omegas, solver
                    )
                    geometry = torch_geometries[str(device)]
                    timings = []
                    solution = None
                    for _ in range(args.repeats):
                        if model == "approx_retarded":
                            torch_approx_retarded.clear_static_cache()
                        _synchronize(device)
                        start = time.perf_counter()
                        solution = _solve(
                            backend, solver, config, geometry, device
                        )
                        _synchronize(device)
                        timings.append(time.perf_counter() - start)
                    assert solution is not None
                    seconds = float(np.median(timings))
                    solution = _to_numpy(solution)
                    error = _relative_error(solution, numpy_solution)
                    speedup = numpy_seconds / seconds
                    results[size, backend, solver] = solution
                    _append_row(
                        writer,
                        file,
                        _result_row(
                            model=model,
                            size=size,
                            rings=rings,
                            frequencies=args.frequencies,
                            solver=solver,
                            backend=backend,
                            device=str(device),
                            status="ok",
                            seconds=seconds,
                            speedup=speedup,
                            error=error,
                            geometry_seconds=torch_geometry_seconds[
                                str(device)
                            ],
                        ),
                    )
                    print(
                        f"  {solver:8s} {backend:12s} "
                        f"{seconds:9.3f}s | "
                        f"NumPy/PyTorch {speedup:6.2f}x | "
                        f"relative error {error:.3e}"
                    )

                if not cuda_available:
                    _append_row(
                        writer,
                        file,
                        _result_row(
                            model=model,
                            size=size,
                            rings=rings,
                            frequencies=args.frequencies,
                            solver=solver,
                            backend="PyTorch CUDA",
                            device="cuda",
                            status="unavailable",
                            seconds="",
                            speedup="",
                            error="",
                            geometry_seconds="",
                        ),
                    )

            if "fast" in solver_names and "straight" in solver_names:
                for backend in (
                    "NumPy CPU",
                    "PyTorch CPU",
                    *(("PyTorch CUDA",) if cuda_available else ()),
                ):
                    error = _relative_error(
                        results[size, backend, "fast"],
                        results[size, backend, "straight"],
                    )
                    print(
                        f"  {backend} fast-vs-straight error: "
                        f"{error:.3e}"
                    )

    print(f"\nBenchmark complete. CSV saved to {args.csv.resolve()}")
