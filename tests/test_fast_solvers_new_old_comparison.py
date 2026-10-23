"""Compare legacy and current fast solvers on a 4x4x4 retarded geometry."""

import sys
from pathlib import Path

import numpy as np

from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import build_geometry
from src.metamaterial.solvers import fast


def _legacy_fast_solver(monkeypatch):
    code_directory = str(Path(__file__).resolve().parents[1] / "Code")
    sys.path.insert(0, code_directory)
    try:
        import Fast_Method
        from Geometry import Rectangle_packing, to3D
        import Retr_matrix_patched
    finally:
        sys.path.remove(code_directory)

    monkeypatch.setitem(
        Fast_Method.Mnms, "Mie_True", Retr_matrix_patched.Mnm
    )
    return Fast_Method.solvesystem, Rectangle_packing, to3D


def test_legacy_and_current_fast_solvers_match_on_4x4x4(monkeypatch):
    legacy_fast, legacy_packing, legacy_to3d = _legacy_fast_solver(monkeypatch)

    omega = 2 * np.pi * 1e8
    orientations = "zyx"
    radius = 0.5e-3
    width = 0.0
    spacing = 3e-3
    resistance = 0.2
    inductance = 1e-9
    capacitance = 1e-12
    config = SimulationConfig(
        name="test_fast_retarded_4x4x4",
        orientations=orientations,
        packing="rectangle",
        Nx=4,
        Ny=4,
        Nz=4,
        Dx=spacing,
        Dy=spacing,
        Dz=spacing,
        radius=radius,
        width=width,
        R=resistance,
        L=inductance,
        C=capacitance,
        omegas=np.array([omega]),
        H_0=1.0,
        H_direction=(1.0, 0.0, 0.0),
        k_direction=(0.0, 0.0, 1.0),
        interaction_model="full_retarded",
        type="border",
        solver="fast",
    )
    geometry = build_geometry(config)
    assert len(geometry.centers) == 240
    current_result = fast.solve_system(config, geometry)[:, 0]

    counts, _ = legacy_to3d(4, 4, 4, orientations, "border")
    legacy_params = {
        "Orientations": orientations,
        "N": counts,
        "L": inductance,
        "C": capacitance,
        "R": resistance,
        "W": width,
        "Radius": radius,
        "Dx": spacing,
        "Dy": spacing,
        "Dz": spacing,
        "Shift_x": 0.0,
        "Shift_y": 0.0,
        "Shift_z": 0.0,
        "Omega": [omega, omega, 1],
        "Scattering": "Mie_True",
        "Solver_name": "gmres",
        "Solver_type": "Fast",
        "Numbers": {"z": 80, "y": 80, "x": 80},
        "P_0z": 1.0,
        "Slices": {},
        "IsSlices": False,
        "MemLim": 1024**3,
    }
    legacy_rings = legacy_packing(legacy_params)

    def excitation(rings_by_orientation):
        return {
            axis: [
                (
                    4 * np.pi * 1e-7
                    * np.pi
                    * radius**2
                    * (axis == "x")
                    * np.exp(1j * omega / 3e8 * ring.z)
                )
                for ring in np.asarray(rings, dtype=object).ravel()
            ]
            for axis, rings in rings_by_orientation.items()
        }

    legacy_result = np.asarray(
        legacy_fast(
            np.array([omega]),
            legacy_params,
            legacy_rings,
            lambda _omega: excitation(legacy_rings),
            Inductance={},
            find="Currents",
            tol=1e-10,
        )["Currents"][0]
    )

    assert current_result.shape == legacy_result.shape == (240,)
    assert np.all(np.isfinite(current_result))
    assert np.all(np.isfinite(legacy_result))

    relative_l2_error = np.linalg.norm(current_result - legacy_result) / max(
        np.linalg.norm(current_result), np.finfo(float).tiny
    )
    max_absolute_error = np.max(np.abs(current_result - legacy_result))
    print(
        "legacy fast vs current fast on 4x4x4: "
        f"max_abs={max_absolute_error:.6e}, "
        f"relative_l2={relative_l2_error:.6e}"
    )
    np.testing.assert_allclose(
        legacy_result,
        current_result,
        rtol=1e-7,
        atol=1e-12,
    )
