"""Compare legacy and current solvers on the same retarded geometry."""

import sys
from pathlib import Path

import numpy as np

from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import build_geometry
from src.metamaterial.solvers import fast, straight


def _legacy_solvers(monkeypatch):
    code_directory = str(Path(__file__).resolve().parents[1] / "Code")
    sys.path.insert(0, code_directory)
    try:
        import Fast_Method
        from Geometry import Rectangle_packing, to3D
        import Retr_matrix_patched
        import Straight_Method
    finally:
        sys.path.remove(code_directory)
    monkeypatch.setitem(
        Fast_Method.Mnms, "Mie_True", Retr_matrix_patched.Mnm
    )
    monkeypatch.setitem(
        Straight_Method.Matrixes, "Mie_True", Retr_matrix_patched.Matrix
    )
    return (
        Fast_Method.solvesystem,
        Straight_Method.solvesystem,
        Rectangle_packing,
        to3D,
    )


def test_legacy_and_current_solvers_on_full_retarded_2x2x2(monkeypatch):
    legacy_fast, legacy_straight, rectangle_packing, to3d = _legacy_solvers(
        monkeypatch
    )

    omega = 2 * np.pi * 1e8
    orientations = "zyx"
    radius = 0.5e-3
    width = 0.0
    spacing = 3e-3
    resistance = 0.2
    inductance = 1e-9
    capacitance = 1e-12
    config = SimulationConfig(
        name=        "test_fast_retarded_3x3x3",
        orientations=orientations,
        packing="rectangle",
        Nx=3,
        Ny=3,
        Nz=3,
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
    assert len(geometry.centers) == 108

    new_straight_result = straight.solve_system(config, geometry)[:, 0]
    new_fast_result = fast.solve_system(config, geometry)[:, 0]

    counts, _ = to3d(3, 3, 3, orientations, "border")

    def make_legacy_rings():
        params = {
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
            "Numbers": {},
            "P_0z": 1.0,
            "Slices": {},
            "IsSlices": False,
            "MemLim": 1024**2,
        }
        rings = rectangle_packing(params)
        return params, rings

    def legacy_excitation(rings_by_orientation):
        result = {}
        for axis, rings in rings_by_orientation.items():
            result[axis] = [
                (
                    4 * np.pi * 1e-7
                    * np.pi
                    * radius**2
                    * (axis == "x")
                    * np.exp(1j * omega / 3e8 * ring.z)
                )
                for ring in np.asarray(rings, dtype=object).ravel()
            ]
        return result

    params_straight, rings_straight = make_legacy_rings()
    old_straight_result = np.asarray(
        legacy_straight(
            np.array([omega]),
            params_straight,
            rings_straight,
            lambda _omega: legacy_excitation(rings_straight),
            Inductance={},
            find="Currents",
        )["Currents"][0]
    )

    params_fast, rings_fast = make_legacy_rings()
    old_fast_output = legacy_fast(
        np.array([omega]),
        params_fast,
        rings_fast,
        lambda _omega: legacy_excitation(rings_fast),
        Inductance={},
        find="Currents",
        tol=1e-10,
    )
    old_fast_result = np.asarray(old_fast_output["Currents"][0])

    results = {
        "old straight": old_straight_result,
        "new straight": new_straight_result,
        "old fast": old_fast_result,
        "new fast": new_fast_result,
    }
    for name, result in results.items():
        assert result.shape == (108,), f"{name} returned shape {result.shape}"
        assert np.all(np.isfinite(result)), f"{name} returned non-finite values"

    np.testing.assert_allclose(
        new_fast_result,
        new_straight_result,
        rtol=1e-8,
        atol=1e-12,
        err_msg="New fast solver differs from new straight reference",
    )
    comparisons = (
        ("old fast", old_fast_result, "old straight", old_straight_result),
        ("new fast", new_fast_result, "new straight", new_straight_result),
        ("old fast", old_fast_result, "new straight", new_straight_result),
        ("new fast", new_fast_result, "old straight", old_straight_result),
        ("old straight", old_straight_result, "new straight", new_straight_result),
    )
    for fast_name, fast_result, reference_name, reference in comparisons:
        relative_error = np.linalg.norm(fast_result - reference) / max(
            np.linalg.norm(reference), np.finfo(float).tiny
        )
        max_absolute_error = np.max(np.abs(fast_result - reference))
        print(
            f"{fast_name} vs {reference_name}: "
            f"max_abs={max_absolute_error:.6e}, "
            f"relative_l2={relative_error:.6e}"
        )

    for fast_name, fast_result, reference_name, reference in comparisons:
        np.testing.assert_allclose(
            fast_result,
            reference,
            rtol=1e-8,
            atol=1e-12,
            err_msg=f"{fast_name} differs from {reference_name}",
        )
