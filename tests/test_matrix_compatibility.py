"""Compare the legacy Code implementation with src.metamaterial matrices."""

import sys
from pathlib import Path

import numpy as np
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import build_geometry
from src.metamaterial.physics import mutual_inductance_matrix, self_inductance


def _legacy_modules():
    code_directory = str(Path(__file__).resolve().parents[1] / "Code")
    sys.path.insert(0, code_directory)
    try:
        from Geometry import Rectangle_packing, to3D
        from Impedance_matrix import Matrix
    finally:
        sys.path.remove(code_directory)
    return Rectangle_packing, to3D, Matrix


def test_final_system_matrices_match_legacy_code():
    rectangle_packing, to3d, legacy_matrix = _legacy_modules()

    radius = 0.1
    width = 0.02
    spacing = 1.0
    resistance = 0.3
    inductance = 1.2e-9
    capacitance = 4.7e-12
    omega = 2 * np.pi * 1e9
    orientations = "zyx"

    config = SimulationConfig(
        name="matrix_compatibility",
        R=resistance,
        L=inductance,
        C=capacitance,
        radius=radius,
        width=width,
        Dx=spacing,
        Dy=spacing,
        Dz=spacing,
        orientations=orientations,
        type="border",
        Nx=1,
        Ny=1,
        Nz=1,
        packing="rectangle",
        interaction_model="quasistatic",
        H_0=1.0,
        H_direction=(0.0, 0.0, 1.0),
        omegas=np.array([omega]),
        solver="straight",
    )

    geometry = build_geometry(config)
    legacy_counts, _ = to3d(1, 1, 1, orientations, "border")
    legacy_params = {
        "Orientations": orientations,
        "N": legacy_counts,
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
    }
    legacy_rings_by_orientation = rectangle_packing(legacy_params)
    legacy_rings = [
        ring
        for orientation in orientations
        for ring in legacy_rings_by_orientation[orientation]
    ]
    legacy_centers = np.array(
        [[ring.x, ring.y, ring.z] for ring in legacy_rings],
        dtype=np.float64,
    )
    legacy_orientations = np.array(
        [ring.pos for ring in legacy_rings],
        dtype="<U1",
    )

    np.testing.assert_allclose(geometry.centers, legacy_centers)
    np.testing.assert_array_equal(geometry.orientations, legacy_orientations)

    modern_mutual = mutual_inductance_matrix(
        geometry.centers,
        geometry.orientations,
        radius=radius,
        width=width,
        model="quasistatic",
    )
    legacy_mutual = legacy_matrix(omega, legacy_rings, Data={})
    np.testing.assert_allclose(
        modern_mutual,
        legacy_mutual,
        rtol=1e-8,
        atol=1e-14,
        err_msg="Mutual-inductance matrices differ",
    )

    modern_system = np.diag(
        np.full(len(geometry.centers), self_inductance(
            omega, resistance, inductance, capacitance
        ))
    ) - modern_mutual
    legacy_system = np.diag(
        np.full(len(legacy_rings), legacy_rings[0].M(omega))
    ) - legacy_mutual
    np.testing.assert_allclose(
        modern_system,
        legacy_system,
        rtol=1e-8,
        atol=1e-14,
        err_msg="Final system matrices differ",
    )
