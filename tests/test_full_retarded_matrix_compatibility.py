"""Compare legacy and current full-retarded mutual-inductance matrices."""

import sys
from pathlib import Path

import numpy as np

from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import build_geometry
from src.metamaterial.physics import mutual_inductance_matrix


def _legacy_matrix_builder():
    code_directory = str(Path(__file__).resolve().parents[1] / "Code")
    sys.path.insert(0, code_directory)
    try:
        from Geometry import Rectangle_packing, to3D
        from Retr_matrix_patched import Matrix
    finally:
        sys.path.remove(code_directory)
    return Rectangle_packing, to3D, Matrix


def test_full_retarded_mutual_inductance_matrices_match_on_4x4x4():
    legacy_packing, legacy_to3d, legacy_matrix = _legacy_matrix_builder()

    frequency = 1e8
    omega = 2 * np.pi * frequency
    radius = 0.5e-3
    spacing = 3e-3
    orientations = "zyx"

    config = SimulationConfig(
        name="full_retarded_matrix_compatibility",
        R=0.2,
        L=1e-9,
        C=1e-12,
        radius=radius,
        width=0.0,
        Dx=spacing,
        Dy=spacing,
        Dz=spacing,
        orientations=orientations,
        type="border",
        Nx=2,
        Ny=2,
        Nz=2,
        packing="rectangle",
        interaction_model="full_retarded",
        H_0=1.0,
        H_direction=(1.0, 0.0, 0.0),
        k_direction=(0.0, 0.0, 1.0),
        omegas=np.array([omega]),
        solver="straight",
    )
    geometry = build_geometry(config)

    legacy_counts, _ = legacy_to3d(2, 2, 2, orientations, "border")
    legacy_params = {
        "Orientations": orientations,
        "N": legacy_counts,
        "L": config.L,
        "C": config.C,
        "R": config.R,
        "W": config.width,
        "Radius": config.radius,
        "Dx": config.Dx,
        "Dy": config.Dy,
        "Dz": config.Dz,
        "Shift_x": 0.0,
        "Shift_y": 0.0,
        "Shift_z": 0.0,
        "Packing": "Rectangle",
    }
    rings_by_orientation = legacy_packing(legacy_params)
    legacy_rings = [
        ring
        for axis in orientations
        for ring in np.asarray(
            rings_by_orientation[axis], dtype=object
        ).ravel()
    ]
    legacy_centers = np.asarray(
        [[ring.x, ring.y, ring.z] for ring in legacy_rings],
        dtype=np.float64,
    )
    legacy_orientations = np.asarray(
        [ring.pos for ring in legacy_rings],
        dtype="<U1",
    )

    assert geometry.centers.shape == (36, 3)
    np.testing.assert_allclose(geometry.centers, legacy_centers)
    np.testing.assert_array_equal(geometry.orientations, legacy_orientations)

    new_matrix = mutual_inductance_matrix(
        geometry.centers,
        geometry.orientations,
        radius=config.radius,
        width=config.width,
        model="full_retarded",
        omega=omega,
    )
    old_matrix = legacy_matrix(omega, legacy_rings, Data={})

    difference = np.abs(new_matrix - old_matrix)
    relative_frobenius_error = np.linalg.norm(difference) / max(
        np.linalg.norm(old_matrix), np.finfo(float).tiny
    )
    print(
        "full_retarded matrix comparison: "
        f"max_abs={difference.max():.6e}, "
        f"relative_frobenius={relative_frobenius_error:.6e}"
    )
    np.testing.assert_allclose(
        new_matrix,
        old_matrix,
        rtol=1e-6,
        atol=1e-14,
        err_msg="Legacy and current full-retarded matrices differ",
    )
