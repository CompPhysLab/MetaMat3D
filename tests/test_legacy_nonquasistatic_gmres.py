"""Check legacy non-quasistatic FFT products and GMRES system scaling."""

import sys
from pathlib import Path

import numpy as np
from scipy.fft import fftn
from scipy.linalg import solve
from scipy.sparse.linalg import LinearOperator, gmres


def _legacy_modules():
    code_directory = str(Path(__file__).resolve().parents[1] / "Code")
    sys.path.insert(0, code_directory)
    try:
        from Fast_Method import Circvec, fft_dot
        from Geometry import Rectangle_packing, to3D
        from Retr_matrix import Matrix
    finally:
        sys.path.remove(code_directory)
    return Circvec, fft_dot, Rectangle_packing, to3D, Matrix


def test_legacy_nonquasistatic_fft_and_gmres_forms_match_dense_solution():
    circvec, fft_dot, rectangle_packing, to3d, dense_matrix = (
        _legacy_modules()
    )

    omega = 2 * np.pi * 1e8
    spacing = 3e-3
    radius = 0.5e-3
    orientations = "zyx"
    counts, _ = to3d(2, 2, 2, orientations, "border")
    params = {
        "Orientations": orientations,
        "N": counts,
        "L": 1e-9,
        "C": 1e-12,
        "R": 0.2,
        "W": 0.0,
        "Radius": radius,
        "Dx": spacing,
        "Dy": spacing,
        "Dz": spacing,
        "Shift_x": 0.0,
        "Shift_y": 0.0,
        "Shift_z": 0.0,
        "Scattering": "Mie_True",
    }
    rings_by_orientation = rectangle_packing(params)
    ring_grids = {
        axis: np.asarray(rings_by_orientation[axis]).reshape(
            counts[axis]["nz"],
            counts[axis]["ny"],
            counts[axis]["nx"],
        )
        for axis in orientations
    }
    rings = [
        ring
        for axis in orientations
        for ring in rings_by_orientation[axis]
    ]

    # One frequency, a non-quasistatic model, and a 2x2x2 mixed-orientation
    # geometry (36 rings) keep the dense reference calculation manageable.
    dense = dense_matrix(omega, rings, Data={})
    fft_kernels = {}
    for target in orientations:
        fft_kernels[target] = {}
        for source in orientations:
            kernel = circvec(
                params,
                ring_grids[target],
                ring_grids[source],
                omega,
                {},
            )
            fft_kernels[target][source] = fftn(kernel, axes=(0, 1, 2))

    def fft_matvec(vector):
        result = np.zeros(len(rings), dtype=np.complex128)
        source_offsets = {}
        target_offsets = {}
        offset = 0
        for axis in orientations:
            source_offsets[axis] = offset
            target_offsets[axis] = offset
            offset += ring_grids[axis].size

        for target in orientations:
            target_start = target_offsets[target]
            target_end = target_start + ring_grids[target].size
            for source in orientations:
                source_start = source_offsets[source]
                source_end = source_start + ring_grids[source].size
                fft_kernel = fft_kernels[target][source]
                fft_shape = fft_kernel.shape
                result[target_start:target_end] += fft_dot(
                    vector[source_start:source_end].reshape(
                        ring_grids[source].shape
                    ),
                    np.zeros(fft_shape, dtype=np.complex128),
                    fft_kernel,
                    np.zeros(fft_shape, dtype=np.complex128),
                    np.zeros(fft_shape, dtype=np.complex128),
                )
        return result

    size = len(rings)
    for column in range(size):
        basis = np.zeros(size, dtype=np.complex128)
        basis[column] = 1.0
        actual = fft_matvec(basis)
        expected = dense @ basis
        worst_row = int(np.argmax(np.abs(actual - expected)))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-9,
            atol=1e-14,
            err_msg=(
                f"FFT mismatch at row {worst_row}, column {column}: "
                f"FFT={actual[worst_row]!r}, "
                f"dense={expected[worst_row]!r}, "
                f"error={abs(actual[worst_row] - expected[worst_row]):.6e}"
            ),
        )

    diagonal = rings[0].M(omega)
    rhs = np.arange(1, size + 1, dtype=np.float64) * (1.0 + 0.25j)
    direct = solve(
        np.diag(np.full(size, diagonal, dtype=np.complex128)) - dense,
        rhs,
    )

    current_operator = LinearOperator(
        (size, size),
        matvec=lambda vector: diagonal * vector - fft_matvec(vector),
        dtype=np.complex128,
    )
    voltage_operator = LinearOperator(
        (size, size),
        matvec=lambda vector: (
            diagonal * vector - fft_matvec(vector)
        ) / diagonal,
        dtype=np.complex128,
    )

    current_solution, current_info = gmres(
        current_operator, rhs, rtol=1e-10, atol=0.0
    )
    voltage_solution, voltage_info = gmres(
        voltage_operator, rhs / diagonal, rtol=1e-10, atol=0.0
    )

    assert current_info == 0
    assert voltage_info == 0
    np.testing.assert_allclose(current_solution, direct, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(voltage_solution, direct, rtol=1e-9, atol=1e-12)

    current_residual = np.linalg.norm(
        current_operator @ current_solution - rhs
    ) / np.linalg.norm(rhs)
    voltage_residual = np.linalg.norm(
        voltage_operator @ voltage_solution - rhs / diagonal
    ) / np.linalg.norm(rhs / diagonal)
    assert current_residual <= 1e-10
    assert voltage_residual <= 1e-10
