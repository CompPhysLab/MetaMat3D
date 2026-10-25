"""Check the legacy FFT matvec against its dense mutual-inductance matrix."""

import sys
from pathlib import Path

import numpy as np
from scipy.fft import fftn


def _legacy_fast_modules():
    code_directory = str(Path(__file__).resolve().parents[1] / "Code")
    sys.path.insert(0, code_directory)
    try:
        from Fast_Method import Circvec, fft_dot
        from Geometry import Rectangle_packing, to3D
        from Impedance_matrix import Matrix
    finally:
        sys.path.remove(code_directory)
    return Circvec, fft_dot, Rectangle_packing, to3D, Matrix


def test_legacy_fft_matvec_matches_dense_matrix():
    circvec, fft_dot, rectangle_packing, to3d, dense_matrix = (
        _legacy_fast_modules()
    )

    omega = 2 * np.pi * 1e8
    orientations = "zyx"
    nx = ny = nz = 2
    spacing = 3e-3
    counts, _ = to3d(nz, ny, nx, orientations, "center")
    params = {
        "Orientations": orientations,
        "N": counts,
        "L": 1e-9,
        "C": 1e-12,
        "R": 0.2,
        "W": 0.0,
        "Radius": 0.5e-3,
        "Dx": spacing,
        "Dy": spacing,
        "Dz": spacing,
        "Shift_x": 0.0,
        "Shift_y": 0.0,
        "Shift_z": 0.0,
        "Scattering": "Mie_False",
    }
    rings_by_orientation = rectangle_packing(params)
    orientation_rings = {
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
    dense = dense_matrix(omega, rings, Data={})

    fft_kernels = {}
    for target in orientations:
        fft_kernels[target] = {}
        for source in orientations:
            kernel = circvec(
                params,
                orientation_rings[target],
                orientation_rings[source],
                omega,
                {},
            )
            fft_kernels[target][source] = fftn(kernel, axes=(0, 1, 2))

    group_sizes = {
        axis: orientation_rings[axis].size
        for axis in orientations
    }
    group_offsets = {}
    offset = 0
    for axis in orientations:
        group_offsets[axis] = offset
        offset += group_sizes[axis]

    def legacy_fft_matvec(vector):
        result = np.zeros_like(vector, dtype=np.complex128)
        for target in orientations:
            target_start = group_offsets[target]
            target_end = target_start + group_sizes[target]
            for source in orientations:
                source_start = group_offsets[source]
                source_end = source_start + group_sizes[source]
                source_shape = orientation_rings[source].shape
                fft_shape = fft_kernels[target][source].shape
                product = fft_dot(
                    vector[source_start:source_end].reshape(source_shape),
                    np.zeros(fft_shape, dtype=np.complex128),
                    fft_kernels[target][source],
                    np.zeros(fft_shape, dtype=np.complex128),
                    np.zeros(fft_shape, dtype=np.complex128),
                )
                result[target_start:target_end] += product
        return result

    for column in range(len(rings)):
        basis = np.zeros(len(rings), dtype=np.complex128)
        basis[column] = 1.0
        np.testing.assert_allclose(
            legacy_fft_matvec(basis),
            dense @ basis,
            rtol=1e-9,
            atol=1e-14,
            err_msg=f"Legacy FFT product differs for matrix column {column}",
        )
