from types import SimpleNamespace
import numpy as np
import pytest
from src.metamaterial.geometry import (
    build_geometry,
    orientation_normals,
)

def config(**overrides):
    values = dict(
        Dx=1.0,
        Dy=1.0,
        Dz=1.0,
        Nx=2,
        Ny=2,
        Nz=2,
        orientations="zyx",
        type="center",
        packing="rectangle",
    )
    values.update(overrides)
    return SimpleNamespace(**values)

def test_center_rectangle_count_dtype_and_order():
    result = build_geometry(config(orientations="z", type="center"))

    assert result.centers.shape == (8, 3)
    assert result.centers.dtype == np.int16
    assert result.orientations.shape == (8,)
    assert result.orientations.dtype.kind == "U"
    assert np.all(result.orientations == "z")

    # Center coordinates are odd integer lattice coordinates.
    np.testing.assert_array_equal(
        result.centers,
        np.array(
            [
                [1, 1, 1],
                [3, 1, 1],
                [1, 3, 1],
                [3, 3, 1],
                [1, 1, 3],
                [3, 1, 3],
                [1, 3, 3],
                [3, 3, 3],
            ],
            dtype=np.int16,
        ),
    )

@pytest.mark.parametrize(
    ("orientation", "expected_count", "normal_axis"),
    [
        ("z", 3 * 2 * 2, 2),
        ("y", 2 * 3 * 2, 1),
        ("x", 2 * 2 * 3, 0),
    ],
)
def test_border_has_extra_plane_on_normal_axis(
    orientation, expected_count, normal_axis
):
    result = build_geometry(config(orientations=orientation, type="border"))

    assert result.centers.shape == (expected_count, 3)
    assert np.all(result.orientations == orientation)

    normal_coordinates = result.centers[:, normal_axis]

    np.testing.assert_array_equal(
        np.unique(normal_coordinates),
        np.array([0, 2, 4], dtype=np.int16),
    )

@pytest.mark.parametrize(
    ("orientation", "normal_axis"),
    [
        ("z", 2),
        ("y", 1),
        ("x", 0),
    ],
)
def test_open_excludes_normal_boundary_planes(orientation, normal_axis):
    result = build_geometry(config(orientations=orientation, type="open"))

    assert len(result.centers) == 4
    assert np.all(result.orientations == orientation)

    # For N=2, the only interior normal coordinate is 2.
    np.testing.assert_array_equal(
        result.centers[:, normal_axis],
        np.full(4, 2, dtype=np.int16),
    )

    # Non-normal axes contain cell-center coordinates.
    for axis in range(3):
        if axis != normal_axis:
            assert np.all(result.centers[:, axis] % 2 == 1)

def test_orientation_blocks_follow_requested_order():
    result = build_geometry(config(orientations="yx", type="center"))

    assert result.centers.shape == (16, 3)
    np.testing.assert_array_equal(result.orientations[:8], "y")
    np.testing.assert_array_equal(result.orientations[8:], "x")

def test_center_coordinates_follow_k_j_i_order():
    result = build_geometry(config(orientations="z", type="center"))

    # With Fortran-order ravel, i changes fastest, then j, then k.
    np.testing.assert_array_equal(
        result.centers[:4],
        np.array(
            [
                [1, 1, 1],
                [3, 1, 1],
                [1, 3, 1],
                [3, 3, 1],
            ],
            dtype=np.int16,
        ),
    )

def test_ellipse_filters_points_without_reordering():
    rectangular = build_geometry(
        config(orientations="z", type="center", packing="rectangle")
    )
    ellipse = build_geometry(
        config(orientations="z", type="center", packing="ellipse")
    )

    assert len(ellipse.centers) <= len(rectangular.centers)

    rectangular_points = {
        tuple(point): index
        for index, point in enumerate(rectangular.centers)
    }
    indices = [rectangular_points[tuple(point)] for point in ellipse.centers]

    assert indices == sorted(indices)

def test_cylinder_filters_points_without_reordering():
    rectangular = build_geometry(
        config(orientations="z", type="center", packing="rectangle")
    )
    cylinder = build_geometry(
        config(orientations="z", type="center", packing="cylinder_z")
    )

    assert len(cylinder.centers) <= len(rectangular.centers)

    rectangular_points = {
        tuple(point): index
        for index, point in enumerate(rectangular.centers)
    }
    indices = [rectangular_points[tuple(point)] for point in cylinder.centers]

    assert indices == sorted(indices)

def test_orientation_normals_maps_axes_to_unit_vectors():
    orientations = np.array(["x", "y", "z", "z", "x"], dtype="<U1")

    normals = orientation_normals(orientations)

    np.testing.assert_array_equal(
        normals,
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=np.int16,
        ),
    )

def test_orientation_normals_preserves_order_and_shape():
    orientations = np.array(["z", "x", "y"], dtype="<U1")

    normals = orientation_normals(orientations)

    assert normals.shape == (3, 3)
    assert normals.dtype == np.int16
    np.testing.assert_array_equal(
        normals,
        np.eye(3, dtype=np.int16)[[2, 0, 1]],
    )