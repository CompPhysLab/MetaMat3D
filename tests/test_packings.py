import numpy as np
import pytest

from src.metamaterial.geometry.packings import cylinder_z


def test_cylinder_z_center_is_inside():
    points = np.array([
        [2.0, 1.0, 1.5],
    ])

    mask = cylinder_z(points, (4.0, 2.0, 3.0))

    np.testing.assert_array_equal(mask, [True])


def test_cylinder_z_ellipse_cross_section():
    # Bounds: Lx=4, Ly=2, Lz=3.
    # Ellipse center: (2, 1), semiaxes: a=2, b=1.
    points = np.array([
        [4.0, 1.0, 1.5],  # boundary on x
        [2.0, 2.0, 1.5],  # boundary on y
        [3.0, 1.5, 1.5],  # inside ellipse
        [3.8, 1.8, 1.5],  # outside ellipse
    ])

    mask = cylinder_z(points, (4.0, 2.0, 3.0))

    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_cylinder_z_height_boundaries():
    points = np.array([
        [2.0, 1.0, 0.0],
        [2.0, 1.0, 3.0],
        [2.0, 1.0, -0.1],
        [2.0, 1.0, 3.1],
    ])

    mask = cylinder_z(points, (4.0, 2.0, 3.0))

    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_cylinder_z_preserves_point_order():
    points = np.array([
        [0.0, 0.0, 0.0],  # ellipse boundary
        [2.0, 1.0, 1.0],  # center
        [4.0, 2.0, 1.0],  # outside ellipse
        [2.0, 1.0, 3.0],  # upper height boundary
    ])

    mask = cylinder_z(points, (4.0, 2.0, 3.0))

    np.testing.assert_array_equal(mask, [False, True, False, True])
    np.testing.assert_array_equal(points[mask], points[[1, 3]])


def test_cylinder_z_empty_points():
    points = np.empty((0, 3))

    mask = cylinder_z(points, (4.0, 2.0, 3.0))

    assert mask.shape == (0,)
    assert mask.dtype == bool


@pytest.mark.parametrize("points", [
    np.array([1.0, 2.0, 3.0]),
    np.empty((2, 2)),
    np.empty((2, 4)),
])
def test_cylinder_z_invalid_points_shape(points):
    with pytest.raises(ValueError, match=r"points must have shape"):
        cylinder_z(points, (4.0, 2.0, 3.0))


@pytest.mark.parametrize("bounds", [
    (4.0, 2.0),
    (4.0, 2.0, 3.0, 1.0),
    (4.0, 0.0, 3.0),
    (4.0, -2.0, 3.0),
])
def test_cylinder_z_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bounds must contain three positive lengths"):
        cylinder_z(np.array([[1.0, 1.0, 1.0]]), bounds)