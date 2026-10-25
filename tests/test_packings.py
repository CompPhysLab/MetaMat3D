
import numpy as np
import pytest
from src.metamaterial.geometry.packings import cylinder_z

COUNTS = (4, 2, 3)

def test_cylinder_z_center_is_inside():
    points = np.array([
        [4, 2, 0],
    ], dtype=np.int16)

    mask = cylinder_z(points, COUNTS)

    np.testing.assert_array_equal(mask, [True])

def test_cylinder_z_ellipse_cross_section():
    # Центр сечения (4, 2), полуоси (4, 2).
    points = np.array([
        [8, 2, 1],   # граница по x
        [4, 4, 1],   # граница по y
        [6, 3, 1],   # внутри эллипса
        [8, 4, 1],   # вне эллипса
    ], dtype=np.int16)

    mask = cylinder_z(points, COUNTS)

    np.testing.assert_array_equal(
        mask,
        [True, True, True, False],
    )

def test_cylinder_z_does_not_filter_by_height():
    # Текущая реализация проверяет только координаты x и y.
    points = np.array([
        [4, 2, 0],
        [4, 2, 3],
        [4, 2, -1],
        [4, 2, 10],
    ], dtype=np.int16)

    mask = cylinder_z(points, COUNTS)

    np.testing.assert_array_equal(mask, [True, True, True, True])

def test_cylinder_z_preserves_point_order():
    points = np.array([
        [8, 2, 0],   # граница сечения
        [4, 2, 1],   # центр сечения
        [8, 4, 1],   # вне сечения
        [4, 2, 10],  # центр сечения, z за пределами высоты
    ], dtype=np.int16)

    mask = cylinder_z(points, COUNTS)

    np.testing.assert_array_equal(
        mask,
        [True, True, False, True],
    )
    np.testing.assert_array_equal(
        points[mask],
        points[[0, 1, 3]],
    )

def test_cylinder_z_empty_points():
    points = np.empty((0, 3), dtype=np.int16)

    mask = cylinder_z(points, COUNTS)

    assert mask.shape == (0,)
    assert mask.dtype == bool

def test_cylinder_z_accepts_integer_compatible_input():
    # Функция приводит входные координаты к целочисленному типу.
    points = np.array([
        [4.9, 2.9, 1.5],
        [8.9, 4.9, 1.5],
    ])

    mask = cylinder_z(points, COUNTS)

    np.testing.assert_array_equal(mask, [True, False])

@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([4, 2, 1], True),    # центр
        ([8, 2, 1], True),    # граница по x
        ([4, 4, 1], True),    # граница по y
        ([8, 4, 1], False),   # вне эллипса
        ([0, 0, 1], False),   # вне эллипса
    ],
)
def test_cylinder_z_individual_points(point, expected):
    points = np.array([point], dtype=np.int16)

    mask = cylinder_z(points, COUNTS)

    assert mask[0] is expected or mask[0] == expected