from types import SimpleNamespace
import numpy as np
import pytest
from src.metamaterial.geometry import build_geometry

def config(**overrides):
    values = dict(
        Dx=1.0, Dy=1.0, Dz=1.0,
        Nx=2, Ny=2, Nz=2,
        orientations="zyx",
        type="center",
        packing="rectangle",
    )
    values.update(overrides)
    return SimpleNamespace(**values)

def test_center_rectangle_count_and_order():
    result = build_geometry(config(orientations="z", type="center"))
    assert result.centers.shape == (8, 3)
    assert np.all(result.orientations == "z")
    np.testing.assert_allclose(
        result.centers[:4],
        [[0.5, 0.5, 0.5],
         [1.5, 0.5, 0.5],
         [0.5, 1.5, 0.5],
         [1.5, 1.5, 0.5]],
    )

def test_border_has_extra_plane_only_on_normal_axis():
    z_result = build_geometry(config(orientations="z", type="border"))
    y_result = build_geometry(config(orientations="y", type="border"))
    x_result = build_geometry(config(orientations="x", type="border"))
    assert len(z_result.centers) == 3 * 2 * 2
    assert len(y_result.centers) == 2 * 3 * 2
    assert len(x_result.centers) == 2 * 2 * 3

def test_open_excludes_normal_boundary_planes():
    result = build_geometry(config(orientations="z", type="open"))
    assert len(result.centers) == 4  # one interior plane times Nx * Ny
    np.testing.assert_allclose(result.centers[:, 2], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.centers[:,0], [0.5, 1.5, 0.5, 1.5])

def test_orientation_blocks_follow_requested_order():
    result = build_geometry(config(orientations="yx", type="center"))
    assert result.orientations[0] == "y"
    assert result.orientations[-1] == "x"
    assert np.all(result.orientations[:8] == "y")
    assert np.all(result.orientations[8:] == "x")

def test_ellipsoid_filters_points_without_reordering():
    rectangular = build_geometry(config(orientations="z", type="center"))
    ellipsoidal = build_geometry(config(orientations="z", type="center",
                                        packing="ellipse"))
    assert len(ellipsoidal.centers) <= len(rectangular.centers)
    # Filtering may remove candidates but must preserve the original sequence
    expected = [p for p in rectangular.centers
                if np.sum(((p - 1.0) / 1.0) ** 2) <= 1.0 + 1e-12]
    np.testing.assert_allclose(ellipsoidal.centers, np.asarray(expected).reshape(-1, 3))

def test_cylinder_filters_points_without_reordering():
    rectangular = build_geometry(config(orientations="z", type="center"))
    cylinder = build_geometry(config(orientations="z", type="center",
                                        packing="cylinder_z"))
    assert len(cylinder.centers) <= len(rectangular.centers)
    # Filtering may remove candidates but must preserve the original sequence
    p_xy = rectangular.centers[:, :2]
    p_z = rectangular.centers[:, 2]
    inside_cylinder_xy = np.sum(((p_xy - 1.0) / 1.0) ** 2, axis=1) <= 1.0 + 1e-12
    inside_cylinder_z = (p_z >= -1e-12) & (p_z <= 2.0 + 1e-12)
    inside_cylinder = inside_cylinder_xy & inside_cylinder_z
    expected = [p for p in rectangular.centers
                if inside_cylinder[np.where((rectangular.centers == p).all(axis=1))[0][0]]]
    np.testing.assert_allclose(cylinder.centers, np.asarray(expected).reshape(-1, 3))

@pytest.mark.parametrize("kind", ["bad", ""])
def test_invalid_type_raises(kind):
    with pytest.raises(ValueError):
        build_geometry(config(type=kind))
