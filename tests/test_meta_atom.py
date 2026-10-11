
import numpy as np
import pytest

from src.metamaterial.geometry import MetaAtom


def test_border_meta_atom_has_two_rings_per_orientation():
    meta_atom = MetaAtom(
        Dx=2.0,
        Dy=4.0,
        Dz=6.0,
        orientations="zyx",
        type="border",
    )

    assert meta_atom.n_rings == 6
    assert meta_atom.centers.shape == (6, 3)
    assert meta_atom.normals.shape == (6, 3)


def test_center_meta_atom_has_one_ring_per_orientation():
    meta_atom = MetaAtom(
        Dx=2.0,
        Dy=4.0,
        Dz=6.0,
        orientations="zyx",
        type="center",
    )

    assert meta_atom.n_rings == 3
    np.testing.assert_array_equal(
        meta_atom.centers,
        np.zeros((3, 3)),
    )


def test_border_ring_centers_are_on_cell_faces():
    meta_atom = MetaAtom(
        Dx=2.0,
        Dy=4.0,
        Dz=6.0,
        orientations="x",
        type="border",
    )

    expected = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    np.testing.assert_allclose(
        meta_atom.centers,
        expected,
    )


def test_normals_do_not_depend_on_face_sign():
    meta_atom = MetaAtom(
        Dx=2.0,
        Dy=2.0,
        Dz=2.0,
        orientations="x",
        type="border",
    )

    expected = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    np.testing.assert_array_equal(
        meta_atom.normals,
        expected,
    )


@pytest.mark.parametrize(
    "meta_atom_type",
    ["border", "open", "center"],
)
def test_number_of_rings_for_each_type(meta_atom_type):
    meta_atom = MetaAtom(
        Dx=2.0,
        Dy=2.0,
        Dz=2.0,
        orientations="zyx",
        type=meta_atom_type,
    )

    expected = 3 if meta_atom_type == "center" else 6

    assert meta_atom.n_rings == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Dx": 0.0, "Dy": 2.0, "Dz": 2.0},
        {"Dx": -1.0, "Dy": 2.0, "Dz": 2.0},
    ],
)
def test_invalid_dimensions_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        MetaAtom(
            **kwargs,
            orientations="x",
            type="border",
        )


@pytest.mark.parametrize(
    "orientations",
    ["", "xx", "xyq", "xyzx"],
)
def test_invalid_orientations_raise_value_error(orientations):
    with pytest.raises(ValueError):
        MetaAtom(
            Dx=2.0,
            Dy=2.0,
            Dz=2.0,
            orientations=orientations,
            type="border",
        )


def test_invalid_type_raises_value_error():
    with pytest.raises(ValueError):
        MetaAtom(
            Dx=2.0,
            Dy=2.0,
            Dz=2.0,
            orientations="xyz",
            type="unknown",
        )
