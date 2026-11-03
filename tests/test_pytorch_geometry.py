from types import SimpleNamespace

import pytest
import torch

from src.pytorch_version.geometry import (
    MetaAtom,
    build_geometry,
    orientation_normals,
)
from src.pytorch_version.geometry.packings import cylinder_z, ellipse


def make_config(**overrides):
    values = dict(
        Nx=2,
        Ny=2,
        Nz=2,
        orientations="zyx",
        type="center",
        packing="rectangle",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_geometry_builds_tensors_on_requested_device(device):
    geometry = build_geometry(make_config(orientations="z"), device=device)

    assert isinstance(geometry.centers, torch.Tensor)
    assert geometry.centers.dtype == torch.int64
    assert geometry.centers.device.type == device
    assert geometry.centers.shape == (8, 3)
    assert geometry.orientations == ("z",) * 8
    torch.testing.assert_close(
        geometry.centers[:4],
        torch.tensor(
            [[1, 1, 1], [3, 1, 1], [1, 3, 1], [3, 3, 1]],
            dtype=torch.int64,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    ("meta_atom_type", "orientation", "expected_count"),
    [
        ("border", "z", 12),
        ("open", "z", 4),
        ("center", "z", 8),
    ],
)
def test_geometry_preserves_orientation_specific_lattice_counts(
    meta_atom_type, orientation, expected_count
):
    geometry = build_geometry(
        make_config(type=meta_atom_type, orientations=orientation)
    )

    assert geometry.centers.shape == (expected_count, 3)
    assert geometry.centers.dtype == torch.int64
    assert geometry.orientations == (orientation,) * expected_count


def test_geometry_preserves_requested_orientation_block_order():
    geometry = build_geometry(make_config(orientations="yx"))

    assert geometry.orientations == ("y",) * 8 + ("x",) * 8
    assert torch.equal(geometry.centers[:8], geometry.centers[8:])


def test_orientation_normals_returns_tensor_on_requested_device():
    normals = orientation_normals(("z", "x", "y"), device="cpu")

    assert normals.dtype == torch.int64
    assert normals.device.type == "cpu"
    torch.testing.assert_close(
        normals,
        torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )


def test_torch_packings_return_boolean_masks_on_input_device():
    points = torch.tensor([[4, 2, 1], [8, 4, 1], [8, 2, 1]])

    ellipse_mask = ellipse(points, (4, 2, 3))
    cylinder_mask = cylinder_z(points, (4, 2, 3))

    assert ellipse_mask.dtype == torch.bool
    assert cylinder_mask.dtype == torch.bool
    assert ellipse_mask.device == points.device
    assert cylinder_mask.device == points.device
    assert torch.equal(cylinder_mask, torch.tensor([True, False, True]))


def test_meta_atom_geometry_uses_tensors_and_requested_device():
    atom = MetaAtom(
        Dx=2.0,
        Dy=4.0,
        Dz=6.0,
        orientations="x",
        type="border",
        device="cpu",
    )

    assert isinstance(atom.centers, torch.Tensor)
    assert isinstance(atom.normals, torch.Tensor)
    assert atom.centers.dtype == torch.float64
    assert atom.normals.dtype == torch.float64
    assert atom.centers.device.type == "cpu"
    torch.testing.assert_close(
        atom.centers,
        torch.tensor(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
