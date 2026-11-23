import numpy as np
import pytest
import torch
from scipy.interpolate import BarycentricInterpolator as SciPyBarycentricInterpolator

from src.metamaterial.physics.coupling.operator import (
    mutual_inductance as numpy_mutual_inductance,
)
from src.pytorch_version.physics.coupling.approx_retarded import (
    BarycentricInterpolator,
)
from src.pytorch_version.physics.coupling.operator import (
    _mutual_for_pairs,
    mutual_inductance as torch_mutual_inductance,
)


@pytest.mark.parametrize(
    "orientation1,orientation2,lattice_displacement,width",
    [
        ("z", "z", (1, 1, 1), 0.0),
        ("z", "z", (200, 0, 0), 0.1e-3),
        ("z", "y", (1, 1, 1), 0.0),
        ("z", "y", (200, 200, 200), 0.1e-3),
    ],
)
def test_torch_approx_retarded_matches_numpy(
    orientation1, orientation2, lattice_displacement, width
):
    omega = 2.0 * np.pi * 1e8
    spacings = np.array([3e-3, 3e-3, 3e-3])
    lattice_displacement = np.asarray(lattice_displacement)
    physical_displacement = lattice_displacement * spacings / 2.0
    arguments = {
        "r1": 0.5e-3,
        "r2": 0.5e-3,
        "width": width,
        "orientation1": orientation1,
        "orientation2": orientation2,
        "model": "approx_retarded",
        "omega": omega,
    }

    expected = numpy_mutual_inductance(
        displacement=lattice_displacement,
        spacings=spacings,
        **arguments,
    )
    actual = torch_mutual_inductance(
        displacement=physical_displacement,
        **arguments,
    )

    np.testing.assert_allclose(
        actual.item(),
        expected,
        rtol=2e-8,
        atol=1e-25,
    )


def test_approx_retarded_selects_full_and_recurrent_models_per_pair():
    omega = 2.0 * np.pi * 1e8
    spacings = np.array([3e-3, 3e-3, 3e-3])
    lattice_displacements = np.array([[1, 1, 1], [200, 0, 0]])
    physical_displacements = torch.as_tensor(
        lattice_displacements * spacings / 2.0,
        dtype=torch.float64,
    )
    actual = _mutual_for_pairs(
        physical_displacements,
        "z",
        "z",
        radius1=0.5e-3,
        radius2=0.5e-3,
        width=0.0,
        model="approx_retarded",
        omega=omega,
    )
    expected = np.array(
        [
            numpy_mutual_inductance(
                displacement=displacement,
                spacings=spacings,
                r1=0.5e-3,
                r2=0.5e-3,
                width=0.0,
                orientation1="z",
                orientation2="z",
                model="approx_retarded",
                omega=omega,
            )
            for displacement in lattice_displacements
        ]
    )

    np.testing.assert_allclose(
        actual.numpy(),
        expected,
        rtol=2e-8,
        atol=1e-25,
    )


@pytest.mark.parametrize(
    ("orientation1", "orientation2"),
    [
        (orientation1, orientation2)
        for orientation1 in "xyz"
        for orientation2 in "xyz"
    ],
)
def test_torch_far_approx_retarded_matches_numpy_for_all_orientations(
    orientation1, orientation2
):
    omega = 2.0 * np.pi * 1e8
    spacings = np.array([3e-3, 3e-3, 3e-3])
    lattice_displacement = np.array([200, 200, 200])
    arguments = {
        "r1": 0.5e-3,
        "r2": 0.5e-3,
        "width": 0.1e-3,
        "orientation1": orientation1,
        "orientation2": orientation2,
        "model": "approx_retarded",
        "omega": omega,
    }

    expected = numpy_mutual_inductance(
        displacement=lattice_displacement,
        spacings=spacings,
        **arguments,
    )
    actual = torch_mutual_inductance(
        displacement=lattice_displacement * spacings / 2.0,
        **arguments,
    )

    np.testing.assert_allclose(
        actual.item(),
        expected,
        rtol=2e-8,
        atol=1e-25,
    )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_torch_barycentric_interpolator_matches_scipy_for_complex_values(device):
    nodes = torch.cos(
        torch.pi * torch.arange(201, dtype=torch.float64) / 200
    ).to(device)
    values = torch.sin(3.0 * nodes) + 1j * torch.cos(5.0 * nodes)
    query = torch.tensor(
        [-1.0, -0.75, -0.1, 0.0, 0.4, 0.9, 1.0], device=device
    )

    actual = BarycentricInterpolator(nodes, values)(query)
    expected = SciPyBarycentricInterpolator(
        nodes.cpu().numpy(), values.cpu().numpy()
    )(query.cpu().numpy())

    np.testing.assert_allclose(
        actual.cpu().numpy(), expected, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        BarycentricInterpolator(nodes, values)(nodes[100]),
        values[100],
        rtol=0.0,
        atol=0.0,
    )
