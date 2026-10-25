
import numpy as np
import pytest
from src.metamaterial.physics.coupling.operator import mutual_inductance

@pytest.mark.parametrize(
    ("orientation1", "orientation2"),
    [
        ("x", "x"),
        ("x", "y"),
        ("x", "z"),
        ("y", "x"),
        ("y", "y"),
        ("y", "z"),
        ("z", "x"),
        ("z", "y"),
        ("z", "z"),
    ],
)
@pytest.mark.parametrize(
    "displacement",
    [
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (2, 2, 0),
        (2, -2, 2),
        (-4, 2, 0),
    ],
)
def test_mutual_inductance_reciprocity(
    orientation1,
    orientation2,
    displacement,
):
    spacings = (1.0, 1.3, 0.8)
    r1 = 0.12
    r2 = 0.15
    width = 0.02

    m12 = mutual_inductance(
        displacement=displacement,
        spacings=spacings,
        r1=r1,
        r2=r2,
        width=width,
        orientation1=orientation1,
        orientation2=orientation2,
    )

    reverse_displacement = tuple(-v for v in displacement)

    m21 = mutual_inductance(
        displacement=reverse_displacement,
        spacings=spacings,
        r1=r2,
        r2=r1,
        width=width,
        orientation1=orientation2,
        orientation2=orientation1,
    )

    np.testing.assert_allclose(
        m12,
        m21,
        rtol=1e-10,
        atol=1e-14,
        err_msg=(
            f"Reciprocity failed for orientations "
            f"{orientation1}, {orientation2}; "
            f"displacement={displacement}"
        ),
    )