"""Geometric inclusion rules for ring centers."""

from collections.abc import Callable

import torch

Tensor = torch.Tensor
PackingRule = Callable[[Tensor, tuple[int, int, int]], Tensor]


def rectangle(points: Tensor, counts: tuple[int, int, int]) -> Tensor:
    """Keep all candidate points in the rectangular bounding box."""
    return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)


def ellipse(
    points: Tensor,
    counts: tuple[int, int, int],
    *,
    tol: float = 1e-12,
) -> Tensor:
    """Keep points inside an ellipsoid centered at the sample midpoint."""
    lengths = torch.tensor(counts, dtype=torch.float64, device=points.device)
    center = lengths
    normalized = (points.to(torch.float64) - center) / center
    return torch.sum(normalized**2, dim=1) <= 1.0 + tol


def cylinder_z(
    points: Tensor,
    counts: tuple[int, int, int],
    *,
    tol: float = 1e-12,
) -> Tensor:
    """Keep points inside an elliptical cross-section along the z axis."""
    lengths = torch.tensor(counts[:2], dtype=torch.float64, device=points.device)
    center_xy = lengths
    normalized_xy = (points[:, :2].to(torch.float64) - center_xy) / center_xy
    return torch.sum(normalized_xy**2, dim=1) <= 1.0 + tol


PACKING_RULES: dict[str, PackingRule] = {
    "rectangle": rectangle,
    "ellipse": ellipse,
    "cylinder_z": cylinder_z,
}


def get_packing_rule(name: str) -> PackingRule:
    """Return a packing rule by name."""
    try:
        return PACKING_RULES[name]
    except (KeyError, AttributeError) as exc:
        supported = ", ".join(PACKING_RULES)
        raise ValueError(
            f"Unsupported packing {name!r}. Supported packings: {supported}"
        ) from exc
