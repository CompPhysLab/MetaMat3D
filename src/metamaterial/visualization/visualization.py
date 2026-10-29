from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry, orientation_normals
ComplexArray = npt.NDArray[np.complex128]

ORIENTATION_COLORS = {
    "x": "blue",
    "y": "red",
    "z": "green",
}

def plot_geometry( geometry: Geometry, config: SimulationConfig, orientations: Iterable[str] | None = None, *, n_points: int = 100, show_legend: bool = True) -> go.Figure:
    """Plots resonator rings in a 3D Plotly figure
    Parameters
    ----------
    geometry
        Geometry containing resonator centers and orientations
    config
        Simulation configuration, providing the ring radius
    orientations
        Orientations to display. If None, display all orientations
    n_points
        Number of points used to draw each ring
    show_legend
        Whether to display one legend entry per orientation

    Returns
    -------
    go.Figure
        Interactive 3D Plotly figure
    """
    if orientations is None:
        selected = set(config.orientations)
    else:
        selected = set(orientations)
    fig = go.Figure()

    centers = np.asarray(geometry.centers)
    orientations = np.asarray(geometry.orientations)
    normals = orientation_normals(orientations)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    radius = config.radius
    spacings = np.asarray([config.Dx, config.Dy, config.Dz], dtype=np.float64)

    for axis in ("x", "y", "z"):
        if axis not in selected:
            continue

        normal = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }[axis]

        indices = np.flatnonzero(
            np.all(np.isclose(normals, normal), axis=1)
        )

        for j, idx in enumerate(indices):
            x0, y0, z0 = centers[idx] * spacings / 2

            if axis == "x":
                x = np.full_like(theta, x0)
                y = y0 + radius * np.cos(theta)
                z = z0 + radius * np.sin(theta)
            elif axis == "y":
                x = x0 + radius * np.cos(theta)
                y = np.full_like(theta, y0)
                z = z0 + radius * np.sin(theta)
            else:
                x = x0 + radius * np.cos(theta)
                y = y0 + radius * np.sin(theta)
                z = np.full_like(theta, z0)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color=ORIENTATION_COLORS[axis]),
                    name=f"{axis.upper()} orientation",
                    legendgroup=axis,
                    showlegend=show_legend and j == 0,
                    hovertemplate=(
                        f"Orientation: {axis}<br>"
                        f"Ring index: {idx}<br>"
                        "x=%{x:.4g}<br>"
                        "y=%{y:.4g}<br>"
                        "z=%{z:.4g}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        legend_title_text="Ring orientation",
    )

    return fig