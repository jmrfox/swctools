"""Visualization helpers for swctools.

- plot_centroid: skeleton plotting from GeneralModel using Scatter3d
- plot_frusta: volumetric frusta plotting from FrustaSet using Mesh3d
"""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go

from .geometry import FrustaSet, PointSet
from .config import apply_layout


def plot_centroid(gm, *, marker_size: float = 2.0, line_width: float = 2.0, show_nodes: bool = True) -> go.Figure:
    """Plot centroid skeleton from a GeneralModel.

    Edges are drawn as line segments in 3D using Scatter3d.
    """
    xs = []
    ys = []
    zs = []

    # Build polyline segments with None separators for Plotly
    for u, v in gm.edges:
        xs.extend([gm.nodes[u]["x"], gm.nodes[v]["x"], None])
        ys.extend([gm.nodes[u]["y"], gm.nodes[v]["y"], None])
        zs.extend([gm.nodes[u]["z"], gm.nodes[v]["z"], None])

    edge_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(width=line_width, color="#1f77b4"),
        name="edges",
    )

    data = [edge_trace]

    if show_nodes:
        xn = [gm.nodes[n]["x"] for n in gm.nodes]
        yn = [gm.nodes[n]["y"] for n in gm.nodes]
        zn = [gm.nodes[n]["z"] for n in gm.nodes]
        node_trace = go.Scatter3d(
            x=xn,
            y=yn,
            z=zn,
            mode="markers",
            marker=dict(size=marker_size, color="#ff7f0e"),
            name="nodes",
        )
        data.append(node_trace)

    fig = go.Figure(data=data)
    apply_layout(fig, title="Centroid Skeleton")
    return fig


def plot_frusta(
    frusta: FrustaSet,
    *,
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
) -> go.Figure:
    """Plot a FrustaSet as a Mesh3d figure.

    Parameters
    ----------
    frusta: FrustaSet
        Batched frusta mesh to render.
    color: str
        Mesh color.
    opacity: float
        Mesh opacity.
    flatshading: bool
        Whether to enable flat shading.
    radius_scale: float
        Uniform scale applied to all segment radii before meshing (1.0 = no change).
    """
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        flatshading=flatshading,
    )
    fig = go.Figure(data=[mesh])
    apply_layout(fig, title="Frusta Mesh")
    return fig


def plot_frusta_with_centroid(
    gm,
    frusta: FrustaSet,
    *,
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
    centroid_color: str = "#1f77b4",
    centroid_line_width: float = 2.0,
    show_nodes: bool = False,
    node_size: float = 2.0,
) -> go.Figure:
    """Overlay frusta mesh with centroid skeleton from a `GeneralModel`.

    Parameters mirror `plot_centroid` and `plot_frusta` with an extra `radius_scale`.
    """
    # Build centroid polyline
    xs, ys, zs = [], [], []
    for u, v in gm.edges:
        xs.extend([gm.nodes[u]["x"], gm.nodes[v]["x"], None])
        ys.extend([gm.nodes[u]["y"], gm.nodes[v]["y"], None])
        zs.extend([gm.nodes[u]["z"], gm.nodes[v]["z"], None])
    centroid = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(width=centroid_line_width, color=centroid_color),
        name="centroid",
    )

    traces = [centroid]
    if show_nodes:
        xn = [gm.nodes[n]["x"] for n in gm.nodes]
        yn = [gm.nodes[n]["y"] for n in gm.nodes]
        zn = [gm.nodes[n]["z"] for n in gm.nodes]
        nodes = go.Scatter3d(
            x=xn,
            y=yn,
            z=zn,
            mode="markers",
            marker=dict(size=node_size, color="#ff7f0e"),
            name="nodes",
        )
        traces.append(nodes)

    # Frusta mesh (optionally scaled)
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        flatshading=flatshading,
        name="frusta",
    )
    traces.append(mesh)

    fig = go.Figure(data=traces)
    apply_layout(fig, title="Centroid + Frusta")
    return fig


def plot_frusta_slider(
    frusta: FrustaSet,
    *,
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    steps: int = 21,
) -> go.Figure:
    """Interactive slider (0..1 default) controlling uniform `radius_scale`.

    Precomputes frames at evenly spaced scales between `min_scale` and `max_scale`.
    """
    steps = max(2, int(steps))
    span = max_scale - min_scale
    scales = [min_scale + (span * k / (steps - 1)) for k in range(steps)]

    # Use i/j/k topology from the unscaled mesh
    base = frusta
    bx, by, bz, bi, bj, bk = base.to_mesh3d_arrays()

    # Initial view: prefer scale = 1.0 if within range; otherwise first scale
    if min_scale <= 1.0 <= max_scale:
        init_idx = min(range(len(scales)), key=lambda idx: abs(scales[idx] - 1.0))
    else:
        init_idx = 0
    init_scale = scales[init_idx]
    init_fr = base if init_scale == 1.0 else base.scaled(init_scale)
    x0, y0, z0, _, _, _ = init_fr.to_mesh3d_arrays()

    mesh = go.Mesh3d(
        x=x0,
        y=y0,
        z=z0,
        i=bi,
        j=bj,
        k=bk,
        color=color,
        opacity=opacity,
        flatshading=flatshading,
        name="frusta",
    )

    frames = []
    for s in scales:
        fr_s = base if s == 1.0 else base.scaled(s)
        xs, ys, zs, _, _, _ = fr_s.to_mesh3d_arrays()
        frames.append(
            go.Frame(
                name=f"scale={s:.2f}",
                data=[go.Mesh3d(x=xs, y=ys, z=zs, i=bi, j=bj, k=bk, color=color, opacity=opacity, flatshading=flatshading)],
            )
        )

    # Slider and play controls
    slider_steps = [
        {
            "label": f"{s:.2f}",
            "method": "animate",
            "args": [[f"scale={s:.2f}"], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}],
        }
        for s in scales
    ]

    sliders = [
        {
            "active": init_idx,
            "currentvalue": {"prefix": "radius_scale: ", "visible": True},
            "steps": slider_steps,
        }
    ]

    updatemenus = [
        {
            "type": "buttons",
            "direction": "left",
            "pad": {"r": 10, "t": 60},
            "showactive": False,
            "x": 0.0,
            "y": 0,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 0}, "transition": {"duration": 0}}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}]},
            ],
        }
    ]

    fig = go.Figure(data=[mesh], frames=frames)
    apply_layout(fig, title="Frusta Mesh — radius_scale slider")
    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    return fig


def plot_model(
    *,
    gm=None,
    frusta: FrustaSet | None = None,
    show_frusta: bool = True,
    show_centroid: bool = True,
    # Frusta build options (used if frusta is None and gm provided)
    sides: int = 16,
    end_caps: bool = False,
    # Frusta appearance
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    # Scaling and interactivity
    radius_scale: float = 1.0,
    slider: bool = False,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    steps: int = 21,
    # Centroid appearance
    centroid_color: str = "#1f77b4",
    centroid_line_width: float = 2.0,
    show_nodes: bool = False,
    node_size: float = 2.0,
    # Extra points overlay (as low-res spheres)
    point_set: PointSet | None = None,
    point_size: float = 1.0,
    point_color: str = "#d62728",
) -> go.Figure:
    """Master visualization combining centroid, frusta, slider, and overlay points.

    - If `frusta` is not provided and `gm` is, a `FrustaSet` is built from `gm`.
    - If `slider=True` and `show_frusta=True`, a Plotly slider controls `radius_scale`.
    - `points` overlays arbitrary xyz positions as small markers.
    """

    traces: list[go.BaseTraceType] = []
    frames: list[go.Frame] | None = None

    # Build frusta if needed
    base_fr = frusta
    if show_frusta and base_fr is None:
        if gm is None:
            raise ValueError("plot_model: provide either `frusta` or a `gm` to build from")
        base_fr = FrustaSet.from_general_model(gm, sides=sides, end_caps=end_caps)

    # Centroid traces
    if show_centroid and gm is not None:
        xs, ys, zs = [], [], []
        for u, v in gm.edges:
            xs.extend([gm.nodes[u]["x"], gm.nodes[v]["x"], None])
            ys.extend([gm.nodes[u]["y"], gm.nodes[v]["y"], None])
            zs.extend([gm.nodes[u]["z"], gm.nodes[v]["z"], None])
        centroid = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(width=centroid_line_width, color=centroid_color),
            name="centroid",
        )
        traces.append(centroid)

        if show_nodes:
            xn = [gm.nodes[n]["x"] for n in gm.nodes]
            yn = [gm.nodes[n]["y"] for n in gm.nodes]
            zn = [gm.nodes[n]["z"] for n in gm.nodes]
            nodes = go.Scatter3d(
                x=xn,
                y=yn,
                z=zn,
                mode="markers",
                marker=dict(size=node_size, color="#ff7f0e"),
                name="nodes",
            )
            traces.append(nodes)

    # Overlay points as small spheres mesh
    if point_set is not None:
        ps = point_set if point_size == 1.0 else point_set.scaled(point_size)
        px, py, pz, pi, pj, pk = ps.to_mesh3d_arrays()
        pts_mesh = go.Mesh3d(
            x=px,
            y=py,
            z=pz,
            i=pi,
            j=pj,
            k=pk,
            color=point_color,
            opacity=1.0,
            flatshading=True,
            name="points",
        )
        # Keep points above centroid but above frusta ordering set below
        traces.append(pts_mesh)

    # Frusta (optionally with slider)
    if show_frusta and base_fr is not None:
        # Use base topology and update x/y/z with radius scales
        bx, by, bz, bi, bj, bk = base_fr.to_mesh3d_arrays()

        if slider:
            span = max_scale - min_scale
            steps = max(2, int(steps))
            scales = [min_scale + (span * k / (steps - 1)) for k in range(steps)]
            # Pick initial scale near 1.0 if in range
            if min_scale <= 1.0 <= max_scale:
                init_idx = min(range(len(scales)), key=lambda idx: abs(scales[idx] - 1.0))
            else:
                init_idx = 0
            init_scale = scales[init_idx]
            init_fr = base_fr if init_scale == 1.0 else base_fr.scaled(init_scale)
            x0, y0, z0, _, _, _ = init_fr.to_mesh3d_arrays()

            mesh = go.Mesh3d(
                x=x0,
                y=y0,
                z=z0,
                i=bi,
                j=bj,
                k=bk,
                color=color,
                opacity=opacity,
                flatshading=flatshading,
                name="frusta",
            )

            # Ensure mesh is the FIRST trace so frames can update just this trace
            traces = [mesh] + traces

            frames = []
            for s in scales:
                fr_s = base_fr if s == 1.0 else base_fr.scaled(s)
                xs, ys, zs, _, _, _ = fr_s.to_mesh3d_arrays()
                frames.append(
                    go.Frame(
                        name=f"scale={s:.2f}",
                        data=[go.Mesh3d(x=xs, y=ys, z=zs, i=bi, j=bj, k=bk, color=color, opacity=opacity, flatshading=flatshading)],
                    )
                )

            slider_steps = [
                {
                    "label": f"{s:.2f}",
                    "method": "animate",
                    "args": [[f"scale={s:.2f}"], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}],
                }
                for s in scales
            ]

            sliders = [
                {
                    "active": init_idx,
                    "currentvalue": {"prefix": "radius_scale: ", "visible": True},
                    "steps": slider_steps,
                }
            ]

            updatemenus = [
                {
                    "type": "buttons",
                    "direction": "left",
                    "pad": {"r": 10, "t": 60},
                    "showactive": False,
                    "x": 0.0,
                    "y": 0,
                    "buttons": [
                        {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 0}, "transition": {"duration": 0}}]},
                        {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}]},
                    ],
                }
            ]

            fig = go.Figure(data=traces, frames=frames)
            apply_layout(fig, title="Model")
            fig.update_layout(sliders=sliders, updatemenus=updatemenus)
            return fig
        else:
            # Static radius scale
            fr = base_fr if radius_scale == 1.0 else base_fr.scaled(radius_scale)
            x, y, z, i, j, k = fr.to_mesh3d_arrays()
            mesh = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=opacity,
                flatshading=flatshading,
                name="frusta",
            )
            traces.insert(0, mesh)  # keep mesh on bottom for visibility

    fig = go.Figure(data=traces)
    apply_layout(fig, title="Model")
    return fig
