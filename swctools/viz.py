"""Visualization helpers for swctools.

- plot_centroid: skeleton plotting from SWCModel using Scatter3d
- plot_frusta: volumetric frusta plotting from FrustaSet using Mesh3d
"""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go
import logging
from contextlib import contextmanager

from .geometry import FrustaSet, PointSet
from .config import apply_layout


# @contextmanager
# def _suppress_geometry_logs():
#     lg = logging.getLogger("swctools.geometry")
#     prev_level = lg.level
#     try:
#         lg.setLevel(max(logging.WARNING, prev_level or 0))
#         yield
#     finally:
#         lg.setLevel(prev_level)


def plot_centroid(
    swc_model: SWCModel,
    *,
    marker_size: float = 2.0,
    line_width: float = 2.0,
    show_nodes: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Plot centroid skeleton from an `SWCModel`.

    Edges are drawn as line segments in 3D using Scatter3d.
    """
    logger = logging.getLogger(__name__)
    xs = []
    ys = []
    zs = []

    # Build polyline segments with None separators for Plotly
    for u, v in swc_model.edges:
        xs.extend([swc_model.nodes[u]["x"], swc_model.nodes[v]["x"], None])
        ys.extend([swc_model.nodes[u]["y"], swc_model.nodes[v]["y"], None])
        zs.extend([swc_model.nodes[u]["z"], swc_model.nodes[v]["z"], None])

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
        xn = [swc_model.nodes[n]["x"] for n in swc_model.nodes]
        yn = [swc_model.nodes[n]["y"] for n in swc_model.nodes]
        zn = [swc_model.nodes[n]["z"] for n in swc_model.nodes]
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
    apply_layout(fig, title=title or "Centroid Skeleton")
    logger.info(
        "plot_centroid edges=%d show_nodes=%s", len(list(swc_model.edges)), show_nodes
    )
    return fig


def plot_frusta(
    frusta: FrustaSet,
    *,
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
    tag_colors: dict[int, str] | None = None,
    title: str | None = None,
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
    tag_colors: dict[int, str] | None
        Optional mapping {tag: color}. If provided, each frustum segment is colored
        uniformly according to its tag (fallback to `color` if a tag is missing).
    """
    logger = logging.getLogger(__name__)
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    if tag_colors is not None:
        slices = fr.per_segment_face_slices()
        facecolor: list[str] = []
        for (start, count), seg in zip(slices, fr.segments):
            c = tag_colors.get(seg.tag, color)
            facecolor.extend([c] * count)
        mesh = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            facecolor=facecolor,
            opacity=opacity,
            flatshading=flatshading,
        )
    else:
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
    apply_layout(fig, title=title or "Frusta Mesh")
    logger.info(
        "plot_frusta segments=%d radius_scale=%s flatshading=%s",
        frusta.segment_count,
        radius_scale,
        flatshading,
    )
    return fig


def plot_frusta_with_centroid(
    swc_model: SWCModel,
    frusta: FrustaSet,
    *,
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
    tag_colors: dict[int, str] | None = None,
    centroid_color: str = "#1f77b4",
    centroid_line_width: float = 2.0,
    show_nodes: bool = False,
    node_size: float = 2.0,
    title: str | None = None,
) -> go.Figure:
    """Overlay frusta mesh with centroid skeleton from an `SWCModel`.

    Parameters mirror `plot_centroid` and `plot_frusta` with an extra `radius_scale`.
    """
    logger = logging.getLogger(__name__)
    # Build centroid polyline
    xs, ys, zs = [], [], []
    for u, v in swc_model.edges:
        xs.extend([swc_model.nodes[u]["x"], swc_model.nodes[v]["x"], None])
        ys.extend([swc_model.nodes[u]["y"], swc_model.nodes[v]["y"], None])
        zs.extend([swc_model.nodes[u]["z"], swc_model.nodes[v]["z"], None])
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
        xn = [swc_model.nodes[n]["x"] for n in swc_model.nodes]
        yn = [swc_model.nodes[n]["y"] for n in swc_model.nodes]
        zn = [swc_model.nodes[n]["z"] for n in swc_model.nodes]
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
    if tag_colors is not None:
        slices = fr.per_segment_face_slices()
        facecolor: list[str] = []
        for (start, count), seg in zip(slices, fr.segments):
            c = tag_colors.get(seg.tag, color)
            facecolor.extend([c] * count)
        mesh = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            facecolor=facecolor,
            opacity=opacity,
            flatshading=flatshading,
            name="frusta",
        )
    else:
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
    apply_layout(fig, title=title or "Centroid + Frusta")
    logger.info(
        "plot_frusta_with_centroid edges=%d segments=%d radius_scale=%s",
        len(list(swc_model.edges)),
        frusta.segment_count,
        radius_scale,
    )
    return fig


def plot_frusta_slider(
    frusta: FrustaSet,
    *,
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    tag_colors: dict[int, str] | None = None,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    steps: int = 21,
    title: str | None = None,
) -> go.Figure:
    """Interactive slider (0..1 default) controlling uniform `radius_scale`.

    Precomputes frames at evenly spaced scales between `min_scale` and `max_scale`.
    """
    logger = logging.getLogger(__name__)
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
    # with _suppress_geometry_logs():
    init_fr = base if init_scale == 1.0 else base.scaled(init_scale)
    x0, y0, z0, _, _, _ = init_fr.to_mesh3d_arrays()

    if tag_colors is not None:
        slices = base.per_segment_face_slices()
        facecolor0: list[str] = []
        for (start, count), seg in zip(slices, base.segments):
            c = tag_colors.get(seg.tag, color)
            facecolor0.extend([c] * count)
        mesh = go.Mesh3d(
            x=x0,
            y=y0,
            z=z0,
            i=bi,
            j=bj,
            k=bk,
            facecolor=facecolor0,
            opacity=opacity,
            flatshading=flatshading,
            name="frusta",
        )
    else:
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
    # with _suppress_geometry_logs():
    for s in scales:
        fr_s = base if s == 1.0 else base.scaled(s)
        xs, ys, zs, _, _, _ = fr_s.to_mesh3d_arrays()
        if tag_colors is not None:
            frames.append(
                go.Frame(
                    name=f"scale={s:.2f}",
                    data=[
                        go.Mesh3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            i=bi,
                            j=bj,
                            k=bk,
                            facecolor=facecolor0,
                            opacity=opacity,
                            flatshading=flatshading,
                        )
                    ],
                )
            )
        else:
            frames.append(
                go.Frame(
                    name=f"scale={s:.2f}",
                    data=[
                        go.Mesh3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            i=bi,
                            j=bj,
                            k=bk,
                            color=color,
                            opacity=opacity,
                            flatshading=flatshading,
                        )
                    ],
                )
            )

    # Slider and play controls
    slider_steps = [
        {
            "label": f"{s:.2f}",
            "method": "animate",
            "args": [
                [f"scale={s:.2f}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0},
                    "transition": {"duration": 0},
                },
            ],
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
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "fromcurrent": True,
                            "frame": {"duration": 0},
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "❚❚ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0},
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]

    fig = go.Figure(data=[mesh], frames=frames)
    apply_layout(fig, title=title or "Frusta Mesh — radius_scale slider")
    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    logger.info(
        "plot_frusta_slider segments=%d scales=%d min=%s max=%s",
        frusta.segment_count,
        len(scales),
        min_scale,
        max_scale,
    )
    return fig


def plot_model(
    *,
    swc_model: SWCModel | None = None,
    frusta: FrustaSet | None = None,
    show_frusta: bool = True,
    show_centroid: bool = True,
    title: str | None = None,
    # Frusta build options (used if frusta is None and gm provided)
    sides: int = 16,
    end_caps: bool = False,
    # Frusta appearance
    color: str = "lightblue",
    opacity: float = 0.8,
    flatshading: bool = True,
    tag_colors: dict[int, str] | None = None,
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

    logger = logging.getLogger(__name__)
    traces: list[go.BaseTraceType] = []
    frames: list[go.Frame] | None = None

    # Build frusta if needed
    base_fr = frusta
    if show_frusta and base_fr is None:
        if swc_model is None:
            raise ValueError(
                "plot_model: provide either `frusta` or a `gm` to build from"
            )
        base_fr = FrustaSet.from_swc_model(swc_model, sides=sides, end_caps=end_caps)

    # Centroid traces
    if show_centroid and swc_model is not None:
        xs, ys, zs = [], [], []
        for u, v in swc_model.edges:
            xs.extend([swc_model.nodes[u]["x"], swc_model.nodes[v]["x"], None])
            ys.extend([swc_model.nodes[u]["y"], swc_model.nodes[v]["y"], None])
            zs.extend([swc_model.nodes[u]["z"], swc_model.nodes[v]["z"], None])
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
            xn = [swc_model.nodes[n]["x"] for n in swc_model.nodes]
            yn = [swc_model.nodes[n]["y"] for n in swc_model.nodes]
            zn = [swc_model.nodes[n]["z"] for n in swc_model.nodes]
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
                init_idx = min(
                    range(len(scales)), key=lambda idx: abs(scales[idx] - 1.0)
                )
            else:
                init_idx = 0
            init_scale = scales[init_idx]
            # with _suppress_geometry_logs():
            init_fr = base_fr if init_scale == 1.0 else base_fr.scaled(init_scale)
            x0, y0, z0, _, _, _ = init_fr.to_mesh3d_arrays()

            if tag_colors is not None:
                slices = base_fr.per_segment_face_slices()
                facecolor0: list[str] = []
                for (start, count), seg in zip(slices, base_fr.segments):
                    c = tag_colors.get(seg.tag, color)
                    facecolor0.extend([c] * count)
                mesh = go.Mesh3d(
                    x=x0,
                    y=y0,
                    z=z0,
                    i=bi,
                    j=bj,
                    k=bk,
                    facecolor=facecolor0,
                    opacity=opacity,
                    flatshading=flatshading,
                    name="frusta",
                )
            else:
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
            # with _suppress_geometry_logs():
            for s in scales:
                fr_s = base_fr if s == 1.0 else base_fr.scaled(s)
                xs, ys, zs, _, _, _ = fr_s.to_mesh3d_arrays()
                if tag_colors is not None:
                    frames.append(
                        go.Frame(
                            name=f"scale={s:.2f}",
                            data=[
                                go.Mesh3d(
                                    x=xs,
                                    y=ys,
                                    z=zs,
                                    i=bi,
                                    j=bj,
                                    k=bk,
                                    facecolor=facecolor0,
                                    opacity=opacity,
                                    flatshading=flatshading,
                                )
                            ],
                        )
                    )
                else:
                    frames.append(
                        go.Frame(
                            name=f"scale={s:.2f}",
                            data=[
                                go.Mesh3d(
                                    x=xs,
                                    y=ys,
                                    z=zs,
                                    i=bi,
                                    j=bj,
                                    k=bk,
                                    color=color,
                                    opacity=opacity,
                                    flatshading=flatshading,
                                )
                            ],
                        )
                    )

            slider_steps = [
                {
                    "label": f"{s:.2f}",
                    "method": "animate",
                    "args": [
                        [f"scale={s:.2f}"],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0},
                            "transition": {"duration": 0},
                        },
                    ],
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
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "frame": {"duration": 0},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "❚❚ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ]

            fig = go.Figure(data=traces, frames=frames)
            apply_layout(fig, title=title or "Model")
            fig.update_layout(sliders=sliders, updatemenus=updatemenus)
            logger.info(
                "plot_model slider=True segments=%d radius_scale_range=[%s,%s]",
                base_fr.segment_count,
                min_scale,
                max_scale,
            )
            return fig
        else:
            # Static radius scale
            fr = base_fr if radius_scale == 1.0 else base_fr.scaled(radius_scale)
            x, y, z, i, j, k = fr.to_mesh3d_arrays()
            if tag_colors is not None:
                slices = fr.per_segment_face_slices()
                facecolor: list[str] = []
                for (start, count), seg in zip(slices, fr.segments):
                    c = tag_colors.get(seg.tag, color)
                    facecolor.extend([c] * count)
                mesh = go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    facecolor=facecolor,
                    opacity=opacity,
                    flatshading=flatshading,
                    name="frusta",
                )
            else:
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
    apply_layout(fig, title=title or "Model")
    logger.info(
        "plot_model slider=False segments=%s show_frusta=%s show_centroid=%s",
        base_fr.segment_count if base_fr is not None else None,
        show_frusta,
        show_centroid,
    )
    return fig


def plot_frusta_timeseries(
    frusta: FrustaSet,
    values: Sequence[Sequence[float]],
    *,
    colorscale: str | list = "Viridis",
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
    fps: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Animate per-segment values over time by coloring frusta faces.

    Parameters
    ----------
    frusta: FrustaSet
        Batched frusta mesh. Optionally scaled via `radius_scale` before rendering.
    values: Sequence[Sequence[float]]
        Time series V_i(t) shaped [T][N], where N = `frusta.segment_count`.
        Each time step provides one scalar per segment in the current order.
    colorscale: str | list
        Plotly colorscale for mapping intensities.
    cmin, cmax: float | None
        Fixed color range. If omitted, inferred from the data across all frames.
    opacity: float
        Mesh opacity.
    flatshading: bool
        Enable flat shading on the mesh.
    radius_scale: float
        Uniform radius scaling applied before meshing (1.0 = no change).
    fps: int
        Playback frame rate for the Play button.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with a Mesh3d trace and animation frames, plus a slider and play/pause controls.
    """
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    slices = fr.per_segment_face_slices()
    if len(values) == 0:
        raise ValueError("values must have at least one time step")
    T = len(values)
    N = fr.segment_count
    if any(len(vt) != N for vt in values):
        raise ValueError(
            "each time step must have N values, matching frusta.segment_count"
        )

    def faces_intensity(vt: Sequence[float]) -> list[float]:
        arr: list[float] = []
        for (start, count), val in zip(slices, vt):
            arr.extend([float(val)] * count)
        return arr

    if cmin is None or cmax is None:
        vmin = min(min(vt) for vt in values)
        vmax = max(max(vt) for vt in values)
        if cmin is None:
            cmin = float(vmin)
        if cmax is None:
            cmax = float(vmax)
    init = values[0]
    intensity0 = faces_intensity(init)
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        opacity=opacity,
        flatshading=flatshading,
        intensity=intensity0,
        intensitymode="cell",
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        showscale=True,
        name="frusta",
    )
    frames = []
    for t, vt in enumerate(values):
        intens = faces_intensity(vt)
        frames.append(
            go.Frame(
                name=f"t={t}",
                data=[
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        opacity=opacity,
                        flatshading=flatshading,
                        intensity=intens,
                        intensitymode="cell",
                        colorscale=colorscale,
                        cmin=cmin,
                        cmax=cmax,
                    )
                ],
            )
        )
    frame_duration = int(1000 / max(1, fps))
    slider_steps = [
        {
            "label": f"{t}",
            "method": "animate",
            "args": [
                [f"t={t}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0},
                    "transition": {"duration": 0},
                },
            ],
        }
        for t in range(T)
    ]
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "t: ", "visible": True},
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
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "fromcurrent": True,
                            "frame": {"duration": frame_duration},
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "❚❚ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0},
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]
    fig = go.Figure(data=[mesh], frames=frames)
    apply_layout(fig, title=title or "Frusta timeseries")
    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    return fig
