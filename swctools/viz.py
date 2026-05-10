"""Visualization helpers for swctools.

- plot_centroid: skeleton plotting from SWCModel using Scatter3d
- plot_frusta: volumetric frusta plotting from FrustaSet using Mesh3d
"""

from __future__ import annotations

from typing import Sequence
import math

import plotly.graph_objects as go
import logging

from .model import SWCModel
from .geometry import FrustaSet, PointSet
from .config import apply_layout


# ----------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------


def _build_centroid_polyline(
    swc_model: SWCModel,
) -> tuple[list[float], list[float], list[float]]:
    """Build polyline coordinates for centroid skeleton edges.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Lists of x, y, z coordinates with None separators for Plotly.
    """
    xs, ys, zs = [], [], []
    for u, v in swc_model.edges:
        xu, yu, zu = swc_model.get_node_xyz(u)
        xv, yv, zv = swc_model.get_node_xyz(v)
        xs.extend([xu, xv, None])
        ys.extend([yu, yv, None])
        zs.extend([zu, zv, None])
    return xs, ys, zs


def _build_node_scatter(
    swc_model: SWCModel,
) -> tuple[list[float], list[float], list[float]]:
    """Build scatter plot coordinates for all nodes.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Lists of x, y, z coordinates for all nodes.
    """
    xs, ys, zs = [], [], []
    for node_id in swc_model.nodes:
        x, y, z = swc_model.get_node_xyz(node_id)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def _build_tag_facecolors(
    frusta: FrustaSet, tag_colors: dict[int, str], default_color: str = "lightblue"
) -> list[str]:
    """Build per-face color list based on frustum tags.

    Parameters
    ----------
    frusta: FrustaSet
        Frusta set with frusta.
    tag_colors: dict[int, str]
        Mapping of tag values to color strings.
    default_color: str
        Default color for tags not in tag_colors.

    Returns
    -------
    list[str]
        Color string for each face in the frusta mesh.
    """
    slices = list(frusta.frustum_face_slices_map().values())
    facecolors = []
    for (start, count), frustum in zip(slices, frusta.frusta):
        color = tag_colors.get(frustum.tag, default_color)
        facecolors.extend([color] * count)
    return facecolors


def _generate_hemisphere_mesh(
    center: tuple[float, float, float],
    radius: float,
    direction: tuple[float, float, float],
    subdivisions: int = 10,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Generate a hemisphere mesh oriented along a direction vector.

    Parameters
    ----------
    center : tuple[float, float, float]
        Center point of the hemisphere (on the flat face).
    radius : float
        Radius of the hemisphere.
    direction : tuple[float, float, float]
        Direction vector pointing from the flat face toward the curved side.
    subdivisions : int
        Number of subdivisions for the hemisphere (default: 10).

    Returns
    -------
    tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]
        (vertices, faces) where vertices are 3D points and faces are triangular indices.
    """
    # Normalize direction vector
    dx, dy, dz = direction
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        dx, dy, dz = 0.0, 0.0, 1.0
    else:
        dx, dy, dz = dx / length, dy / length, dz / length

    # Build orthonormal frame with W = direction
    W = (dx, dy, dz)
    # Pick a vector not parallel to W
    abs_w = (abs(dx), abs(dy), abs(dz))
    tmp = (1.0, 0.0, 0.0) if abs_w[0] <= 0.9 else (0.0, 1.0, 0.0)
    # U = tmp × W
    ux = tmp[1] * W[2] - tmp[2] * W[1]
    uy = tmp[2] * W[0] - tmp[0] * W[2]
    uz = tmp[0] * W[1] - tmp[1] * W[0]
    u_len = math.sqrt(ux * ux + uy * uy + uz * uz)
    if u_len < 1e-12:
        tmp = (0.0, 1.0, 0.0)
        ux = tmp[1] * W[2] - tmp[2] * W[1]
        uy = tmp[2] * W[0] - tmp[0] * W[2]
        uz = tmp[0] * W[1] - tmp[1] * W[0]
        u_len = math.sqrt(ux * ux + uy * uy + uz * uz)
    U = (ux / u_len, uy / u_len, uz / u_len)
    # V = W × U
    vx = W[1] * U[2] - W[2] * U[1]
    vy = W[2] * U[0] - W[0] * U[2]
    vz = W[0] * U[1] - W[1] * U[0]
    V = (vx, vy, vz)

    vertices = []
    faces = []

    # Add center point as first vertex
    vertices.append(center)

    # Generate hemisphere vertices (only phi from 0 to pi/2)
    for i in range(1, subdivisions + 1):
        phi = (math.pi / 2) * (i / subdivisions)  # 0 to pi/2
        for j in range(subdivisions):
            theta = 2 * math.pi * (j / subdivisions)

            # Spherical coordinates: x = r*sin(phi)*cos(theta), y = r*sin(phi)*sin(theta), z = r*cos(phi)
            r_proj = radius * math.sin(phi)
            z_comp = radius * math.cos(phi)

            # Local coordinates in the UVW frame
            local_x = r_proj * math.cos(theta)
            local_y = r_proj * math.sin(theta)
            local_z = z_comp

            # Transform to world coordinates
            px = center[0] + local_x * U[0] + local_y * V[0] + local_z * W[0]
            py = center[1] + local_x * U[1] + local_y * V[1] + local_z * W[1]
            pz = center[2] + local_x * U[2] + local_y * V[2] + local_z * W[2]

            vertices.append((px, py, pz))

    # Generate faces
    # Connect center to first ring
    for j in range(subdivisions):
        j_next = (j + 1) % subdivisions
        faces.append((0, 1 + j, 1 + j_next))

    # Connect rings
    for i in range(subdivisions - 1):
        for j in range(subdivisions):
            j_next = (j + 1) % subdivisions
            current = 1 + i * subdivisions + j
            current_next = 1 + i * subdivisions + j_next
            next_ring = 1 + (i + 1) * subdivisions + j
            next_ring_next = 1 + (i + 1) * subdivisions + j_next

            faces.append((current, next_ring, next_ring_next))
            faces.append((current, next_ring_next, current_next))

    return vertices, faces


def _get_proper_terminals(
    swc_model: SWCModel,
) -> list[int]:
    """Get proper terminal points: nodes with exactly 1 edge, not in reconnections.

    Parameters
    ----------
    swc_model : SWCModel
        The SWC model to query.

    Returns
    -------
    list[int]
        List of node IDs that are proper terminal points.
    """
    reconnections = swc_model.graph.get("reconnections", [])
    reconnection_nodes = set()
    for u, v in reconnections:
        reconnection_nodes.add(u)
        reconnection_nodes.add(v)

    terminals = []
    for node_id in swc_model.nodes:
        # Check if node has exactly 1 edge (degree 1 in undirected view)
        if swc_model.degree(node_id) == 1 and node_id not in reconnection_nodes:
            terminals.append(node_id)

    return terminals


def _build_endcap_meshes(
    swc_model: SWCModel,
    frusta: FrustaSet,
    tag_colors: dict[int, str] | None = None,
    default_color: str = "lightblue",
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int]],
    list[str] | None,
]:
    """Build endcap hemisphere meshes for proper terminal points.

    Parameters
    ----------
    swc_model : SWCModel
        The SWC model.
    frusta : FrustaSet
        The frusta set to get radii from.
    tag_colors : dict[int, str] | None
        Optional tag-to-color mapping.
    default_color : str
        Default color if tag_colors not provided or tag not found.

    Returns
    -------
    tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[str] | None]
        (vertices, faces, facecolors) for all endcaps combined.
    """
    terminals = _get_proper_terminals(swc_model)
    if not terminals:
        return [], [], None

    # Build mapping from node to frustum
    node_to_frustum = {}
    if frusta.edge_uvs is not None:
        for idx, (u, v) in enumerate(frusta.edge_uvs):
            node_to_frustum[u] = (idx, v)  # (frustum_idx, other_node)
            node_to_frustum[v] = (idx, u)

    all_vertices = []
    all_faces = []
    all_facecolors = [] if tag_colors is not None else None

    for terminal_id in terminals:
        if terminal_id not in node_to_frustum:
            continue

        frustum_idx, parent_id = node_to_frustum[terminal_id]
        frustum = frusta.frusta[frustum_idx]

        # Get terminal position and radius
        terminal_xyz = swc_model.get_node_xyz(terminal_id)
        parent_xyz = swc_model.get_node_xyz(parent_id)

        # Determine which end of the frustum corresponds to the terminal
        # Check if terminal is at 'a' or 'b' end of frustum
        dist_to_a = sum((terminal_xyz[i] - frustum.a[i]) ** 2 for i in range(3))
        dist_to_b = sum((terminal_xyz[i] - frustum.b[i]) ** 2 for i in range(3))

        if dist_to_a < dist_to_b:
            # Terminal is at 'a' end
            radius = frustum.ra
        else:
            # Terminal is at 'b' end
            radius = frustum.rb

        # Direction vector: from parent to terminal
        direction = (
            terminal_xyz[0] - parent_xyz[0],
            terminal_xyz[1] - parent_xyz[1],
            terminal_xyz[2] - parent_xyz[2],
        )

        # Generate hemisphere mesh
        vertices, faces = _generate_hemisphere_mesh(
            center=terminal_xyz,
            radius=radius,
            direction=direction,
            subdivisions=10,
        )

        # Offset face indices by current vertex count
        vertex_offset = len(all_vertices)
        offset_faces = [
            (f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset)
            for f in faces
        ]

        all_vertices.extend(vertices)
        all_faces.extend(offset_faces)

        # Add face colors if needed
        if all_facecolors is not None:
            color = (
                tag_colors.get(frustum.tag, default_color)
                if tag_colors
                else default_color
            )
            all_facecolors.extend([color] * len(faces))

    return all_vertices, all_faces, all_facecolors


# ----------------------------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------------------------


def plot_centroid(
    swc_model: SWCModel,
    *,
    marker_size: float = 2.0,
    line_width: float = 2.0,
    show_nodes: bool = True,
    title: str | None = None,
    width: int = 1200,
    height: int = 900,
) -> go.Figure:
    """Plot centroid skeleton from an `SWCModel`.

    Edges are drawn as line segments in 3D using Scatter3d.

    Parameters
    ----------
    width : int
        Figure width in pixels (default: 1200).
    height : int
        Figure height in pixels (default: 900).
    """
    logger = logging.getLogger(__name__)

    xs, ys, zs = _build_centroid_polyline(swc_model)
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
        xn, yn, zn = _build_node_scatter(swc_model)
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
    fig.update_layout(width=width, height=height)
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
    width: int = 1200,
    height: int = 900,
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
        Uniform scale applied to all frustum radii before meshing (1.0 = no change).
    tag_colors: dict[int, str] | None
        Optional mapping {tag: color}. If provided, each frustum is colored
        uniformly according to its tag (fallback to `color` if a tag is missing).
    """
    logger = logging.getLogger(__name__)
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    if tag_colors is not None:
        facecolor = _build_tag_facecolors(fr, tag_colors, color)
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
    fig.update_layout(width=width, height=height)
    logger.info(
        "plot_frusta frusta=%d radius_scale=%s flatshading=%s",
        frusta.n_frusta,
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
    node_color: str = "#1f77b4",
    title: str | None = None,
    width: int = 1200,
    height: int = 900,
) -> go.Figure:
    """Overlay frusta mesh with centroid skeleton from an `SWCModel`.

    Parameters mirror `plot_centroid` and `plot_frusta` with an extra `radius_scale`.
    """
    logger = logging.getLogger(__name__)

    # Build centroid polyline
    xs, ys, zs = _build_centroid_polyline(swc_model)
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
        xn, yn, zn = _build_node_scatter(swc_model)
        nodes = go.Scatter3d(
            x=xn,
            y=yn,
            z=zn,
            mode="markers",
            marker=dict(size=node_size, color=node_color),
            name="nodes",
        )
        traces.append(nodes)

    # Frusta mesh (optionally scaled)
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    if tag_colors is not None:
        facecolor = _build_tag_facecolors(fr, tag_colors, color)
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
    fig.update_layout(width=width, height=height)
    logger.info(
        "plot_frusta_with_centroid edges=%d frusta=%d radius_scale=%s",
        len(list(swc_model.edges)),
        frusta.n_frusta,
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
    width: int = 1200,
    height: int = 900,
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
        slices = list(base.frustum_face_slices_map().values())
        facecolor0: list[str] = []
        for (start, count), frustum in zip(slices, base.frusta):
            c = tag_colors.get(frustum.tag, color)
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
    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus,
        width=width,
        height=height,
    )
    logger.info(
        "plot_frusta_slider frusta=%d scales=%d min=%s max=%s",
        frusta.n_frusta,
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
    plot_endcaps: bool = False,
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
    node_color: str = "#1f77b4",
    # Extra points overlay (as low-res spheres)
    point_set: PointSet | None = None,
    point_size: float = 1.0,
    point_color: str = "#d62728",
    # HTML output options
    output_path: str | None = None,
    auto_open: bool = False,
    width: int = 1200,
    height: int = 900,
    show_axes: bool = True,
) -> go.Figure:
    """Master visualization combining centroid, frusta, slider, and overlay points.

    - If `frusta` is not provided and `swc_model` is, a `FrustaSet` is built from `swc_model`.
    - If `slider=True` and `show_frusta=True`, a Plotly slider controls `radius_scale`.
    - `points` overlays arbitrary xyz positions as small markers.

    Parameters
    ----------
    plot_endcaps : bool
        If True, plot hemisphere endcaps on proper terminal points (nodes with
        exactly 1 edge that are not in reconnections list). Endcaps are oriented
        along the tangent direction (parent to terminal). Default: False.
    output_path : str | None
        If provided, saves the figure to an HTML file at this path.
    auto_open : bool
        If True and output_path is provided, opens the HTML file in browser.
    width : int
        Figure width in pixels (default: 1200).
    height : int
        Figure height in pixels (default: 900).
    show_axes : bool
        If True, shows all axes, grid, and background (default: True).
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
        xs, ys, zs = _build_centroid_polyline(swc_model)
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
            xn, yn, zn = _build_node_scatter(swc_model)
            nodes = go.Scatter3d(
                x=xn,
                y=yn,
                z=zn,
                mode="markers",
                marker=dict(size=node_size, color=node_color),
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
                facecolor0 = _build_tag_facecolors(base_fr, tag_colors, color)
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
            fig.update_layout(
                sliders=sliders,
                updatemenus=updatemenus,
                width=width,
                height=height,
            )
            if not show_axes:
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                    )
                )
            logger.info(
                "plot_model slider=True frusta=%d radius_scale_range=[%s,%s]",
                base_fr.n_frusta,
                min_scale,
                max_scale,
            )

            # Save to HTML file if requested
            if output_path is not None:
                from pathlib import Path
                import webbrowser

                output_file = Path(output_path)
                logger.info("Saving plot to: %s", output_file.absolute())
                fig.write_html(str(output_file), auto_play=False)
                logger.info("Plot saved successfully")

                if auto_open:
                    logger.info("Opening plot in default browser...")
                    webbrowser.open(f"file://{output_file.absolute()}")

            return fig
        else:
            # Static radius scale
            fr = base_fr if radius_scale == 1.0 else base_fr.scaled(radius_scale)
            x, y, z, i, j, k = fr.to_mesh3d_arrays()
            if tag_colors is not None:
                facecolor = _build_tag_facecolors(fr, tag_colors, color)
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

    # Add endcaps if requested
    if plot_endcaps and swc_model is not None and base_fr is not None:
        endcap_verts, endcap_faces, endcap_colors = _build_endcap_meshes(
            swc_model, base_fr, tag_colors, color
        )
        if endcap_verts:
            ex = [v[0] for v in endcap_verts]
            ey = [v[1] for v in endcap_verts]
            ez = [v[2] for v in endcap_verts]
            ei = [f[0] for f in endcap_faces]
            ej = [f[1] for f in endcap_faces]
            ek = [f[2] for f in endcap_faces]

            if endcap_colors is not None:
                endcap_mesh = go.Mesh3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    i=ei,
                    j=ej,
                    k=ek,
                    facecolor=endcap_colors,
                    opacity=opacity,
                    flatshading=flatshading,
                    name="endcaps",
                )
            else:
                endcap_mesh = go.Mesh3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    i=ei,
                    j=ej,
                    k=ek,
                    color=color,
                    opacity=opacity,
                    flatshading=flatshading,
                    name="endcaps",
                )
            traces.insert(1, endcap_mesh)

    fig = go.Figure(data=traces)
    apply_layout(fig, title=title or "Model")
    fig.update_layout(width=width, height=height)
    if not show_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )
    logger.info(
        "plot_model slider=False frusta=%s show_frusta=%s show_centroid=%s",
        base_fr.n_frusta if base_fr is not None else None,
        show_frusta,
        show_centroid,
    )

    # Save to HTML file if requested
    if output_path is not None:
        from pathlib import Path
        import webbrowser

        output_file = Path(output_path)
        logger.info("Saving plot to: %s", output_file.absolute())
        fig.write_html(str(output_file), auto_play=False)
        logger.info("Plot saved successfully")

        if auto_open:
            logger.info("Opening plot in default browser...")
            webbrowser.open(f"file://{output_file.absolute()}")

    return fig


def animate_frusta_timeseries(
    frusta: FrustaSet,
    time_domain: Sequence[float],
    amplitudes: Sequence[Sequence[float]],
    *,
    colorscale: str = "Viridis",
    clim: tuple[float, float] | None = None,
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
    fps: int = 30,
    stride: int = 1,
    title: str | None = None,
    output_path: str | None = None,
    auto_open: bool = False,
):
    """Animate per-frustum values over time with interactive 3D controls.

    Creates a Plotly animation with play/pause controls, time slider, and full
    3D interactivity (rotate, zoom, pan). The animation is saved to an HTML file
    that can be opened in any web browser.

    Parameters
    ----------
    frusta : FrustaSet
        Batched frusta mesh representing the neuron compartments.
    time_domain : Sequence[float]
        Time values for each frame. Length must match the time axis of amplitudes.
    amplitudes : Sequence[Sequence[float]]
        Time series V_i(t) shaped (T, N), where T = len(time_domain) and
        N = frusta.n_frusta. Each time step provides one scalar per frustum.
    colorscale : str
        Plotly colorscale name (default: "Viridis"). Examples: "Viridis", "Plasma",
        "Inferno", "Jet", "RdBu", etc.
    clim : tuple[float, float] | None
        Color limits (vmin, vmax). If None, inferred from amplitudes.
    opacity : float
        Mesh opacity (default: 0.8).
    flatshading : bool
        Enable flat shading on the mesh (default: True).
    radius_scale : float
        Uniform radius scaling applied to frusta before meshing (default: 1.0).
    fps : int
        Frames per second for animation playback (default: 30).
    stride : int
        Temporal downsampling factor - use every `stride` time steps (default: 1).
    title : str | None
        Figure title. If None, defaults to "Frusta Animation".
    output_path : str | None
        Path to save the HTML file. If None, defaults to "frusta_animation.html".
    auto_open : bool
        If True, automatically open the HTML file in the default browser when saving (default: False).

    Returns
    -------
    go.Figure
        The Plotly figure object with animation frames.

    Notes
    -----
    The resulting HTML file contains a fully interactive 3D visualization with:
    - Play/Pause buttons for animation control
    - Time slider to scrub through frames
    - Full 3D rotation, zoom, and pan controls
    - Colorbar showing value mapping

    The file can be shared and opened on any system with a web browser, making
    it highly portable and robust across different OS and display configurations.
    """
    import numpy as np
    import webbrowser
    from pathlib import Path

    logger = logging.getLogger(__name__)

    logger.info(
        "animate_frusta_timeseries: n_frusta=%d time_steps=%d fps=%d stride=%d",
        frusta.n_frusta,
        len(time_domain),
        fps,
        stride,
    )

    # Apply stride to time domain and amplitudes
    time_domain = np.asarray(time_domain)
    amplitudes = np.asarray(amplitudes)

    if stride > 1:
        time_domain = time_domain[::stride]
        amplitudes = amplitudes[::stride]
        logger.info("Applied stride=%d: reduced to %d frames", stride, len(time_domain))

    # Scale frusta if needed
    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    slices = list(fr.frustum_face_slices_map().values())

    # Validate dimensions
    if len(amplitudes) == 0:
        raise ValueError("amplitudes must have at least one time step")
    if len(time_domain) != len(amplitudes):
        raise ValueError(
            f"time_domain length ({len(time_domain)}) must match amplitudes "
            f"time axis length ({len(amplitudes)})"
        )
    T = len(amplitudes)
    N = fr.n_frusta
    if any(len(vt) != N for vt in amplitudes):
        raise ValueError(
            f"each time step must have N values, matching frusta.n_frusta ({N})"
        )

    def faces_intensity(vt: Sequence[float]) -> list[float]:
        """Map per-frustum values to per-face intensity."""
        arr: list[float] = []
        for (start, count), val in zip(slices, vt):
            arr.extend([float(val)] * count)
        return arr

    # Determine color limits
    if clim is None:
        vmin = float(np.min(amplitudes))
        vmax = float(np.max(amplitudes))
        cmin, cmax = vmin, vmax
    else:
        cmin, cmax = clim

    logger.info("Color limits: [%.3f, %.3f]", cmin, cmax)

    # Create initial mesh
    intensity0 = faces_intensity(amplitudes[0])
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
        colorbar=dict(title="Amplitude"),
    )

    # Create animation frames
    logger.info("Creating %d animation frames...", T)
    frames = []
    for t, vt in enumerate(amplitudes):
        intens = faces_intensity(vt)
        frames.append(
            go.Frame(
                name=f"time={time_domain[t]:.3f}",
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
        if (t + 1) % 10 == 0 or (t + 1) == T:
            logger.debug("Created %d/%d frames", t + 1, T)

    # Animation controls
    frame_duration = int(1000 / max(1, fps))
    slider_steps = [
        {
            "label": f"{time_domain[t]:.3f}",
            "method": "animate",
            "args": [
                [f"time={time_domain[t]:.3f}"],
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
            "currentvalue": {"prefix": "t = ", "visible": True},
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

    # Create figure
    fig = go.Figure(data=[mesh], frames=frames)
    apply_layout(fig, title=title or "Frusta Animation")
    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus,
        # Increase figure size for better visibility in HTML
        width=1200,
        height=900,
    )

    logger.info("Animation figure created with %d frames at %d fps", T, fps)

    # Save to HTML file
    if output_path is None:
        output_path = "frusta_animation.html"

    output_file = Path(output_path)
    logger.info("Saving animation to: %s", output_file.absolute())
    # auto_play=False prevents animation from starting on page load
    fig.write_html(str(output_file), auto_play=False)
    logger.info("Animation saved successfully")

    # Auto-open in browser
    if auto_open:
        logger.info("Opening animation in default browser...")
        webbrowser.open(f"file://{output_file.absolute()}")

    return fig


def plot_points(
    point_set: PointSet,
    *,
    color: str = "#ff7f0e",
    opacity: float = 1.0,
    size_scale: float = 1.0,
    title: str | None = None,
    width: int = 1200,
    height: int = 900,
) -> go.Figure:
    """Plot a PointSet as a collection of small spheres.

    Parameters
    ----------
    point_set: PointSet
        Point set to visualize.
    color: str
        Color for all spheres.
    opacity: float
        Sphere opacity.
    size_scale: float
        Uniform scale applied to sphere radii (1.0 = no change).
    title: str | None
        Figure title.

    Returns
    -------
    go.Figure
        Plotly figure with Mesh3d trace.
    """
    logger = logging.getLogger(__name__)
    ps = point_set if size_scale == 1.0 else point_set.scaled(size_scale)
    x, y, z, i, j, k = ps.to_mesh3d_arrays()

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        name="points",
    )

    fig = go.Figure(data=[mesh])
    apply_layout(fig, title=title or "Point Set")
    fig.update_layout(width=width, height=height)
    logger.info(
        "plot_points count=%d size_scale=%s",
        len(point_set.points),
        size_scale,
    )
    return fig
