"""Geometry utilities for segment/frustum mesh generation.

- Segment: oriented frustum defined by two points with radii
- frustum_mesh: build vertices/faces for a single frustum
- batch_frusta: combine multiple frusta into one mesh

Implementation is pure-Python (standard library math), returning lists
of vertices and triangular faces suitable for Plotly Mesh3d or other
renderers after light conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Any, Optional, Union
import os
import io
import math

# Types
Point3 = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


# --------------------------------------------------------------------------------------
# Vector helpers (pure Python)
# --------------------------------------------------------------------------------------


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def v_unit(a: Vec3, eps: float = 1e-12) -> Vec3:
    n = v_norm(a)
    if n < eps:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


# --------------------------------------------------------------------------------------
# Core structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """Oriented frustum segment between endpoints `a` and `b`.

    Attributes
    ----------
    a, b: Point3
        Endpoints in model/world coordinates.
    ra, rb: float
        Radii at `a` and `b`.
    """

    a: Point3
    b: Point3
    ra: float
    rb: float

    def vector(self) -> Vec3:
        return v_sub(self.b, self.a)

    def length(self) -> float:
        return v_norm(self.vector())

    def midpoint(self) -> Point3:
        return (
            self.a[0] * 0.5 + self.b[0] * 0.5,
            self.a[1] * 0.5 + self.b[1] * 0.5,
            self.a[2] * 0.5 + self.b[2] * 0.5,
        )


# --------------------------------------------------------------------------------------
# Frames and rings
# --------------------------------------------------------------------------------------


def _orthonormal_frame(z_axis: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    """Return (U, V, W) forming a right-handed orthonormal basis with W along z_axis.

    Handles near-colinearity by choosing a stable temporary axis.
    """
    W = v_unit(z_axis)
    # Fallback if zero vector
    if W == (0.0, 0.0, 0.0):
        W = (0.0, 0.0, 1.0)

    # Pick a vector not parallel to W
    abs_w = tuple(abs(c) for c in W)
    tmp = (1.0, 0.0, 0.0) if abs_w[0] <= 0.9 else (0.0, 1.0, 0.0)
    U = v_cross(tmp, W)
    U = v_unit(U)
    # If still degenerate (happens when tmp ~ W), switch tmp
    if U == (0.0, 0.0, 0.0):
        tmp = (0.0, 1.0, 0.0)
        U = v_unit(v_cross(tmp, W))
    V = v_cross(W, U)
    return U, V, W


def _circle_ring(
    center: Point3, radius: float, U: Vec3, V: Vec3, sides: int
) -> List[Point3]:
    pts: List[Point3] = []
    for k in range(sides):
        theta = 2.0 * math.pi * (k / sides)
        c = math.cos(theta)
        s = math.sin(theta)
        offset = v_add(v_mul(U, radius * c), v_mul(V, radius * s))
        pts.append(v_add(center, offset))
    return pts


# --------------------------------------------------------------------------------------
# Mesh generation
# --------------------------------------------------------------------------------------


def frustum_mesh(
    seg: Segment, *, sides: int = 16, end_caps: bool = False
) -> Tuple[List[Point3], List[Face]]:
    """Generate a frustum mesh for a single `Segment`.

    Returns
    -------
    (vertices, faces):
        - vertices: List[Point3]
        - faces: List[Face], each = (i, j, k) indexing into `vertices`
    """
    # Local frame
    U, V, W = _orthonormal_frame(seg.vector())

    ring_a = _circle_ring(seg.a, seg.ra, U, V, sides)
    ring_b = _circle_ring(seg.b, seg.rb, U, V, sides)

    vertices: List[Point3] = []
    vertices.extend(ring_a)
    vertices.extend(ring_b)

    faces: List[Face] = []

    # Side faces (two triangles per quad)
    for i in range(sides):
        a0 = i
        a1 = (i + 1) % sides
        b0 = i + sides
        b1 = ((i + 1) % sides) + sides
        faces.append((a0, b0, b1))
        faces.append((a0, b1, a1))

    # Optional end caps
    if end_caps and seg.ra > 0.0:
        ca = len(vertices)
        vertices.append(seg.a)
        for i in range(sides):
            a0 = i
            a1 = (i + 1) % sides
            # Wind towards center for cap
            faces.append((ca, a1, a0))

    if end_caps and seg.rb > 0.0:
        cb = len(vertices)
        vertices.append(seg.b)
        for i in range(sides):
            b0 = i + sides
            b1 = ((i + 1) % sides) + sides
            faces.append((cb, b0, b1))

    return vertices, faces


def batch_frusta(
    segments: Iterable[Segment], *, sides: int = 16, end_caps: bool = False
) -> Tuple[List[Point3], List[Face]]:
    """Batch multiple frusta into a single mesh.

    Returns a concatenated list of `vertices` and `faces` with the proper index offsets.
    """
    all_vertices: List[Point3] = []
    all_faces: List[Face] = []
    offset = 0

    for seg in segments:
        v, f = frustum_mesh(seg, sides=sides, end_caps=end_caps)
        all_vertices.extend(v)
        # Re-index faces
        all_faces.extend([(a + offset, b + offset, c + offset) for (a, b, c) in f])
        offset += len(v)

    return all_vertices, all_faces


# --------------------------------------------------------------------------------------
# Spheres for point sets
# --------------------------------------------------------------------------------------


def sphere_mesh(
    center: Point3, radius: float, *, stacks: int = 6, slices: int = 12
) -> Tuple[List[Point3], List[Face]]:
    """Generate a low-res UV sphere mesh at `center` with given `radius`.

    Parameters
    ----------
    stacks: int
        Number of latitudinal divisions (>= 2).
    slices: int
        Number of longitudinal divisions (>= 3).
    """
    stacks = max(2, int(stacks))
    slices = max(3, int(slices))

    verts: List[Point3] = []
    faces: List[Face] = []

    # Generate vertices (excluding poles initially)
    for i in range(1, stacks):
        theta = math.pi * (i / stacks)  # (0, pi)
        st = math.sin(theta)
        ct = math.cos(theta)
        for j in range(slices):
            phi = 2.0 * math.pi * (j / slices)
            sp = math.sin(phi)
            cp = math.cos(phi)
            x = center[0] + radius * st * cp
            y = center[1] + radius * st * sp
            z = center[2] + radius * ct
            verts.append((x, y, z))

    # Add poles
    north_idx = len(verts)
    verts.append((center[0], center[1], center[2] + radius))
    south_idx = len(verts)
    verts.append((center[0], center[1], center[2] - radius))

    # Index helper for ring vertices
    def vid(i: int, j: int) -> int:
        # i in [0, stacks-2], j in [0, slices-1]
        return i * slices + (j % slices)

    # Faces between rings
    for i in range(stacks - 2):
        for j in range(slices):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j)
            d = vid(i + 1, j + 1)
            faces.append((a, c, d))
            faces.append((a, d, b))

    # Triangles to poles
    # Top ring (i = 0) connects to north pole
    for j in range(slices):
        a = vid(0, j)
        b = vid(0, j + 1)
        faces.append((north_idx, a, b))
    # Bottom ring (i = stacks-2) connects to south pole
    base = stacks - 2
    for j in range(slices):
        a = vid(base, j)
        b = vid(base, j + 1)
        faces.append((south_idx, b, a))

    return verts, faces


def batch_spheres(
    points: Iterable[Point3], *, radius: float = 1.0, stacks: int = 6, slices: int = 12
) -> Tuple[List[Point3], List[Face]]:
    """Batch multiple spheres into a single mesh.

    Returns concatenated `vertices` and reindexed `faces`.
    """
    all_vertices: List[Point3] = []
    all_faces: List[Face] = []
    offset = 0

    for p in points:
        v, f = sphere_mesh(p, radius, stacks=stacks, slices=slices)
        all_vertices.extend(v)
        all_faces.extend([(a + offset, b + offset, c + offset) for (a, b, c) in f])
        offset += len(v)

    return all_vertices, all_faces


@dataclass(frozen=True)
class PointSet:
    """A batched mesh of small spheres placed at given 3D points."""

    vertices: List[Point3]
    faces: List[Face]
    points: List[Point3]
    base_radius: float
    stacks: int
    slices: int

    @classmethod
    def from_points(
        cls,
        points: Sequence[Point3],
        *,
        base_radius: float = 1.0,
        stacks: int = 6,
        slices: int = 12,
    ) -> "PointSet":
        verts, faces = batch_spheres(
            points, radius=base_radius, stacks=stacks, slices=slices
        )
        return cls(
            vertices=verts,
            faces=faces,
            points=list(points),
            base_radius=base_radius,
            stacks=stacks,
            slices=slices,
        )

    @classmethod
    def from_txt(
        cls,
        source: Union[str, os.PathLike, Iterable[str], io.TextIOBase],
        *,
        base_radius: float = 1.0,
        stacks: int = 6,
        slices: int = 12,
        allow_extra_columns: bool = True,
    ) -> "PointSet":
        """Load a simple text format with `x y z` coordinates per non-empty line.

        - Lines beginning with `#` or blank lines are ignored.
        - If `allow_extra_columns=True`, extra columns after the first three are ignored.
        - Raises `ValueError` on malformed lines.
        """

        # Normalize to an iterator of lines
        lines: Iterable[str]
        if hasattr(source, "read"):
            # file-like or IO stream; iterating yields lines
            lines = source  # type: ignore[assignment]
        elif isinstance(source, (str, os.PathLike)):
            # path or text
            p = str(source)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().splitlines()
                lines = content
            else:
                lines = str(source).splitlines()
        else:
            lines = source

        pts: List[Point3] = []
        for idx, raw in enumerate(lines, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Line {idx}: expected at least 3 columns for x y z, got {len(parts)}"
                )
            if not allow_extra_columns and len(parts) != 3:
                raise ValueError(
                    f"Line {idx}: expected exactly 3 columns for x y z, got {len(parts)}"
                )
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except Exception as e:
                raise ValueError(f"Line {idx}: could not parse floats: {e}")
            pts.append((x, y, z))

        return cls.from_points(
            pts, base_radius=base_radius, stacks=stacks, slices=slices
        )

    def to_mesh3d_arrays(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
        x = [p[0] for p in self.vertices]
        y = [p[1] for p in self.vertices]
        z = [p[2] for p in self.vertices]
        i = [f[0] for f in self.faces]
        j = [f[1] for f in self.faces]
        k = [f[2] for f in self.faces]
        return x, y, z, i, j, k

    def scaled(self, radius_scale: float) -> "PointSet":
        """Return a new `PointSet` with all sphere radii scaled by `radius_scale`."""
        if radius_scale == 1.0:
            return self
        r = self.base_radius * radius_scale
        verts, faces = batch_spheres(
            self.points, radius=r, stacks=self.stacks, slices=self.slices
        )
        return PointSet(
            vertices=verts,
            faces=faces,
            points=self.points,
            base_radius=self.base_radius,
            stacks=self.stacks,
            slices=self.slices,
        )


# --------------------------------------------------------------------------------------
# Frusta set derived from a GeneralModel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrustaSet:
    """A batched frusta mesh derived from a `GeneralModel`.

    Attributes
    ----------
    vertices: List[Point3]
        Concatenated vertices for all frusta.
    faces: List[Face]
        Triangular faces indexing into `vertices`.
    sides: int
        Circumferential resolution used per frustum.
    end_caps: bool
        Whether end caps were included during construction.
    segment_count: int
        Number of segments used (one per graph edge).
    edge_count: int
        Alias for `segment_count` for clarity.
    """

    vertices: List[Point3]
    faces: List[Face]
    sides: int
    end_caps: bool
    segment_count: int
    edge_count: int
    segments: List[Segment]

    @classmethod
    def from_general_model(
        cls,
        gm: Any,
        *,
        sides: int = 16,
        end_caps: bool = False,
    ) -> "FrustaSet":
        """Build a `FrustaSet` by converting each undirected edge into a `Segment`.

        Expects nodes to have attributes `x, y, z, r`.
        """
        segments: List[Segment] = []
        for u, v in gm.edges:
            xu, yu, zu = gm.nodes[u]["x"], gm.nodes[u]["y"], gm.nodes[u]["z"]
            xv, yv, zv = gm.nodes[v]["x"], gm.nodes[v]["y"], gm.nodes[v]["z"]
            ru, rv = float(gm.nodes[u]["r"]), float(gm.nodes[v]["r"])
            segments.append(Segment(a=(xu, yu, zu), b=(xv, yv, zv), ra=ru, rb=rv))

        vertices, faces = batch_frusta(segments, sides=sides, end_caps=end_caps)
        return cls(
            vertices=vertices,
            faces=faces,
            sides=sides,
            end_caps=end_caps,
            segment_count=len(segments),
            edge_count=len(segments),
            segments=segments,
        )

    def to_mesh3d_arrays(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
        """Return Plotly Mesh3d arrays: x, y, z, i, j, k."""
        x = [p[0] for p in self.vertices]
        y = [p[1] for p in self.vertices]
        z = [p[2] for p in self.vertices]
        i = [f[0] for f in self.faces]
        j = [f[1] for f in self.faces]
        k = [f[2] for f in self.faces]
        return x, y, z, i, j, k

    def scaled(self, radius_scale: float) -> "FrustaSet":
        """Return a new FrustaSet with all segment radii scaled by `radius_scale`.

        This rebuilds vertices/faces from the stored `segments` list.
        """
        if radius_scale == 1.0:
            return self
        scaled_segments = [
            Segment(a=s.a, b=s.b, ra=s.ra * radius_scale, rb=s.rb * radius_scale)
            for s in self.segments
        ]
        vertices, faces = batch_frusta(
            scaled_segments, sides=self.sides, end_caps=self.end_caps
        )
        return FrustaSet(
            vertices=vertices,
            faces=faces,
            sides=self.sides,
            end_caps=self.end_caps,
            segment_count=self.segment_count,
            edge_count=self.edge_count,
            segments=scaled_segments,
        )


__all__ = [
    "Segment",
    "frustum_mesh",
    "batch_frusta",
    "sphere_mesh",
    "batch_spheres",
    "PointSet",
    "FrustaSet",
]
