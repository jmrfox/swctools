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
from typing import Iterable, List, Sequence, Tuple, Any, Optional, Union, Mapping
import os
import io
import math
import numpy as np
import logging

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

    def scale(self, scalar: float) -> "Segment":
        """Return a new `Segment` uniformly scaled by `scalar` (positions and radii)."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("scalar must be a number")
        ax, ay, az = self.a
        bx, by, bz = self.b
        return Segment(
            a=(ax * scalar, ay * scalar, az * scalar),
            b=(bx * scalar, by * scalar, bz * scalar),
            ra=self.ra * scalar,
            rb=self.rb * scalar,
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

    logger = logging.getLogger(__name__)
    logger.debug(
        "frustum_mesh sides=%d end_caps=%s verts=%d faces=%d",
        sides,
        end_caps,
        len(vertices),
        len(faces),
    )
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

    logger = logging.getLogger(__name__)
    logger.info(
        "batch_frusta count=%d sides=%d end_caps=%s verts=%d faces=%d",
        len(list(segments)) if isinstance(segments, list) else -1,
        sides,
        end_caps,
        len(all_vertices),
        len(all_faces),
    )
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

    logger = logging.getLogger(__name__)
    logger.debug(
        "sphere_mesh stacks=%d slices=%d verts=%d faces=%d",
        stacks,
        slices,
        len(verts),
        len(faces),
    )
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

    logger = logging.getLogger(__name__)
    logger.info(
        "batch_spheres count=%d stacks=%d slices=%d verts=%d faces=%d",
        len(list(points)) if isinstance(points, list) else -1,
        stacks,
        slices,
        len(all_vertices),
        len(all_faces),
    )
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
        """Build a batched low-res spheres mesh from a list of 3D points.

        Parameters
        ----------
        points: sequence of (x, y, z)
            Sphere centers.
        base_radius: float
            Sphere radius used when building the mesh (scaled later via `scaled()`).
        stacks, slices: int
            Sphere tessellation parameters (>=2 and >=3 respectively).
        """
        logger = logging.getLogger(__name__)
        verts, faces = batch_spheres(
            points, radius=base_radius, stacks=stacks, slices=slices
        )
        logger.info(
            "PointSet.from_points n=%d base_radius=%s stacks=%d slices=%d",
            len(points),
            base_radius,
            stacks,
            slices,
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
    def from_txt_file(
        cls,
        path: Union[str, os.PathLike],
        *,
        base_radius: float = 1.0,
        stacks: int = 6,
        slices: int = 12,
        comments: str = "#",
    ) -> "PointSet":
        logger = logging.getLogger(__name__)
        try:
            arr = np.loadtxt(path, comments=comments, usecols=(0, 1, 2))
        except Exception as e:
            logger.error("PointSet.from_txt_file failed: %s", e)
            return None
        if getattr(arr, "ndim", 1) == 1:
            arr = arr.reshape(1, 3)
        pts = [tuple(map(float, row)) for row in arr]
        logger.info("PointSet.from_txt_file path=%s n=%d", os.fspath(path), len(pts))
        return cls.from_points(
            pts, base_radius=base_radius, stacks=stacks, slices=slices
        )

    def to_mesh3d_arrays(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
        """Return Plotly `Mesh3d` arrays `(x, y, z, i, j, k)` for this point set."""
        x = [p[0] for p in self.vertices]
        y = [p[1] for p in self.vertices]
        z = [p[2] for p in self.vertices]
        i = [f[0] for f in self.faces]
        j = [f[1] for f in self.faces]
        k = [f[2] for f in self.faces]
        return x, y, z, i, j, k

    def to_txt_file(self, path: Union[str, os.PathLike]) -> None:
        arr = np.asarray(self.points, dtype=float)
        try:
            np.savetxt(path, arr, fmt="%.6f", delimiter=" ")
        except Exception as e:
            logging.getLogger(__name__).error("PointSet.to_txt_file failed: %s", e)

    def scaled(self, radius_scale: float) -> "PointSet":
        """Return a new `PointSet` with all sphere radii scaled by `radius_scale`."""
        if radius_scale == 1.0:
            return self
        r = self.base_radius * radius_scale
        logger = logging.getLogger(__name__)
        verts, faces = batch_spheres(
            self.points, radius=r, stacks=self.stacks, slices=self.slices
        )
        logger.info("PointSet.scaled radius_scale=%s", radius_scale)
        return PointSet(
            vertices=verts,
            faces=faces,
            points=self.points,
            base_radius=self.base_radius,
            stacks=self.stacks,
            slices=self.slices,
        )

    def scale(self, scalar: float) -> "PointSet":
        """Return a new `PointSet` with coordinates and radii scaled by `scalar`."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("scalar must be a number")
        if scalar == 1.0:
            return self
        new_points = [
            (p[0] * scalar, p[1] * scalar, p[2] * scalar) for p in self.points
        ]
        new_radius = self.base_radius * scalar
        logger = logging.getLogger(__name__)
        verts, faces = batch_spheres(
            new_points, radius=new_radius, stacks=self.stacks, slices=self.slices
        )
        logger.info("PointSet.scale scalar=%s", scalar)
        return PointSet(
            vertices=verts,
            faces=faces,
            points=new_points,
            base_radius=new_radius,
            stacks=self.stacks,
            slices=self.slices,
        )

    # --------------------------------------------------------------------------------------
    # Frusta set derived from a SWCModel
    # --------------------------------------------------------------------------------------
    def project_onto_frusta(
        self, frusta: "FrustaSet", include_end_caps: Optional[bool] = None
    ) -> "PointSet":
        """Project each point to the nearest surface of the nearest frustum.

        Parameters
        ----------
        frusta: FrustaSet
            Set of oriented frusta (as `Segment`s) to project onto.
        include_end_caps: Optional[bool]
            If None (default), follow `frusta.end_caps`. If True/False, explicitly
            include or ignore projections to the circular end caps.

        Returns
        -------
        PointSet
            A new `PointSet` whose `points` have been moved onto the closest
            surface points of the closest frusta; sphere mesh is rebuilt.

        Notes
        -----
        For each input point, the algorithm iterates all frusta segments and
        evaluates the squared distance to:
        - The lateral surface: project the point to the segment axis (clamped
          t in [0,1]), interpolate radius r(t), then move along the radial
          direction to the mantle.
        - The end caps (optional): orthogonal distance to each cap plane; if
          the projected point falls outside the disk, distance to the rim is used.
        Degenerate segments (zero length) are treated as a sphere of radius
        max(ra, rb) centered at the endpoint.
        Complexity is O(N_points × N_segments), implemented in pure Python.
        """
        # Resolve whether to include end-cap projections (explicit flag overrides frusta setting)
        logger = logging.getLogger(__name__)
        use_caps = frusta.end_caps if include_end_caps is None else include_end_caps
        eps = 1e-12
        # Accumulate the new, projected points in order
        new_points: List[Point3] = []
        logger.info(
            "project_onto_frusta points=%d segments=%d use_caps=%s",
            len(self.points),
            frusta.segment_count,
            use_caps,
        )
        # For each input point, search all frusta and keep the closest surface point
        for idx, p in enumerate(self.points):
            best_q: Optional[Point3] = None
            best_d2 = float("inf")
            # Check every frustum segment
            for s in frusta.segments:
                a = s.a
                b = s.b
                ra = s.ra
                rb = s.rb
                d = v_sub(b, a)
                l2 = v_dot(d, d)
                # Degenerate segment: treat as a sphere of radius max(ra, rb) at endpoint a
                if l2 < eps:
                    R = max(ra, rb)
                    ap = v_sub(p, a)
                    n = v_norm(ap)
                    if n < eps:
                        U, V, W = _orthonormal_frame((1.0, 0.0, 0.0))
                        q = v_add(a, v_mul(U, R))
                        d2 = R * R
                    else:
                        u = v_mul(ap, 1.0 / n)
                        q = v_add(a, v_mul(u, R))
                        d2 = (n - R) * (n - R)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_q = q
                    continue
                # Project point onto segment axis and clamp to [0, 1]
                t = v_dot(v_sub(p, a), d) / l2
                if t < 0.0:
                    t_clamped = 0.0
                elif t > 1.0:
                    t_clamped = 1.0
                else:
                    t_clamped = t
                # Local orthonormal frame with axis W along the segment direction
                U, V, W = _orthonormal_frame(d)
                c = v_add(a, v_mul(d, t_clamped))
                nvec = v_sub(p, c)
                nr = v_norm(nvec)
                rt = ra + (rb - ra) * t_clamped
                if nr < eps:
                    normal_dir = U
                else:
                    normal_dir = v_mul(nvec, 1.0 / nr)
                # Nearest point on the lateral surface (mantle)
                q_lat = v_add(c, v_mul(normal_dir, rt))
                dl = nr - rt
                d2_lat = dl * dl
                if d2_lat < best_d2:
                    best_d2 = d2_lat
                    best_q = q_lat
                # Optionally consider projection to end caps (disks) and their rims
                if use_caps:
                    # Cap at endpoint a
                    sa = v_dot(v_sub(p, a), W)
                    pa = v_sub(v_sub(p, a), v_mul(W, sa))
                    ra_len = v_norm(pa)
                    if ra_len <= ra + eps:
                        q_ca = v_add(a, pa)
                        d2_ca = sa * sa
                    else:
                        if ra_len < eps:
                            pa_dir = U
                        else:
                            pa_dir = v_mul(pa, 1.0 / ra_len)
                        rim_a = v_add(a, v_mul(pa_dir, ra))
                        diff_a = ra_len - ra
                        d2_ca = sa * sa + diff_a * diff_a
                        q_ca = rim_a
                    if d2_ca < best_d2:
                        best_d2 = d2_ca
                        best_q = q_ca
                    # Cap at endpoint b
                    sb = v_dot(v_sub(p, b), W)
                    pb = v_sub(v_sub(p, b), v_mul(W, sb))
                    rb_len = v_norm(pb)
                    if rb_len <= rb + eps:
                        q_cb = v_add(b, pb)
                        d2_cb = sb * sb
                    else:
                        if rb_len < eps:
                            pb_dir = U
                        else:
                            pb_dir = v_mul(pb, 1.0 / rb_len)
                        rim_b = v_add(b, v_mul(pb_dir, rb))
                        diff_b = rb_len - rb
                        d2_cb = sb * sb + diff_b * diff_b
                        q_cb = rim_b
                    if d2_cb < best_d2:
                        best_d2 = d2_cb
                        best_q = q_cb
            # Accept the best (closest) candidate for this point
            if best_q is not None:
                moved = v_norm(v_sub(best_q, p))
                logger.debug(
                    "project_onto_frusta point=%d moved=%.6f on_surface=True",
                    idx,
                    moved,
                )
                new_points.append(best_q)
            else:
                logger.debug(
                    "project_onto_frusta point=%d moved=0.000000 on_surface=False",
                    idx,
                )
                new_points.append(p)
        # Rebuild sphere mesh centered at the moved points
        verts, faces = batch_spheres(
            new_points, radius=self.base_radius, stacks=self.stacks, slices=self.slices
        )
        logger.info("project_onto_frusta done moved_points=%d", len(new_points))
        return PointSet(
            vertices=verts,
            faces=faces,
            points=new_points,
            base_radius=self.base_radius,
            stacks=self.stacks,
            slices=self.slices,
        )


@dataclass(frozen=True)
class FrustaSet:
    """A batched frusta mesh derived from a `SWCModel`.

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
    segments: List[Segment]
        The segments used to construct the frusta.
    edge_uvs: Optional[List[Tuple[int, int]]]
        Optional labels preserving which (u, v) edge generated each segment, in the same order.
    """

    vertices: List[Point3]
    faces: List[Face]
    sides: int
    end_caps: bool
    segment_count: int
    edge_count: int
    segments: List[Segment]
    edge_uvs: Optional[List[Tuple[int, int]]] = None

    @classmethod
    def from_swc_model(
        cls,
        model: SWCModel,
        *,
        sides: int = 16,
        end_caps: bool = False,
    ) -> "FrustaSet":
        """Build a `FrustaSet` by converting each undirected edge into a `Segment`.

        Expects nodes to have attributes `x, y, z, r`.
        """
        segments: List[Segment] = []
        edge_uvs: List[Tuple[int, int]] = []
        for u, v in model.edges:
            xu, yu, zu = model.nodes[u]["x"], model.nodes[u]["y"], model.nodes[u]["z"]
            xv, yv, zv = model.nodes[v]["x"], model.nodes[v]["y"], model.nodes[v]["z"]
            ru, rv = float(model.nodes[u]["r"]), float(model.nodes[v]["r"])
            segments.append(Segment(a=(xu, yu, zu), b=(xv, yv, zv), ra=ru, rb=rv))
            edge_uvs.append((int(u), int(v)))

        logger = logging.getLogger(__name__)
        vertices, faces = batch_frusta(segments, sides=sides, end_caps=end_caps)
        logger.info(
            "FrustaSet.from_swc_model edges=%d sides=%d end_caps=%s",
            len(segments),
            sides,
            end_caps,
        )
        return cls(
            vertices=vertices,
            faces=faces,
            sides=sides,
            end_caps=end_caps,
            segment_count=len(segments),
            edge_count=len(segments),
            segments=segments,
            edge_uvs=edge_uvs,
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
        logger = logging.getLogger(__name__)
        scaled_segments = [
            Segment(a=s.a, b=s.b, ra=s.ra * radius_scale, rb=s.rb * radius_scale)
            for s in self.segments
        ]
        vertices, faces = batch_frusta(
            scaled_segments, sides=self.sides, end_caps=self.end_caps
        )
        logger.info("FrustaSet.scaled radius_scale=%s", radius_scale)
        return FrustaSet(
            vertices=vertices,
            faces=faces,
            sides=self.sides,
            end_caps=self.end_caps,
            segment_count=self.segment_count,
            edge_count=self.edge_count,
            segments=scaled_segments,
            edge_uvs=self.edge_uvs[:] if self.edge_uvs is not None else None,
        )

    def scale(self, scalar: float) -> "FrustaSet":
        """Return a new `FrustaSet` with coordinates and radii scaled by `scalar`."""
        logger = logging.getLogger(__name__)
        if not isinstance(scalar, (int, float)):
            raise TypeError("scalar must be a number")
        if scalar == 1.0:
            return self
        scaled_segments = [
            Segment(
                a=(s.a[0] * scalar, s.a[1] * scalar, s.a[2] * scalar),
                b=(s.b[0] * scalar, s.b[1] * scalar, s.b[2] * scalar),
                ra=s.ra * scalar,
                rb=s.rb * scalar,
            )
            for s in self.segments
        ]
        vertices, faces = batch_frusta(
            scaled_segments, sides=self.sides, end_caps=self.end_caps
        )
        logger.info("FrustaSet.scale scalar=%s", scalar)
        return FrustaSet(
            vertices=vertices,
            faces=faces,
            sides=self.sides,
            end_caps=self.end_caps,
            segment_count=self.segment_count,
            edge_count=self.edge_count,
            segments=scaled_segments,
            edge_uvs=self.edge_uvs[:] if self.edge_uvs is not None else None,
        )

    # ----------------------------------------------------------------------------------
    # Segment ordering utilities
    # ----------------------------------------------------------------------------------
    def print_segment_order(self) -> None:
        """Print the current segment ordering; uses `(u, v)` labels if available."""
        if self.edge_uvs is not None:
            for idx, uv in enumerate(self.edge_uvs):
                print(f"{idx}: {uv}")
        else:
            for idx, s in enumerate(self.segments):
                a = f"({s.a[0]:.3f},{s.a[1]:.3f},{s.a[2]:.3f})"
                b = f"({s.b[0]:.3f},{s.b[1]:.3f},{s.b[2]:.3f})"
                print(f"{idx}: a={a} -> b={b}")

    def reordered(
        self,
        new_order: Sequence[int] | None = None,
        *,
        label_remap: Optional[Mapping[Tuple[int, int], int]] = None,
    ) -> "FrustaSet":
        """Return a new set with segments reordered by index or (u, v) label mapping."""
        n = self.segment_count
        if label_remap is not None:
            if self.edge_uvs is None:
                raise ValueError("label_remap requires edge_uvs on this FrustaSet")
            if len(label_remap) != n:
                raise ValueError(
                    "label_remap must assign every existing edge to a unique index"
                )
            # Build inverse permutation: new_index -> old_index
            inv: List[Optional[int]] = [None] * n
            for old_idx, uv in enumerate(self.edge_uvs):
                if uv not in label_remap:
                    raise ValueError(f"edge label {uv} missing from label_remap")
                new_idx = label_remap[uv]
                if new_idx < 0 or new_idx >= n or inv[new_idx] is not None:
                    raise ValueError("label_remap must be a bijection over 0..N-1")
                inv[new_idx] = old_idx
            assert all(i is not None for i in inv)
            new_order = [int(i) for i in inv]  # type: ignore

        logger = logging.getLogger(__name__)
        if new_order is None:
            raise ValueError("Provide either new_order or label_remap")

        if len(new_order) != n or sorted(new_order) != list(range(n)):
            raise ValueError("new_order must be a permutation of range(N)")

        # Reorder segments and optional labels, rebuild mesh
        segs_new = [self.segments[i] for i in new_order]
        labels_new = (
            [self.edge_uvs[i] for i in new_order] if self.edge_uvs is not None else None
        )
        vertices, faces = batch_frusta(
            segs_new, sides=self.sides, end_caps=self.end_caps
        )
        logger.info("FrustaSet.reordered n=%d", n)
        return FrustaSet(
            vertices=vertices,
            faces=faces,
            sides=self.sides,
            end_caps=self.end_caps,
            segment_count=n,
            edge_count=n,
            segments=segs_new,
            edge_uvs=labels_new,
        )

    def per_segment_face_slices(self) -> List[Tuple[int, int]]:
        """Return (start, count) face spans for each segment in current mesh order."""
        slices: List[Tuple[int, int]] = []
        face_offset = 0
        for s in self.segments:
            _, f = frustum_mesh(s, sides=self.sides, end_caps=self.end_caps)
            count = len(f)
            slices.append((face_offset, count))
            face_offset += count
        if face_offset != len(self.faces):
            logger = logging.getLogger(__name__)
            logger.debug(
                "per_segment_face_slices mismatch faces_computed=%d faces_total=%d",
                face_offset,
                len(self.faces),
            )
            return slices
        return slices


__all__ = [
    "Segment",
    "frustum_mesh",
    "batch_frusta",
    "sphere_mesh",
    "batch_spheres",
    "PointSet",
    "FrustaSet",
]
