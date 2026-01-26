"""SWC graph data model.

SWCModel stores SWC morphology as an undirected graph where nodes are SWC points,
and the original directed parent -> child relationships are preserved in an
internal parent map.

IMPORTANT: SWCModel represents valid SWC directed tree structures only. If you need
to work with cyclic graphs (e.g., after applying reconnections), use the
`make_cycle_connections()` method which returns a standard nx.Graph.

Notes
-----
- Use `SWCModel` for topology and attribute management of parsed SWC trees.
- Use `SWCModel.make_cycle_connections()` to merge reconnection pairs; this returns
  an nx.Graph (not SWCModel) since the result may contain cycles.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Any
import os
import networkx as nx
import numpy as np
import logging

from .io import SWCRecord, SWCParseResult, parse_swc


# ----------------------------------------------------------------------------------------------
# Graph attribute computation
# ----------------------------------------------------------------------------------------------
def _graph_attributes(G: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    """Compute generic attributes for a graph.

    Returns a dictionary including:
    - graph_type: "DiGraph" or "Graph"
    - directed: bool
    - nodes: int
    - edges: int
    - components: int (computed on the undirected view)
    - cycles: int (cyclomatic number on undirected view: E - N + C)
    - branch_points_count: int
      * DiGraph: nodes with out-degree > 1
      * Graph: nodes with degree > 2
    - roots_count: int | None (only for DiGraph; nodes with in-degree == 0)
    - leaves_count: int (DiGraph: out-degree == 0; Graph: degree == 1)
    - self_loops: int
    - density: float (on undirected view)
    """
    directed = G.is_directed()
    U = G.to_undirected()

    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    components = nx.number_connected_components(U)
    cycles = U.number_of_edges() - U.number_of_nodes() + components

    if directed:
        branch_points = [n for n in G.nodes if G.out_degree(n) > 1]
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        roots_count: int | None = len(roots)
    else:
        branch_points = [n for n in G.nodes if G.degree(n) > 2]
        roots_count = None
        leaves = [n for n in G.nodes if G.degree(n) == 1]

    self_loops = nx.number_of_selfloops(G)
    density = nx.density(U)

    return {
        "graph_type": type(G).__name__,
        "directed": directed,
        "nodes": nodes,
        "edges": edges,
        "components": components,
        "cycles": int(cycles),
        "branch_points_count": len(branch_points),
        "roots_count": roots_count,
        "leaves_count": len(leaves),
        "self_loops": int(self_loops),
        "density": float(density),
    }


class SWCModel(nx.DiGraph):
    """SWC morphology graph representing a valid directed tree structure.

    SWCModel conforms to the SWC format specification, which requires a directed
    tree structure (no cycles). The underlying storage is a directed nx.DiGraph
    that preserves the original parent -> child relationships from the SWC format.

    Nodes are keyed by SWC id `n` and store attributes:
    - t: int (tag)
    - x, y, z: float (coordinates)
    - r: float (radius)
    - line: int (line number in source; informational)

    For graphs with cycles (e.g., after applying reconnections), use
    `make_cycle_connections()` which returns a standard nx.Graph instead of SWCModel.

    Methods like `to_swc_file()` rely on the tree structure and will only work
    correctly for valid SWC trees.
    """

    def __init__(self) -> None:
        # Initialize as a directed DiGraph; we don't need multigraph features.
        super().__init__()
        self._parents: dict[int, int | None] = {}

    # ----------------------------------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_parse_result(cls, result: SWCParseResult) -> "SWCModel":
        """Build a model from a parsed SWC result."""
        logger = logging.getLogger(__name__)
        model = cls.from_records(result.records)
        model.graph["header"] = result.header
        try:
            model.graph["reconnections"] = list(result.reconnections)
        except Exception:
            model.graph["reconnections"] = []
        logger.info(
            "SWCModel.from_parse_result records=%d reconnections=%d header=%d",
            len(result.records),
            len(result.reconnections),
            len(result.header),
        )
        return model

    @classmethod
    def from_records(
        cls, records: Mapping[int, SWCRecord] | Iterable[SWCRecord]
    ) -> "SWCModel":
        """Build a model from SWC records.

        Accepts either a mapping of id->record or any iterable of SWCRecord.
        """
        logger = logging.getLogger(__name__)
        model = cls()

        # Materialize to a list once so we can iterate twice safely
        if isinstance(records, Mapping):
            rec_values = list(records.values())
        else:
            rec_values = list(records)

        # First pass: add all nodes with attributes
        for rec in rec_values:
            model.add_node(
                rec.n,
                t=rec.t,
                x=rec.x,
                y=rec.y,
                z=rec.z,
                r=rec.r,
                line=rec.line,
            )

        # Second pass: add directed edges (parent -> child) and maintain _parents for compatibility
        pmap: dict[int, int | None] = {}
        for rec in rec_values:
            parent = None if rec.parent == -1 else rec.parent
            pmap[rec.n] = parent
            if parent is not None:
                model.add_edge(parent, rec.n)

        # store original tree parent mapping
        model._parents = pmap
        return model

    @classmethod
    def from_swc_file(
        cls,
        source: str | os.PathLike[str] | Iterable[str],
        *,
        strict: bool = True,
        validate_reconnections: bool = True,
        float_tol: float = 1e-9,
    ) -> "SWCModel":
        """Parse an SWC source then build a model.

        The `source` is passed through to `parse_swc`, which supports a path,
        a file-like object, a string with the full contents, or an iterable of lines.
        """
        logger = logging.getLogger(__name__)
        result = parse_swc(
            source,
            strict=strict,
            validate_reconnections=validate_reconnections,
            float_tol=float_tol,
        )
        model = cls.from_parse_result(result)
        logger.info(
            "SWCModel.from_swc_file built nodes=%d edges=%d strict=%s validate_reconnections=%s",
            model.number_of_nodes(),
            model.number_of_edges(),
            strict,
            validate_reconnections,
        )
        return model

    # ----------------------------------------------------------------------------------------------
    # Convenience queries
    # ----------------------------------------------------------------------------------------------
    def roots(self) -> list[int]:
        """Return nodes with no parent in the original SWC tree."""
        return [n for n in self.nodes if self.in_degree(n) == 0]

    def parent_of(self, n: int) -> int | None:
        """Return the parent id of node n from the original SWC tree (or None)."""
        preds = list(self.predecessors(n))
        return preds[0] if preds else None

    def children_of(self, node_id: int) -> list[int]:
        """Return list of child node IDs in the original SWC tree.

        Parameters
        ----------
        node_id: int
            Node ID to query.

        Returns
        -------
        list[int]
            List of node IDs that have node_id as their parent.
        """
        return list(self.successors(node_id))

    def path_to_root(self, n: int) -> list[int]:
        """Return the path from node n up to its root, inclusive.

        Example: For edges 1->2->3, `path_to_root(3)` returns `[3, 2, 1]`.
        """
        path: list[int] = [n]
        current = n
        while True:
            p = self.parent_of(current)
            if p is None:
                break
            path.append(p)
            current = p
        return path

    def get_node_xyz(
        self, node_id: int, as_array: bool = False
    ) -> tuple[float, float, float] | np.ndarray:
        """Get xyz coordinates for a node.

        Parameters
        ----------
        node_id: int
            Node ID to query.
        as_array: bool
            If True, return as numpy array. If False (default), return as tuple.

        Returns
        -------
        tuple[float, float, float] | np.ndarray
            The (x, y, z) coordinates of the node.

        Raises
        ------
        KeyError
            If node_id is not in the graph.
        ValueError
            If the node is missing x, y, or z attributes.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in graph")

        node = self.nodes[node_id]
        try:
            x = float(node["x"])
            y = float(node["y"])
            z = float(node["z"])
            if as_array:
                return np.array([x, y, z], dtype=float)
            return (x, y, z)
        except KeyError as e:
            raise ValueError(f"Node {node_id} missing coordinate attribute: {e}")

    def get_node_radius(self, node_id: int) -> float:
        """Get radius for a node.

        Parameters
        ----------
        node_id: int
            Node ID to query.

        Returns
        -------
        float
            The radius of the node. Returns 0.0 if 'r' attribute is not present.

        Raises
        ------
        KeyError
            If node_id is not in the graph.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in graph")

        return float(self.nodes[node_id].get("r", 0.0))

    def set_node_xyz(
        self,
        node_id: int,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        *,
        xyz: tuple[float, float, float] | list[float] | np.ndarray | None = None,
    ) -> None:
        """Set xyz coordinates for a node.

        Parameters
        ----------
        node_id: int
            Node ID to update.
        x, y, z: float | None
            New coordinates as separate arguments.
        xyz: tuple | list | np.ndarray | None
            New coordinates as a sequence (x, y, z). If provided, takes precedence
            over separate x, y, z arguments.

        Raises
        ------
        KeyError
            If node_id is not in the graph.
        ValueError
            If neither (x, y, z) nor xyz is provided, or if xyz has wrong length.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in graph")

        if xyz is not None:
            if len(xyz) != 3:
                raise ValueError(f"xyz must have length 3, got {len(xyz)}")
            x_val, y_val, z_val = float(xyz[0]), float(xyz[1]), float(xyz[2])
        elif x is not None and y is not None and z is not None:
            x_val, y_val, z_val = float(x), float(y), float(z)
        else:
            raise ValueError("Must provide either (x, y, z) or xyz")

        self.nodes[node_id]["x"] = x_val
        self.nodes[node_id]["y"] = y_val
        self.nodes[node_id]["z"] = z_val

    def set_node_radius(self, node_id: int, radius: float) -> None:
        """Set radius for a node.

        Parameters
        ----------
        node_id: int
            Node ID to update.
        radius: float
            New radius value.

        Raises
        ------
        KeyError
            If node_id is not in the graph.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in graph")

        self.nodes[node_id]["r"] = float(radius)

    def get_node_tag(self, node_id: int) -> int:
        """Get tag for a node.

        Parameters
        ----------
        node_id: int
            Node ID to query.

        Returns
        -------
        int
            The tag of the node. Returns 0 if 't' attribute is not present.

        Raises
        ------
        KeyError
            If node_id is not in the graph.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in graph")

        return int(self.nodes[node_id].get("t", 0))

    def set_node_tag(self, node_id: int, tag: int) -> None:
        """Set tag for a node.

        Parameters
        ----------
        node_id: int
            Node ID to update.
        tag: int
            New tag value.

        Raises
        ------
        KeyError
            If node_id is not in the graph.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in graph")

        self.nodes[node_id]["t"] = int(tag)

    def get_edge_length(self, u: int, v: int) -> float:
        """Compute Euclidean distance between two nodes.

        Parameters
        ----------
        u, v: int
            Node IDs. They do not need to be connected by an edge.

        Returns
        -------
        float
            Euclidean distance between the nodes.

        Raises
        ------
        KeyError
            If either node is not in the graph.
        ValueError
            If either node is missing coordinate attributes.
        """
        xyz_u = self.get_node_xyz(u)
        xyz_v = self.get_node_xyz(v)

        dx = xyz_v[0] - xyz_u[0]
        dy = xyz_v[1] - xyz_u[1]
        dz = xyz_v[2] - xyz_u[2]

        return float((dx * dx + dy * dy + dz * dz) ** 0.5)

    def update_radii(self, radii_dict: dict[int, float]) -> None:
        """Update radii for multiple nodes at once.

        Parameters
        ----------
        radii_dict: dict[int, float]
            Mapping of node_id -> new radius value.

        Raises
        ------
        KeyError
            If any node_id is not in the graph.
        """
        for node_id, radius in radii_dict.items():
            if node_id not in self.nodes:
                raise KeyError(f"Node {node_id} not found in graph")
            self.nodes[node_id]["r"] = float(radius)

    def update_tags(self, tags_dict: dict[int, int]) -> None:
        """Update tags for multiple nodes at once.

        Parameters
        ----------
        tags_dict: dict[int, int]
            Mapping of node_id -> new tag value.

        Raises
        ------
        KeyError
            If any node_id is not in the graph.
        """
        for node_id, tag in tags_dict.items():
            if node_id not in self.nodes:
                raise KeyError(f"Node {node_id} not found in graph")
            self.nodes[node_id]["t"] = int(tag)

    def leaves(self) -> list[int]:
        """Return leaf nodes (nodes with no children in the original SWC tree).

        Returns
        -------
        list[int]
            List of node IDs that have no children.
        """
        return [n for n in self.nodes if self.out_degree(n) == 0]

    def branch_points(self) -> list[int]:
        """Return branch point nodes (nodes with more than one child).

        Returns
        -------
        list[int]
            List of node IDs with out-degree > 1 (branch points in the directed tree).
        """
        return [n for n in self.nodes if self.out_degree(n) > 1]

    def get_subtree(self, root_id: int) -> list[int]:
        """Return all node IDs in the subtree rooted at root_id.

        Uses the original SWC tree parent relationships to traverse descendants.

        Parameters
        ----------
        root_id: int
            Root node of the subtree.

        Returns
        -------
        list[int]
            List of all node IDs in the subtree, including root_id.

        Raises
        ------
        KeyError
            If root_id is not in the graph.
        """
        if root_id not in self.nodes:
            raise KeyError(f"Node {root_id} not found in graph")

        subtree = [root_id]
        queue = [root_id]

        while queue:
            current = queue.pop(0)
            children = self.children_of(current)
            subtree.extend(children)
            queue.extend(children)

        return subtree

    def iter_edges_with_data(self):
        """Iterate edges with node attributes for both endpoints.

        Yields
        ------
        tuple[int, int, dict]
            For each edge (u, v), yields (u, v, data_dict) where data_dict contains:
            - 'u_xyz': tuple of (x, y, z) for node u
            - 'v_xyz': tuple of (x, y, z) for node v
            - 'u_r': radius of node u
            - 'v_r': radius of node v
            - 'u_t': tag of node u
            - 'v_t': tag of node v
            - 'length': Euclidean distance between u and v
        """
        for u, v in self.edges():
            yield u, v, {
                "u_xyz": self.get_node_xyz(u),
                "v_xyz": self.get_node_xyz(v),
                "u_r": self.get_node_radius(u),
                "v_r": self.get_node_radius(v),
                "u_t": self.get_node_tag(u),
                "v_t": self.get_node_tag(v),
                "length": self.get_edge_length(u, v),
            }

    def validate(self, strict: bool = True) -> list[str]:
        """Validate the model and return list of issues found.

        Parameters
        ----------
        strict: bool
            If True, perform stricter validation checks.

        Returns
        -------
        list[str]
            List of validation issue descriptions. Empty list if no issues found.
        """
        issues = []

        # Check for required attributes
        for node_id in self.nodes:
            node = self.nodes[node_id]
            for attr in ["x", "y", "z", "r"]:
                if attr not in node:
                    issues.append(f"Node {node_id} missing required attribute '{attr}'")

            # Check for zero or negative radii
            if "r" in node:
                r = node["r"]
                if r < 0:
                    issues.append(f"Node {node_id} has negative radius: {r}")
                elif strict and r == 0:
                    issues.append(f"Node {node_id} has zero radius")

        # Check parent references using DiGraph structure
        for node_id in self.nodes:
            preds = list(self.predecessors(node_id))
            if len(preds) > 1:
                issues.append(
                    f"Node {node_id} has multiple parents: {preds}"
                )
            for parent_id in preds:
                if parent_id not in self.nodes:
                    issues.append(
                        f"Node {node_id} has invalid parent reference: {parent_id}"
                    )

        # Check for disconnected components (if strict)
        if strict:
            num_components = nx.number_weakly_connected_components(self)
            if num_components > 1:
                issues.append(f"Graph has {num_components} disconnected components")

        return issues

    def print_attributes(
        self, *, node_info: bool = False, edge_info: bool = False
    ) -> None:
        """Print graph attributes and optional node/edge details.

        Parameters
        ----------
        node_info: bool
            If True, print per-node attributes (t, x, y, z, r, line where present).
        edge_info: bool
            If True, print all edges (u -- v) with edge attributes if any.
        """
        info = _graph_attributes(self)
        roots_count = len(self.roots())
        header = (
            f"SWCModel: nodes={info['nodes']}, edges={info['edges']}, "
            f"components={info['components']}, cycles={info['cycles']}, "
            f"branch_points={info['branch_points_count']}, roots={roots_count}, "
            f"leaves={info['leaves_count']}, self_loops={info['self_loops']}, density={info['density']:.4f}"
        )
        print(header)

        if node_info:
            print("Nodes:")
            ordered = ["t", "x", "y", "z", "r", "line"]
            for n, attrs in self.nodes(data=True):
                parts = [f"{k}={attrs[k]}" for k in ordered if k in attrs]
                print(f"  {n}: " + ", ".join(parts))

        if edge_info:
            print("Edges:")
            for u, v, attrs in self.edges(data=True):
                if attrs:
                    print(f"  {u} -- {v}: {dict(attrs)}")
                else:
                    print(f"  {u} -- {v}")

    def copy(self) -> "SWCModel":
        """Return a shallow copy of this model (nodes/edges/attributes)."""
        logger = logging.getLogger(__name__)
        new = super().copy(as_view=False)
        # Preserve original tree parent mapping on the copy
        try:
            new._parents = dict(self._parents)
        except AttributeError:
            # In case upstream ever returns a base nx.Graph, reconstruct
            nm = SWCModel()
            nm.add_nodes_from(new.nodes(data=True))
            nm.add_edges_from(new.edges(data=True))
            nm.graph.update(dict(new.graph))
            nm._parents = dict(self._parents)
            logger.debug(
                "SWCModel.copy reconstructed fallback nodes=%d edges=%d",
                nm.number_of_nodes(),
                nm.number_of_edges(),
            )
            return nm
        logger.debug(
            "SWCModel.copy nodes=%d edges=%d",
            new.number_of_nodes(),
            new.number_of_edges(),
        )
        return new

    def to_swc_file(
        self,
        path: str | os.PathLike[str],
        *,
        precision: int = 6,
        header: Iterable[str] | None = None,
    ) -> None:
        """Write the model to an SWC file.

        The output uses the standard 7-column SWC format per row:
        "n T x y z r parent" with floats formatted to the requested precision.

        Parameters
        ----------
        path: str | os.PathLike[str]
            Destination file path.
        precision: int
            Decimal places for floating-point fields (x, y, z, r). Default 6.
        header: Iterable[str] | None
            Optional additional header comment lines (without leading '#').
        """
        logger = logging.getLogger(__name__)
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("precision must be a non-negative integer")

        # Prepare header lines
        header_lines: list[str] = self.graph.get("header", [])
        if header:
            for line in header:
                text = str(line).rstrip("\n")
                if text.startswith("#"):
                    header_lines.append(text)
                else:
                    header_lines.append(f"# {text}")

        fmt = f"{{:.{precision}f}}"

        # Write file
        path_str = os.fspath(path)
        with open(path_str, "w", encoding="utf-8", newline="\n") as f:
            for line in header_lines:
                f.write(line + "\n")

            # Write nodes ordered by id
            for n in sorted(int(i) for i in self.nodes):
                attrs = self.nodes[n]
                t = int(attrs.get("t", 0))
                x = fmt.format(float(attrs.get("x", 0.0)))
                y = fmt.format(float(attrs.get("y", 0.0)))
                z = fmt.format(float(attrs.get("z", 0.0)))
                r = fmt.format(float(attrs.get("r", 0.0)))
                parent = self.parent_of(n)
                pval = -1 if parent is None else int(parent)
                f.write(f"{n} {t} {x} {y} {z} {r} {pval}\n")
        logger.info(
            "SWCModel.to_swc_file wrote path=%s nodes=%d edges=%d header_lines=%d",
            os.fspath(path),
            self.number_of_nodes(),
            self.number_of_edges(),
            len(header_lines),
        )

    def scale(self, scalar: float) -> "SWCModel":
        """Return a new model with all node coordinates and radii scaled by `scalar`.

        Multiplies each node's `x`, `y`, `z`, and `r` by `scalar` on a copy.
        """
        logger = logging.getLogger(__name__)
        if not isinstance(scalar, (int, float)):
            raise TypeError("scalar must be a number")
        new = self.copy()
        for _, attrs in new.nodes(data=True):
            if "x" in attrs:
                attrs["x"] *= scalar
            if "y" in attrs:
                attrs["y"] *= scalar
            if "z" in attrs:
                attrs["z"] *= scalar
            if "r" in attrs:
                attrs["r"] *= scalar
        return new

    def set_tag_by_sphere(
        self,
        center: tuple[float, float, float] | list[float],
        radius: float,
        new_tag: int,
        old_tag: int | None = None,
        include_boundary: bool = True,
        copy: bool = False,
    ) -> "SWCModel":
        """Override node 't' values for points inside a sphere.

        Sets the tag 't' for all nodes whose Euclidean distance from
        `center` is less than `radius` (or equal if `include_boundary` is True).

        If `old_tag` is specified, only nodes with that tag are modified.

        Parameters
        ----------
        center: tuple[float, float, float] | list[float]
            Sphere center as (x, y, z).
        radius: float
            Sphere radius (same units as coordinates).
        new_tag: int
            Tag to assign to matching nodes.
        old_tag: int | None
            If specified, only nodes with this tag are modified.
        include_boundary: bool
            If True, include nodes exactly at distance == radius. Default True.
        copy: bool
            If True, operate on and return a copy; otherwise mutate in place and return self.
        """
        logger = logging.getLogger(__name__)
        try:
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        except Exception as e:  # noqa: BLE001
            raise ValueError("center must be a sequence of three numbers") from e
        if not isinstance(radius, (int, float)) or radius < 0:
            raise ValueError("radius must be a non-negative number")
        if not isinstance(new_tag, int):
            raise TypeError("new_tag must be an int")
        target = self.copy() if copy else self
        r2 = float(radius) * float(radius)
        changed = 0
        for _, attrs in target.nodes(data=True):
            x = float(attrs.get("x", 0.0))
            y = float(attrs.get("y", 0.0))
            z = float(attrs.get("z", 0.0))
            dx = x - cx
            dy = y - cy
            dz = z - cz
            d2 = dx * dx + dy * dy + dz * dz
            inside = d2 <= r2 if include_boundary else d2 < r2
            if inside:
                if old_tag is not None and attrs.get("t") != int(old_tag):
                    continue
                if attrs.get("t") != int(new_tag):
                    changed += 1
                attrs["t"] = int(new_tag)
        logger.info(
            "set_tag_by_sphere center=(%.6f, %.6f, %.6f) radius=%.6f new_tag=%d old_tag=%s changed=%d copy=%s",
            cx,
            cy,
            cz,
            radius,
            int(new_tag),
            old_tag,
            changed,
            copy,
        )
        return target

    # ----------------------------------------------------------------------------------------------
    # Junction management
    # ----------------------------------------------------------------------------------------------
    def add_junction(
        self,
        node_id: int | None = None,
        *,
        t: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        r: float = 0.0,
        parent: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Add a junction (node) to the model.

        Parameters
        ----------
        node_id: int | None
            Node ID to use. If None, automatically assigns the next available ID.
        t: int
            Node tag. Default 0.
        x, y, z: float
            Node coordinates. Default 0.0.
        r: float
            Node radius. Default 0.0.
        parent: int | None
            Parent node ID. If specified, creates an edge to the parent.
            Default None (root node).
        **kwargs: Any
            Additional node attributes.

        Returns
        -------
        int
            The ID of the added node.
        """
        logger = logging.getLogger(__name__)

        if node_id is None:
            node_id = max(self.nodes, default=0) + 1
        else:
            node_id = int(node_id)
            if node_id in self.nodes:
                raise ValueError(f"Node {node_id} already exists")

        if parent is not None:
            parent = int(parent)
            if parent not in self.nodes:
                raise ValueError(f"Parent node {parent} does not exist")

        attrs = {
            "t": int(t),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "r": float(r),
        }
        attrs.update(kwargs)

        self.add_node(node_id, **attrs)
        # Maintain _parents map for compatibility with existing methods
        self._parents[node_id] = parent

        if parent is not None:
            self.add_edge(parent, node_id)

        logger.debug(
            "add_junction node_id=%d parent=%s t=%d pos=(%.3f, %.3f, %.3f) r=%.3f",
            node_id,
            parent,
            t,
            x,
            y,
            z,
            r,
        )
        return node_id

    def remove_junction(
        self,
        node_id: int,
        *,
        reconnect_children: bool = False,
    ) -> None:
        """Remove a junction (node) from the model.

        Parameters
        ----------
        node_id: int
            ID of the node to remove.
        reconnect_children: bool
            If True, reconnect children of the removed node to its parent.
            If False (default), children become orphaned (roots).
        """
        logger = logging.getLogger(__name__)

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")

        parent = self.parent_of(node_id)
        children = list(self.successors(node_id))

        if reconnect_children and parent is not None:
            for child in children:
                self.add_edge(parent, child)
                # Update _parents map for compatibility
                self._parents[child] = parent
        else:
            for child in children:
                # Update _parents map for compatibility
                self._parents[child] = None

        # Remove from _parents map
        if node_id in self._parents:
            del self._parents[node_id]
        self.remove_node(node_id)

        logger.debug(
            "remove_junction node_id=%d parent=%s children=%d reconnect=%s",
            node_id,
            parent,
            len(children),
            reconnect_children,
        )

    def make_cycle_connections(
        self,
        *,
        validate_reconnections: bool = True,
        float_tol: float = 1e-9,
    ) -> nx.Graph:
        """Return an undirected nx.Graph with reconnection pairs merged.

        Uses reconnection pairs stored under `self.graph['reconnections']` if present.
        Node attributes are merged; provenance kept under `merged_ids` and `lines`.

        The returned graph may contain cycles and is no longer a valid SWC tree structure,
        so it returns nx.Graph instead of SWCModel. SWCModel should only represent valid
        directed tree structures conforming to the SWC format.

        Returns
        -------
        nx.Graph
            Undirected graph with merged nodes and edges. Node attributes include
            x, y, z, r, t, merged_ids (list of original node IDs), and lines.
        """
        logger = logging.getLogger(__name__)
        pairs = list(self.graph.get("reconnections", []))
        if not pairs:
            logger.info("make_cycle_connections no pairs; returning copy")
            return self.copy()

        parent: dict[int, int] = {}
        rank: dict[int, int] = {}

        def uf_find(a: int) -> int:
            pa = parent.get(a, a)
            if pa != a:
                parent[a] = uf_find(pa)
            else:
                parent.setdefault(a, a)
                rank.setdefault(a, 0)
            return parent[a]

        def uf_union(a: int, b: int) -> None:
            ra, rb = uf_find(a), uf_find(b)
            if ra == rb:
                return
            rra, rrb = rank.get(ra, 0), rank.get(rb, 0)
            if rra < rrb or (rra == rrb and ra > rb):
                ra, rb = rb, ra
                rra, rrb = rrb, rra
            parent[rb] = ra
            rank[ra] = max(rra, rrb + 1)

        def identical_xyzr(i: int, j: int) -> bool:
            ai = self.nodes[i]
            aj = self.nodes[j]
            return (
                abs(ai["x"] - aj["x"]) <= float_tol
                and abs(ai["y"] - aj["y"]) <= float_tol
                and abs(ai["z"] - aj["z"]) <= float_tol
                and abs(ai["r"] - aj["r"]) <= float_tol
            )

        # Seed UF with all ids present
        for n in self.nodes:
            parent[int(n)] = int(n)
            rank[int(n)] = 0

        # Apply merges
        for i, j in pairs:
            if i not in self.nodes or j not in self.nodes:
                raise ValueError(
                    f"Reconnection pair ({i}, {j}) refers to undefined node id(s)"
                )
            if validate_reconnections and not identical_xyzr(int(i), int(j)):
                ai = self.nodes[int(i)]
                aj = self.nodes[int(j)]
                raise ValueError(
                    "Reconnection requires identical (x, y, z, r) but got:\n"
                    f"  {i}: (x={ai['x']}, y={ai['y']}, z={ai['z']}, r={ai['r']})\n"
                    f"  {j}: (x={aj['x']}, y={aj['y']}, z={aj['z']}, r={aj['r']})"
                )
            uf_union(int(i), int(j))

        # Build groups by representative
        groups: dict[int, list[int]] = {}
        for n in self.nodes:
            r = uf_find(int(n))
            groups.setdefault(r, []).append(int(n))

        # Create the merged graph (nx.Graph, not SWCModel)
        model = nx.Graph()
        for rep, ids in groups.items():
            ids_sorted = sorted(ids)
            first = self.nodes[ids_sorted[0]]
            lines = sorted(
                int(self.nodes[i]["line"])
                for i in ids_sorted
                if "line" in self.nodes[i]
            )
            attrs = {
                "n": ids_sorted[0],
                "x": float(first["x"]),
                "y": float(first["y"]),
                "z": float(first["z"]),
                "r": float(first["r"]),
                "t": int(first["t"]) if "t" in first else first.get("t"),
                "merged_ids": ids_sorted,
                "lines": lines,
            }
            model.add_node(int(rep), **attrs)

        # Add undirected edges between merged representatives (skip self-loops)
        for n, p in self._parents.items():
            if p is None:
                continue
            u = uf_find(int(p))
            v = uf_find(int(n))
            if u != v:
                model.add_edge(u, v)

        # Store reconnection metadata in graph attributes
        model.graph["reconnections"] = pairs
        logger.info(
            "make_cycle_connections merged=%d groups=%d nodes=%d edges=%d",
            len(pairs),
            len(groups),
            model.number_of_nodes(),
            model.number_of_edges(),
        )
        return model
