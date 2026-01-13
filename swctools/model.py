"""SWC graph data model.

SWCModel stores SWC morphology as an undirected graph where nodes are SWC points,
and the original directed parent -> child relationships are preserved in an
internal parent map.

Notes
-----
- Use `SWCModel` for topology and attribute management of parsed SWC, and for
  reconnection merges via `SWCModel.make_cycle_connections()`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Any
import os
import networkx as nx
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


class SWCModel(nx.Graph):
    """SWC morphology graph (undirected storage with directed-tree metadata).

    Nodes are keyed by SWC id `n` and store attributes:
    - t: int (tag)
    - x, y, z: float (coordinates)
    - r: float (radius)
    - line: int (line number in source; informational)

    Directed parent -> child relationships from the SWC are preserved via an
    internal parent map (`_parents`). The underlying graph is undirected. Cycle
    connections can be applied via `make_cycle_connections()` which may merge nodes.
    """

    def __init__(self) -> None:
        # Initialize as a plain Graph; we don't need multigraph features.
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

        # Second pass: record tree parents and add undirected edges
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
        return [n for n, p in self._parents.items() if p is None]

    def parent_of(self, n: int) -> int | None:
        """Return the parent id of node n from the original SWC tree (or None)."""
        return self._parents.get(n)

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
                parent = self._parents.get(n)
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

        parent = self._parents.get(node_id)
        children = [n for n, p in self._parents.items() if p == node_id]

        if reconnect_children and parent is not None:
            for child in children:
                self._parents[child] = parent
                self.add_edge(parent, child)
        else:
            for child in children:
                self._parents[child] = None

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
    ) -> "SWCModel":
        """Return a new SWCModel with reconnection pairs merged and undirected edges across merged reps.

        Uses reconnection pairs stored under `self.graph['reconnections']` if present.
        Node attributes are merged; provenance kept under `merged_ids` and `lines`.
        After merges, the original tree parent mapping no longer applies and parents are set to None.
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

        # Create the merged model nodes
        model = SWCModel()
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

        # After merges, original tree mapping is not applicable
        model._parents = {int(rep): None for rep in groups.keys()}
        model.graph["reconnections"] = pairs
        logger.info(
            "make_cycle_connections merged=%d groups=%d nodes=%d edges=%d",
            len(pairs),
            len(groups),
            model.number_of_nodes(),
            model.number_of_edges(),
        )
        return model
