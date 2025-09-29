"""SWC graph data model.

SWCModel is a directed graph (forest) where nodes are SWC points and
edges are directed from parent -> child according to the SWC format.

Notes
-----
- Use `SWCModel` for topology and attribute management of parsed SWC.
- Visualization should operate on a separate undirected `GeneralModel`
  that may perform reconnection merges; see project TODOs.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Any
import os
import networkx as nx

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
    """Directed SWC morphology graph.

    Nodes are keyed by SWC id `n` and store attributes:
    - t: int (structure type)
    - x, y, z: float (coordinates)
    - r: float (radius)
    - line: int (line number in source; informational)

    Edges are directed parent -> child.
    """

    def __init__(self) -> None:
        # Initialize as a plain DiGraph; we don't need multigraph features.
        super().__init__()

    # ----------------------------------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_parse_result(cls, result: SWCParseResult) -> "SWCModel":
        """Build a model from a parsed SWC result."""
        return cls.from_records(result.records)

    @classmethod
    def from_records(
        cls, records: Mapping[int, SWCRecord] | Iterable[SWCRecord]
    ) -> "SWCModel":
        """Build a model from SWC records.

        Accepts either a mapping of id->record or any iterable of SWCRecord.
        """
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

        # Second pass: add edges parent -> child
        for rec in rec_values:
            if rec.parent != -1:
                model.add_edge(rec.parent, rec.n)

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
        result = parse_swc(
            source,
            strict=strict,
            validate_reconnections=validate_reconnections,
            float_tol=float_tol,
        )
        return cls.from_parse_result(result)

    # ----------------------------------------------------------------------------------------------
    # Convenience queries
    # ----------------------------------------------------------------------------------------------
    def roots(self) -> list[int]:
        """Return nodes with in-degree 0 (forest roots)."""
        return [n for n, deg in self.in_degree() if deg == 0]

    def parent_of(self, n: int) -> int | None:
        """Return the parent id of node n, or None if n is a root.

        SWC trees should have at most one parent per node; if multiple are found
        this indicates invalid structure for SWC and an error is raised.
        """
        preds = list(self.predecessors(n))
        if not preds:
            return None
        if len(preds) > 1:
            raise ValueError(
                f"Node {n} has multiple parents in SWCModel; expected a tree/forest"
            )
        return preds[0]

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

    def print_attributes(self, *, node_info: bool = False, edge_info: bool = False) -> None:
        """Print graph attributes and optional node/edge details.

        Parameters
        ----------
        node_info: bool
            If True, print per-node attributes (t, x, y, z, r, line where present).
        edge_info: bool
            If True, print all edges (u -> v) with edge attributes if any.
        """
        info = _graph_attributes(self)
        header = (
            f"SWCModel: nodes={info['nodes']}, edges={info['edges']}, "
            f"components={info['components']}, cycles={info['cycles']}, "
            f"branch_points={info['branch_points_count']}, roots={info['roots_count']}, "
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
                    print(f"  {u} -> {v}: {dict(attrs)}")
                else:
                    print(f"  {u} -> {v}")


class GeneralModel(nx.Graph):
    """Undirected morphology graph with reconnection merges.

    - Subclasses `networkx.Graph`.
    - Nodes correspond to merged SWC points according to header annotations
      `# CYCLE_BREAK reconnect i j`.
    - Node attributes include: `x, y, z, r` (identical across merged ids),
      representative `n`, optional `t`, and provenance lists `merged_ids`, `lines`.
    - Edges are undirected between merged nodes; self-loops are skipped if
      parent/child collapse into the same merged node.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------------------------------
    @classmethod
    def from_parse_result(
        cls,
        result: SWCParseResult,
        *,
        validate_reconnections: bool = True,
        float_tol: float = 1e-9,
    ) -> "GeneralModel":
        """Build a merged undirected model from a parsed SWC result.

        If `validate_reconnections` is True, enforce identical (x, y, z, r)
        for each reconnect pair before merging (useful when `parse_swc` was
        called with validation disabled).
        """
        # Materialize record mapping
        records = result.records

        # ---- Union-Find for merges -------------------------------------------------------------
        parent: dict[int, int] = {}
        rank: dict[int, int] = {}

        def uf_find(a: int) -> int:
            # Path compression
            pa = parent.get(a, a)
            if pa != a:
                parent[a] = uf_find(pa)
            else:
                parent.setdefault(a, a)
                rank.setdefault(a, 0)
            return parent[a]

        def identical_xyzr(a: SWCRecord, b: SWCRecord) -> bool:
            return (
                abs(a.x - b.x) <= float_tol
                and abs(a.y - b.y) <= float_tol
                and abs(a.z - b.z) <= float_tol
                and abs(a.r - b.r) <= float_tol
            )

        def uf_union(a: int, b: int) -> None:
            ra, rb = uf_find(a), uf_find(b)
            if ra == rb:
                return
            # Union by rank; tie-breaker on smaller id for stability
            rra, rrb = rank.get(ra, 0), rank.get(rb, 0)
            if rra < rrb or (rra == rrb and ra > rb):
                ra, rb = rb, ra
                rra, rrb = rrb, rra
            parent[rb] = ra
            rank[ra] = max(rra, rrb + 1)

        # Seed UF with all ids
        for n in records.keys():
            parent[n] = n
            rank[n] = 0

        # Apply merges from reconnection annotations
        for i, j in result.reconnections:
            if i not in records or j not in records:
                raise ValueError(
                    f"Reconnection pair ({i}, {j}) refers to undefined node id(s)"
                )
            if validate_reconnections:
                if not identical_xyzr(records[i], records[j]):
                    raise ValueError(
                        "Reconnection requires identical (x, y, z, r) but got:\n"
                        f"  {i}: (x={records[i].x}, y={records[i].y}, z={records[i].z}, r={records[i].r})\n"
                        f"  {j}: (x={records[j].x}, y={records[j].y}, z={records[j].z}, r={records[j].r})"
                    )
            uf_union(i, j)

        # Build groups by representative
        groups: dict[int, list[int]] = {}
        for n in records.keys():
            r = uf_find(n)
            groups.setdefault(r, []).append(n)

        # Create the Graph nodes with merged attributes
        model = cls()
        for rep, ids in groups.items():
            # Sort ids for stable ordering and reproducibility
            ids_sorted = sorted(ids)
            first = records[ids_sorted[0]]
            # Attributes are taken from the first (coordinates identical by contract)
            attrs = {
                "n": ids_sorted[0],
                "x": first.x,
                "y": first.y,
                "z": first.z,
                "r": first.r,
                # Representative type; may vary across merged ids, but keep one for convenience
                "t": first.t,
                # Provenance
                "merged_ids": ids_sorted,
                "lines": sorted(records[i].line for i in ids_sorted),
            }
            model.add_node(rep, **attrs)

        # Add undirected edges between merged representatives (skip self-loops)
        for rec in records.values():
            if rec.parent == -1:
                continue
            u = uf_find(rec.parent)
            v = uf_find(rec.n)
            if u != v:
                model.add_edge(u, v)

        return model

    @classmethod
    def from_swc_file(
        cls,
        source: str | os.PathLike[str] | Iterable[str],
        *,
        strict: bool = True,
        validate_reconnections: bool = True,
        float_tol: float = 1e-9,
    ) -> "GeneralModel":
        """Parse an SWC source and build a merged undirected model."""
        result = parse_swc(
            source,
            strict=strict,
            validate_reconnections=validate_reconnections,
            float_tol=float_tol,
        )
        return cls.from_parse_result(
            result,
            validate_reconnections=validate_reconnections,
            float_tol=float_tol,
        )

    def print_attributes(self, *, node_info: bool = False, edge_info: bool = False) -> None:
        """Print graph attributes and optional node/edge details.

        Parameters
        ----------
        node_info: bool
            If True, print per-node attributes (n, x, y, z, r, t, merged_ids, lines where present).
        edge_info: bool
            If True, print all edges (u -- v) with edge attributes if any.
        """
        info = _graph_attributes(self)
        header = (
            f"GeneralModel: nodes={info['nodes']}, edges={info['edges']}, "
            f"components={info['components']}, cycles={info['cycles']}, "
            f"branch_points={info['branch_points_count']}, leaves={info['leaves_count']}, "
            f"self_loops={info['self_loops']}, density={info['density']:.4f}"
        )
        print(header)

        if node_info:
            print("Nodes:")
            ordered = ["n", "x", "y", "z", "r", "t", "merged_ids", "lines"]
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
