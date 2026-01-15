import pytest
import networkx as nx
from swctools import Segment, frustum_mesh, batch_frusta, FrustaSet, SWCModel


def test_single_frustum_mesh_counts():
    """Ensure `frustum_mesh` yields expected vertex/triangle counts for one segment without end caps."""
    seg = Segment(a=(0, 0, 0), b=(1, 0, 0), ra=0.5, rb=0.25)
    v, f = frustum_mesh(seg, sides=12, end_caps=False)
    # 2 * sides vertices; 2 * sides faces for side quads
    assert len(v) == 24
    assert len(f) == 24


def test_batch_frusta_reindexing():
    """Verify `batch_frusta` concatenates meshes across segments and correctly reindexes face indices."""
    segs = [
        Segment(a=(0, 0, 0), b=(1, 0, 0), ra=0.5, rb=0.25),
        Segment(a=(1, 0, 0), b=(2, 0, 0), ra=0.25, rb=0.2),
    ]
    v, f = batch_frusta(segs, sides=10, end_caps=False)
    assert len(v) == 40  # 2 segments * (2 * sides)
    assert len(f) == 40  # 2 segments * (2 * sides) -> 40 triangles
    # Ensure face indices are in range
    assert max(max(face) for face in f) < len(v)


def test_frustaset_from_swc_model_and_arrays():
    """Build `FrustaSet` from an `SWCModel` and confirm Mesh3d arrays match vertices/faces lengths."""
    swc = """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1
3 3 1 0 0 0.5 1
4 3 2 0 0 0.4 2
""".strip()
    m = SWCModel.from_swc_file(swc, strict=True, validate_reconnections=True)
    gm = m.make_cycle_connections(validate_reconnections=True)
    fr = FrustaSet.from_swc_model(gm, sides=8, end_caps=False)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    # Basic shape checks
    assert len(x) == len(y) == len(z) == len(fr.vertices)
    assert len(i) == len(j) == len(k) == len(fr.faces)


def test_frustaset_validates_required_attributes():
    """Verify FrustaSet.from_swc_model validates that nodes have x, y, z, r attributes."""
    # Create a graph missing required attributes
    G = nx.Graph()
    G.add_node(1, x=0.0, y=0.0, z=0.0)  # Missing 'r'
    G.add_node(2, x=1.0, y=0.0, z=0.0, r=0.5)
    G.add_edge(1, 2)

    with pytest.raises(ValueError, match="Node 1 missing required attributes.*'r'"):
        FrustaSet.from_swc_model(G, sides=8)

    # Create a graph with all required attributes - should work
    G2 = nx.Graph()
    G2.add_node(1, x=0.0, y=0.0, z=0.0, r=1.0, t=1)
    G2.add_node(2, x=1.0, y=0.0, z=0.0, r=0.5, t=3)
    G2.add_edge(1, 2)

    fr = FrustaSet.from_swc_model(G2, sides=8, end_caps=False)
    assert fr.segment_count == 1
