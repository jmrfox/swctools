from swctools import Segment, frustum_mesh, batch_frusta, FrustaSet, GeneralModel


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


def test_frustaset_from_general_model_and_arrays():
    """Build `FrustaSet` from a `GeneralModel` and confirm Mesh3d arrays match vertices/faces lengths."""
    swc = """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1
3 3 1 0 0 0.5 1
4 3 2 0 0 0.4 2
""".strip()
    gm = GeneralModel.from_swc_file(swc, strict=True, validate_reconnections=True)
    fr = FrustaSet.from_general_model(gm, sides=8, end_caps=False)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    # Basic shape checks
    assert len(x) == len(y) == len(z) == len(fr.vertices)
    assert len(i) == len(j) == len(k) == len(fr.faces)
