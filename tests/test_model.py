from pathlib import Path
import pytest

from swctools import parse_swc, SWCParseResult, SWCRecord, SWCModel, GeneralModel


def test_build_from_parse_result_basic():
    """Build SWCModel from a parsed result and verify topology and queries.

    Checks: nodes, directed edges, roots(), parent_of(), path_to_root().
    """
    swc = """
# Simple SWC
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1
3 3 2 0 0 0.4 2
""".strip()
    result = parse_swc(swc)
    m = SWCModel.from_parse_result(result)

    assert set(m.nodes) == {1, 2, 3}
    # Edges parent -> child
    assert m.has_edge(1, 2)
    assert m.has_edge(2, 3)
    # Roots and queries
    assert m.roots() == [1]
    assert m.parent_of(2) == 1
    assert m.path_to_root(3) == [3, 2, 1]


def test_from_records_iterable():
    """Build records manually and construct SWCModel via from_records().

    Verifies nodes, an edge, and root detection.
    """
    # Build records manually to exercise iterable path
    lines = [
        "1 1 0 0 0 1 -1",
        "2 3 1 0 0 0.5 1",
    ]
    result = parse_swc(lines)
    recs = list(result.records.values())
    m = SWCModel.from_records(recs)
    assert set(m.nodes) == {1, 2}
    assert m.has_edge(1, 2)
    assert m.roots() == [1]


def test_from_swc_file(tmp_path: Path):
    """Parse an SWC from a file path and validate roots and path_to_root()."""
    p = tmp_path / "tmp_model.swc"
    p.write_text(
        """
# header
1 1 0 0 0 1 -1
2 3 2 0 0 0.5 1
3 3 2 0 0 0.5 1
""".strip()
    )
    m = SWCModel.from_swc_file(p)
    assert set(m.nodes) == {1, 2, 3}
    assert m.roots() == [1]
    assert m.path_to_root(2) == [2, 1]


def test_general_model_reconnection_merge_and_print(capsys):
    """Ensure GeneralModel merges reconnection pairs and print_attributes outputs a summary.

    Verifies merged node set, presence of an edge, and that printed output contains
    expected summary tokens.
    """
    swc = """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 2 0 0 0.5 1
3 3 2 0 0 0.5 1
""".strip()
    gm = GeneralModel.from_swc_file(swc, strict=True, validate_reconnections=True)
    # Nodes 2 and 3 should merge into a single representative; expect nodes {1, 2}
    assert set(gm.nodes) == {1, 2}
    assert gm.has_edge(1, 2)

    # print_attributes should not raise and should include summary keywords
    gm.print_attributes()
    gm.print_attributes(node_info=True, edge_info=True)
    out = capsys.readouterr().out
    assert "GeneralModel:" in out
    assert "nodes=" in out and "edges=" in out


def test_swcmodel_print_attributes_no_crash(capsys):
    """Smoke test for SWCModel.print_attributes with and without details.

    Asserts the header line is present in stdout.
    """
    swc = """
# Simple SWC
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1
""".strip()
    m = SWCModel.from_swc_file(swc)
    m.print_attributes()
    m.print_attributes(node_info=True, edge_info=True)
    out = capsys.readouterr().out
    assert "SWCModel:" in out


def test_optional_data_directory_models_if_present():
    """Optionally scan example SWC files under data/swc and validate basic structure.

    Skips gracefully if the data directory or .swc files are absent.
    """
    # Optional scan mirroring test_io; skip if data/swc absent
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "swc"
    if not data_dir.exists():
        pytest.skip("data/swc not present")
    files = list(data_dir.glob("*.swc"))
    if not files:
        pytest.skip("no .swc files found in data/swc")

    for swc_path in files:
        # Be lenient about reconnection validation for arbitrary examples
        m = SWCModel.from_swc_file(swc_path, strict=True, validate_reconnections=False)
        assert isinstance(m, SWCModel)
        assert m.number_of_nodes() > 0
        # Every non-root should have in-degree 1
        for n in m.nodes:
            indeg = m.in_degree(n)
            if n in set(m.roots()):
                assert indeg == 0
            else:
                assert indeg == 1
