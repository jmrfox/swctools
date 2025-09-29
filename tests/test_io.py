from pathlib import Path
import pytest
from swcviz import parse_swc, SWCParseResult


def test_parse_basic_from_string():
    swc = """
# Simple SWC
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1
""".strip()
    result = parse_swc(swc)
    assert isinstance(result, SWCParseResult)
    assert set(result.records.keys()) == {1, 2}
    assert result.records[1].parent == -1
    assert result.records[2].parent == 1
    assert result.reconnections == []


def test_parse_reconnections_valid_identical_xyzr():
    swc = """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 2 0 0 0.5 1
3 3 2 0 0 0.5 1
""".strip()
    result = parse_swc(swc, strict=True, validate_reconnections=True)
    assert (2, 3) in result.reconnections


def test_parse_reconnections_invalid_mismatch_radius():
    swc = """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 2 0 0 0.5 1
3 3 2 0 0 0.6 1
""".strip()
    with pytest.raises(
        ValueError, match=r"Reconnection requires identical.*\(x, y, z, r\)"
    ):
        parse_swc(swc, strict=True, validate_reconnections=True)


def test_strict_columns_enforced_and_relaxed():
    swc_strict_fail = """
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1 99
""".strip()
    with pytest.raises(ValueError, match=r"expected exactly 7 columns"):
        parse_swc(swc_strict_fail, strict=True)

    # Non-strict should ignore extra columns beyond 7
    result = parse_swc(swc_strict_fail, strict=False)
    assert set(result.records.keys()) == {1, 2}


def test_missing_parent_raises():
    swc = """
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 99
""".strip()
    with pytest.raises(ValueError, match=r"parent id 99 does not exist"):
        parse_swc(swc, strict=True)


def test_integer_coercion_and_iterable_input():
    lines = [
        "# header",
        "1 1 0 0 0 1 -1",
        "2 3 1 0 0 0.5 1.0",  # parent appears as 1.0
        "3.0 3 2 0 0 0.5 1",  # id appears as 3.0
    ]
    result = parse_swc(lines)
    assert set(result.records.keys()) == {1, 2, 3}
    assert result.records[2].parent == 1


def test_reconnect_directive_case_insensitive_and_sorted():
    swc = """
# cycle_break RECONNECT 9 2
1 1 0 0 0 1 -1
2 3 1 0 0 0.5 1
9 3 1 0 0 0.5 1
""".strip()
    result = parse_swc(swc)
    # Should normalize order to (2, 9)
    assert (2, 9) in result.reconnections


def test_parse_from_file_path(tmp_path: Path):
    p = tmp_path / "tmp.swc"
    p.write_text(
        """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 2 0 0 0.5 1
3 3 2 0 0 0.5 1
""".strip()
    )
    result = parse_swc(p)
    assert set(result.records.keys()) == {1, 2, 3}
    assert (2, 3) in result.reconnections


def test_parse_example_data_directory_if_present():
    # This test is optional; it will be skipped if the data/swc directory is missing or empty
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "swc"
    if not data_dir.exists():
        pytest.skip("data/swc not present")
    files = list(data_dir.glob("*.swc"))
    if not files:
        pytest.skip("no .swc files found in data/swc")

    for swc_path in files:
        # Be lenient about reconnection validation for arbitrary examples
        result = parse_swc(swc_path, strict=True, validate_reconnections=False)
        assert isinstance(result, SWCParseResult)
        assert len(result.records) > 0
