"""Test script to identify write/read bug in SWCModel.

This script creates an SWCModel, writes it to a file, reads it back,
and compares graph attributes to identify any discrepancies.
"""

import tempfile
import os
from pathlib import Path
import networkx as nx
from swctools import SWCModel, parse_swc


def compare_models(original: SWCModel, loaded: SWCModel, label: str = ""):
    """Compare two SWCModel instances and report differences."""
    print(f"\n{'='*70}")
    print(f"Comparison: {label}")
    print(f"{'='*70}")

    # Basic graph properties
    print(f"\nNodes:")
    print(f"  Original: {original.number_of_nodes()}")
    print(f"  Loaded:   {loaded.number_of_nodes()}")
    print(f"  Match: {original.number_of_nodes() == loaded.number_of_nodes()}")

    print(f"\nEdges:")
    print(f"  Original: {original.number_of_edges()}")
    print(f"  Loaded:   {loaded.number_of_edges()}")
    print(f"  Match: {original.number_of_edges() == loaded.number_of_edges()}")

    # Components (use weakly connected components for directed graphs)
    orig_components = nx.number_weakly_connected_components(original)
    load_components = nx.number_weakly_connected_components(loaded)
    print(f"\nConnected Components:")
    print(f"  Original: {orig_components}")
    print(f"  Loaded:   {load_components}")
    print(f"  Match: {orig_components == load_components}")

    # Roots
    orig_roots = original.roots()
    load_roots = loaded.roots()
    print(f"\nRoots:")
    print(f"  Original: {orig_roots}")
    print(f"  Loaded:   {load_roots}")
    print(f"  Match: {orig_roots == load_roots}")

    # Parent map
    print(f"\nParent map (_parents):")
    print(f"  Original keys: {sorted(original._parents.keys())}")
    print(f"  Loaded keys:   {sorted(loaded._parents.keys())}")
    print(f"  Match: {original._parents == loaded._parents}")

    if original._parents != loaded._parents:
        print(f"\n  Differences in parent map:")
        all_keys = set(original._parents.keys()) | set(loaded._parents.keys())
        for key in sorted(all_keys):
            orig_val = original._parents.get(key, "MISSING")
            load_val = loaded._parents.get(key, "MISSING")
            if orig_val != load_val:
                print(f"    Node {key}: original={orig_val}, loaded={load_val}")

    # Node attributes
    print(f"\nNode attributes:")
    node_attrs_match = True
    for node in original.nodes():
        if node not in loaded.nodes():
            print(f"  Node {node} missing in loaded model")
            node_attrs_match = False
            continue
        orig_attrs = original.nodes[node]
        load_attrs = loaded.nodes[node]
        if orig_attrs != load_attrs:
            print(f"  Node {node} attributes differ:")
            print(f"    Original: {orig_attrs}")
            print(f"    Loaded:   {load_attrs}")
            node_attrs_match = False
    print(f"  All node attributes match: {node_attrs_match}")

    # Graph attributes
    print(f"\nGraph attributes (graph dict):")
    print(f"  Original: {original.graph}")
    print(f"  Loaded:   {loaded.graph}")
    print(f"  Match: {original.graph == loaded.graph}")

    return orig_components == load_components


def test_simple_tree():
    """Test with a simple tree structure."""
    print("\n" + "=" * 70)
    print("TEST 1: Simple Tree")
    print("=" * 70)

    swc_content = """# Simple tree
1 1 0 0 0 1.0 -1
2 3 1 0 0 0.5 1
3 3 2 0 0 0.4 2
4 3 3 0 0 0.3 2
"""

    # Create original model
    original = SWCModel.from_swc_file(swc_content.strip().split("\n"))

    # Write and read
    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        original.to_swc_file(temp_path)
        loaded = SWCModel.from_swc_file(temp_path)

        match = compare_models(original, loaded, "Simple Tree")
        assert match, "Models should match for Simple Tree"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_branching_tree():
    """Test with a branching tree structure."""
    print("\n" + "=" * 70)
    print("TEST 2: Branching Tree")
    print("=" * 70)

    swc_content = """# Branching tree
1 1 0 0 0 1.0 -1
2 3 1 0 0 0.5 1
3 3 2 0 0 0.4 2
4 3 1 1 0 0.5 1
5 3 2 1 0 0.4 4
"""

    original = SWCModel.from_swc_file(swc_content.strip().split("\n"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        original.to_swc_file(temp_path)
        loaded = SWCModel.from_swc_file(temp_path)

        match = compare_models(original, loaded, "Branching Tree")
        assert match, "Models should match for Branching Tree"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_multiple_roots():
    """Test with multiple disconnected components (multiple roots)."""
    print("\n" + "=" * 70)
    print("TEST 3: Multiple Roots (Disconnected Components)")
    print("=" * 70)

    swc_content = """# Multiple disconnected trees
1 1 0 0 0 1.0 -1
2 3 1 0 0 0.5 1
3 3 2 0 0 0.4 2
10 1 10 0 0 1.0 -1
11 3 11 0 0 0.5 10
12 3 12 0 0 0.4 11
"""

    original = SWCModel.from_swc_file(swc_content.strip().split("\n"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        original.to_swc_file(temp_path)
        loaded = SWCModel.from_swc_file(temp_path)

        match = compare_models(original, loaded, "Multiple Roots")
        assert match, "Models should match for Multiple Roots"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_with_reconnections():
    """Test with cycle reconnections."""
    print("\n" + "=" * 70)
    print("TEST 4: With Reconnections")
    print("=" * 70)

    swc_content = """# Tree with reconnection
# CYCLE_BREAK reconnect 3 5
1 1 0 0 0 1.0 -1
2 3 1 0 0 0.5 1
3 3 2 0 0 0.4 2
4 3 1 1 0 0.5 1
5 3 2 0 0 0.4 4
"""

    original = SWCModel.from_swc_file(swc_content.strip().split("\n"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        original.to_swc_file(temp_path)
        loaded = SWCModel.from_swc_file(temp_path)

        match = compare_models(original, loaded, "With Reconnections")
        assert match, "Models should match for With Reconnections"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_after_make_cycle_connections():
    """Test that make_cycle_connections() returns nx.Graph, not SWCModel.

    This test verifies the fix: make_cycle_connections() now returns nx.Graph
    since the result may contain cycles and is no longer a valid SWC tree.
    """
    print("\n" + "=" * 70)
    print("TEST 5: make_cycle_connections() returns nx.Graph")
    print("=" * 70)

    swc_content = """# Tree with reconnection
# CYCLE_BREAK reconnect 3 5
1 1 0 0 0 1.0 -1
2 3 1 0 0 0.5 1
3 3 2 0 0 0.4 2
4 3 1 1 0 0.5 1
5 3 2 0 0 0.4 4
"""

    # Load and apply cycle connections
    model_before_merge = SWCModel.from_swc_file(swc_content.strip().split("\n"))
    result = model_before_merge.make_cycle_connections()

    print(f"\nBefore make_cycle_connections():")
    print(f"  Type: {type(model_before_merge).__name__}")
    print(f"  Nodes: {model_before_merge.number_of_nodes()}")
    print(f"  Edges: {model_before_merge.number_of_edges()}")

    print(f"\nAfter make_cycle_connections():")
    print(f"  Type: {type(result).__name__}")
    print(f"  Nodes: {result.number_of_nodes()}")
    print(f"  Edges: {result.number_of_edges()}")
    print(f"  Components: {nx.number_connected_components(result)}")

    # Verify the return type
    is_nx_graph = isinstance(result, nx.Graph)
    is_not_swcmodel = not isinstance(result, SWCModel)

    print(f"\nType checks:")
    print(f"  Is nx.Graph: {is_nx_graph}")
    print(f"  Is NOT SWCModel: {is_not_swcmodel}")

    # Verify it doesn't have SWCModel-specific attributes
    has_parents = hasattr(result, "_parents")
    print(f"  Has _parents attribute: {has_parents}")

    passed = is_nx_graph and is_not_swcmodel and not has_parents
    print(f"\n  Test result: {'PASS' if passed else 'FAIL'}")

    assert passed, "make_cycle_connections() should return nx.Graph, not SWCModel"


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# SWCModel Write/Read Bug Test Suite")
    print("#" * 70)

    results = []
    results.append(("Simple Tree", test_simple_tree()))
    results.append(("Branching Tree", test_branching_tree()))
    results.append(("Multiple Roots", test_multiple_roots()))
    results.append(("With Reconnections", test_with_reconnections()))
    results.append(
        ("After make_cycle_connections()", test_after_make_cycle_connections())
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\n⚠ Some tests failed - bug detected!")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
