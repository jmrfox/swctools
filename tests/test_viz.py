"""Tests for visualization functions in swctools.viz module."""

import pytest
from pathlib import Path
from swctools import SWCModel, plot_model, FrustaSet


@pytest.fixture
def test_output_dir():
    """Ensure test_outputs directory exists."""
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_swc_model():
    """Create a simple SWC model for testing."""
    swc_text = """# Simple test neuron
1 1 0.0 0.0 0.0 1.0 -1
2 3 0.0 0.0 5.0 0.8 1
3 3 0.0 0.0 10.0 0.6 2
4 3 0.0 5.0 10.0 0.5 3
5 3 0.0 10.0 10.0 0.4 4
6 3 5.0 0.0 10.0 0.5 3
7 3 10.0 0.0 10.0 0.4 6
"""
    return SWCModel.from_swc_file(swc_text.strip().split("\n"))


def test_plot_model_basic(sample_swc_model, test_output_dir):
    """Test basic plot_model with frusta and centroid."""
    output_path = test_output_dir / "plot_model_basic.html"

    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=True,
        title="Test Basic Model",
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_model_frusta_only(sample_swc_model, test_output_dir):
    """Test plot_model with only frusta (no centroid)."""
    output_path = test_output_dir / "plot_model_frusta_only.html"

    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=False,
        color="lightcoral",
        opacity=0.9,
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()


def test_plot_model_centroid_only(sample_swc_model, test_output_dir):
    """Test plot_model with only centroid (no frusta)."""
    output_path = test_output_dir / "plot_model_centroid_only.html"

    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=False,
        show_centroid=True,
        show_nodes=True,
        centroid_color="#ff0000",
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()


def test_plot_model_with_slider(sample_swc_model, test_output_dir):
    """Test plot_model with interactive radius scale slider."""
    output_path = test_output_dir / "plot_model_slider.html"

    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=True,
        slider=True,
        min_scale=0.5,
        max_scale=1.5,
        steps=11,
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()
    # Slider plots should be larger due to animation frames
    assert output_path.stat().st_size > 10000


def test_plot_model_with_tag_colors(sample_swc_model, test_output_dir):
    """Test plot_model with tag-based coloring."""
    output_path = test_output_dir / "plot_model_tag_colors.html"

    tag_colors = {
        1: "#ff0000",  # soma - red
        3: "#00ff00",  # dendrite - green
    }

    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=False,
        tag_colors=tag_colors,
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()


def test_plot_model_with_frusta_set(sample_swc_model, test_output_dir):
    """Test plot_model using pre-built FrustaSet."""
    output_path = test_output_dir / "plot_model_frustaset.html"

    # Pre-build frusta
    frusta = FrustaSet.from_swc_model(sample_swc_model, sides=12, end_caps=True)

    fig = plot_model(
        frusta=frusta,
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=True,
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()


def test_plot_model_scaled_radius(sample_swc_model, test_output_dir):
    """Test plot_model with static radius scaling."""
    output_path = test_output_dir / "plot_model_scaled.html"

    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=True,
        radius_scale=1.5,
        output_path=str(output_path),
        auto_open=False,
    )

    assert fig is not None
    assert output_path.exists()


def test_plot_model_without_output_path(sample_swc_model):
    """Test plot_model returns figure without saving to file."""
    fig = plot_model(
        swc_model=sample_swc_model,
        show_frusta=True,
        show_centroid=True,
    )

    assert fig is not None
    # Verify it's a plotly figure
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")
