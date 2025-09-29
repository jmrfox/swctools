"""Global visualization configuration for swcviz Plotly figures.

Use `set_config(...)` to override defaults in notebooks/apps, and
`apply_layout(fig, title=...)` to apply them to a Plotly Figure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass
class VizConfig:
    # Figure size
    width: int = 800
    height: int = 600
    # Plotly template
    template: str = "plotly_white"
    # Scene aspect
    # 'cube' gives equal aspect in x/y/z and tends to look less stretched in notebooks
    scene_aspectmode: str = "auto"  # alternatives: 'data', 'auto', 'manual'
    # Always enforce equal scale on x/y/z (uses aspectmode='data'). If True, overrides scene_aspectmode above.
    force_equal_axes: bool = True
    scene_aspectratio: Dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )  # used only if aspectmode == 'manual'
    # Margins and legend
    margin: Dict[str, int] = field(
        default_factory=lambda: {"l": 0, "r": 0, "t": 40, "b": 0}
    )
    showlegend: bool = False


_config = VizConfig()


def get_config() -> VizConfig:
    """Return the current visualization configuration (live object)."""
    return _config


def set_config(**kwargs: Any) -> None:
    """Update global visualization configuration.

    Example:
        set_config(width=800, height=600, scene_aspectmode="cube")
    """
    for k, v in kwargs.items():
        if not hasattr(_config, k):
            raise AttributeError(f"Unknown viz config key: {k}")
        setattr(_config, k, v)


def apply_layout(fig, *, title: str | None = None) -> None:
    """Apply global layout defaults to a Plotly figure in-place."""
    # Determine aspect mode (force equal axes if requested)
    aspectmode = "data" if _config.force_equal_axes else _config.scene_aspectmode

    fig.update_layout(
        width=_config.width,
        height=_config.height,
        template=_config.template,
        margin=_config.margin,
        showlegend=_config.showlegend,
        scene_aspectmode=aspectmode,
    )
    # Only used when manual aspect is requested
    if aspectmode == "manual":
        fig.update_layout(scene=dict(aspectratio=_config.scene_aspectratio))
    if title is not None:
        fig.update_layout(title=title)
