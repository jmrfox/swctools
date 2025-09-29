"""swctools package scaffolding.

Public API is evolving; currently exposes SWC parsing, models, geometry, and visualization utilities.
"""

from .io import SWCRecord, SWCParseResult, parse_swc
from .model import SWCModel, GeneralModel
from .geometry import Segment, frustum_mesh, batch_frusta, FrustaSet, PointSet
from .viz import plot_centroid, plot_frusta, plot_frusta_with_centroid, plot_frusta_slider, plot_model
from .config import get_config, set_config, apply_layout

__all__ = [
    "SWCRecord",
    "SWCParseResult",
    "parse_swc",
    "SWCModel",
    "GeneralModel",
    "Segment",
    "frustum_mesh",
    "batch_frusta",
    "PointSet",
    "FrustaSet",
    "plot_centroid",
    "plot_frusta",
    "plot_frusta_with_centroid",
    "plot_frusta_slider",
    "plot_model",
    "get_config",
    "set_config",
    "apply_layout",
]
