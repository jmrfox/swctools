"""swctools package scaffolding.

Public API is evolving; currently exposes SWC parsing, models, geometry, and visualization utilities.
"""

import logging

from .io import SWCRecord, SWCParseResult, parse_swc
from .model import SWCModel
from .geometry import Segment, frustum_mesh, batch_frusta, FrustaSet, PointSet
from .viz import (
    plot_centroid,
    plot_frusta,
    plot_frusta_with_centroid,
    plot_frusta_slider,
    plot_model,
    plot_frusta_timeseries,
)
from .config import get_config, set_config, apply_layout
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SWCRecord",
    "SWCParseResult",
    "parse_swc",
    "SWCModel",
    "Segment",
    "frustum_mesh",
    "batch_frusta",
    "PointSet",
    "FrustaSet",
    "plot_centroid",
    "plot_frusta",
    "plot_frusta_with_centroid",
    "plot_frusta_slider",
    "plot_frusta_timeseries",
    "plot_model",
    "get_config",
    "set_config",
    "apply_layout",
]
