# swcviz — 3D Visualization of SWC Neuronal Morphologies

swcviz is a Python package for loading, analyzing, and visualizing neuronal morphologies stored in the SWC format. It is designed for interactive use in Jupyter notebooks and leverages Plotly for 3D visualization and NetworkX for graph-based topology.

Status: pre-alpha, actively evolving. Core parsing, models, geometry, and basic visualization are implemented.

Docs: [https://jmrfox.github.io/swcviz/](https://jmrfox.github.io/swcviz/)

Demo notebooks can be found in the `notebooks` directory.

## Features

- **SWC parser**: `parse_swc()` with robust error messages, header reconnection directives, iterable/file/string sources
- **Data models**:
  - `SWCModel` (`networkx.DiGraph`) for directed parent➔child topology with node attributes (`t, x, y, z, r`)
  - `GeneralModel` (`networkx.Graph`) for visualization; applies `# CYCLE_BREAK reconnect i j` merges (union-find)
  - Shared graph metrics via `_graph_attributes()` and `print_attributes()` helpers
- **Geometry**:
  - `Segment` dataclass and frustum meshing utilities (`frustum_mesh`, `batch_frusta`)
  - `FrustaSet.from_general_model()` to build a batched frusta mesh from a `GeneralModel`
  - `PointSet` for low-res spheres at arbitrary xyz points (for overlay markers)
- **Visualization**:
  - `plot_centroid(general_model, ...)` for skeleton plotting (`Scatter3d`)
  - `plot_frusta(frusta_set, ..., radius_scale=1.0)` for volumetric frusta rendering (`Mesh3d`)
  - `plot_frusta_with_centroid(gm, frusta, ...)` to overlay skeleton and mesh
  - `plot_frusta_slider(frusta, min_scale, max_scale, steps)` interactive radius scale slider
  - `plot_model(...)` master entry point combining centroid, frusta, slider, and `PointSet` overlays
  - Global config via `set_config(...)` (equal axes enforced by default, width/height, template)
- Roadmap: time-varying scalars and animations

## Design overview

- `SWCModel` (`networkx.DiGraph`)
  - Nodes keyed by SWC id `n`
  - Node attributes: `x`, `y`, `z`, `r`, `t` (type), and optional metadata
- `GeneralModel` (`networkx.Graph`)
  - Built from `SWCParseResult` with header-based reconnection merges
  - Node attributes preserved; provenance includes `merged_ids`, `lines`
- `Segment` dataclass and `FrustaSet` (batched frusta mesh)
- `PointSet` (batched spheres for overlay points)
- Visualization functions in `viz.py`: `plot_centroid`, `plot_frusta`, `plot_frusta_with_centroid`, `plot_frusta_slider`, `plot_model`

- Use in Jupyter:
  - Launch a notebook and import `swcviz`
  - Load or paste an SWC and use `GeneralModel`, `FrustaSet`, `plot_centroid`, `plot_frusta`

## Configuration (Plotly)

```python
from swcviz import set_config

# Enforce equal x/y/z scale globally (default True) and size
set_config(force_equal_axes=True, width=800, height=600)
```

## Acknowledgements

- Community conventions around SWC and tools in the ecosystem
- NetworkX and Plotly for the backbone of graph analysis and 3D visualization
