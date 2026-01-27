# swctools — SWC parsing, modeling, analysis, geometry, and visualization

swctools is a Python toolbox for loading, parsing, modeling, analyzing, and visualizing neuronal morphologies stored in the SWC format. It is designed for interactive use in Jupyter notebooks and provides a graph-based core (NetworkX), computational geometry utilities (NumPy), and interactive 3D visualization (Plotly).

Status: pre-alpha, actively evolving. Core parsing, models, geometry, and basic visualization are implemented.

Docs: [https://jmrfox.github.io/swctools/](https://jmrfox.github.io/swctools/)

Demo notebooks can be found in the `notebooks` directory.

## Features

- **SWC parser**: `parse_swc()` with robust error messages, header reconnection directives, iterable/file/string sources
- **Data model**:
  - `SWCModel` (`networkx.Graph`) represents valid SWC directed tree structures with undirected storage and an internal parent map for original SWC directed tree relations
  - `make_cycle_connections()` applies `# CYCLE_BREAK reconnect i j` merges (union-find) and returns `nx.Graph` (not `SWCModel`) since the result may contain cycles
  - Shared graph metrics via `_graph_attributes()` and `print_attributes()` helpers
- **Geometry**:
  - `Frustum` dataclass and frustum meshing utilities (`frustum_mesh`, `batch_frusta`)
  - `FrustaSet.from_swc_model()` to build a batched frusta mesh from an `SWCModel`
  - `PointSet` for low-res spheres at arbitrary xyz points (for overlay markers)
- **Visualization**:
  - `plot_centroid(model, ...)` for skeleton plotting (`Scatter3d`)
  - `plot_frusta(frusta_set, ..., radius_scale=1.0)` for volumetric frusta rendering (`Mesh3d`)
  - `plot_frusta_with_centroid(model, frusta, ...)` to overlay skeleton and mesh
  - `plot_frusta_slider(frusta, min_scale, max_scale, steps)` interactive radius scale slider
  - `plot_model(...)` master entry point combining centroid, frusta, slider, and `PointSet` overlays
  - `animate_frusta_timeseries(frusta, values, ...)` animate time-dependent per-frustum scalars V_i(t) with a slider and playback controls
  - Global config via `set_config(...)` (equal axes enforced by default, width/height, template)
- Roadmap: morphometrics and analyses, I/O conversions, time-varying scalars and animations

## Design overview

- `SWCModel` (`networkx.Graph` with parent map) **— represents valid SWC directed tree structures only**
  - Nodes keyed by SWC id `n`
  - Node attributes: `x`, `y`, `z`, `r` (radius), `t` (tag), and optional metadata
  - `_parents` preserves the original directed parent of each node; `roots()`, `parent_of()`, `path_to_root()` use this map
  - Methods like `to_swc_file()` rely on the tree structure and only work correctly for valid SWC trees
  - `make_cycle_connections()` merges reconnection pairs and **returns `nx.Graph`** (not `SWCModel`) since the result may contain cycles
- `Frustum` dataclass and `FrustaSet` (batched frusta mesh)
- `FrustaSet` ordering utilities: `frustum_order_map()`, `reordered(...)`, and `frustum_face_slices_map()` to help align external per-frustum arrays with mesh order
- `PointSet` (batched spheres for overlay points)
- Visualization functions in `viz.py`: `plot_centroid`, `plot_frusta`, `plot_frusta_with_centroid`, `plot_frusta_slider`, `plot_model`

- Use in Jupyter:
  - Launch a notebook and import `swctools`
  - Load or paste an SWC and use `SWCModel`, `FrustaSet`, `plot_centroid`, `plot_frusta`

## Configuration (Plotly)

```python
from swctools import set_config

# Enforce equal x/y/z scale globally (default True) and size
set_config(force_equal_axes=True, width=800, height=600)
```

## Acknowledgements

- Community conventions around SWC and tools in the ecosystem
- NetworkX and Plotly for the backbone of graph analysis and 3D visualization
