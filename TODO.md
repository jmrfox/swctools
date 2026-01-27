# swctools TODO and roadmap

This document tracks the plan for building `swctools`, a Python toolbox for SWC parsing, modeling, analysis, geometry, and visualization. It is optimized for Jupyter workflows, with a graph-based core using NetworkX, computational geometry with NumPy, and interactive 3D via Plotly.

Status: planning. No public API is stable yet.

## Guiding principles

- Prefer simple, composable APIs that work well in notebooks.
- Separate parsing (I/O), data model, geometry, and visualization concerns.
- Make default plots beautiful yet configurable.
- Keep computational geometry numerically stable and reasonably fast (vectorize with NumPy where practical).

## Milestones and tasks

### M0 — Documentation and planning

- [x] Draft `README.md` with overview, features, and references
- [x] Draft `TODO.md` roadmap (this file)

### M1 — Project scaffolding

- [x] Decide initial package layout (initial modules in place)
  - `swctools/` package with modules:
    - [x] `io.py` (SWC reader/validator: `parse_swc`, `SWCRecord`, `SWCParseResult`)
    - [x] `model.py` (`SWCModel` DiGraph; `_graph_attributes`; `print_attributes`)
    - [x] `geometry.py` (`Frustum`, `FrustaSet`, `PointSet` spheres, helper math)
    - [x] `viz.py` (centroid, volumetric, `plot_model`, slider, overlay points)
    - [x] `config.py` (global Plotly config + `apply_layout`, equal-axes enforcement)
    - [ ] `animation.py` (time-dependent scalar visualization) — consolidated into `viz.py` via `animate_frusta_timeseries`
  - `data/` sample SWC files (small, clearly licensed)
  - `notebooks/` examples for Jupyter (user-authored; do not auto-create notebooks)
  - `tests/` unit tests
- [x] Initialize packaging with `pyproject.toml`
  - Core deps: `networkx`, `plotly`, `numpy`
  - Nice-to-have: `pandas` (tabular ops), `scipy` (optional geometry)
  - Dev deps: `pytest`, `ruff`, `black`, `mypy` (optional)
- [x] Add `LICENSE` (MIT) and set `license` metadata and classifiers in `pyproject.toml`
- [ ] Review runtime vs dev dependencies; move `pytest` to dev; make Jupyter optional; drop `matplotlib` unless needed
- [ ] Configure `uv` workflow (venv, add deps, run scripts)
- [ ] Set up linters/formatters and pre-commit hooks
- [ ] Add GitHub Actions CI (lint + test)

### M2 — Data model

- [x] Implement `SWCModel(networkx.DiGraph)`
  - Node key: SWC id `n` (int)
  - Node attrs: `t` (int), `x`, `y`, `z` (floats), `r` (float), optional `meta`
  - Directed edges: parent ➔ child; support forest (multiple roots)
  - Tracks cycles using header reconnection directives (`CYCLE_BREAK reconnect i j`)
  - Constructors: `from_swc_file(...)`, `from_parse_result(...)`
- [x] Graph metrics and printing helpers (`_graph_attributes`, `SWCModel.print_attributes`)
- [ ] Validation helpers
  - Ensure unique ids; parent either `-1` or existing id
  - Detect cycles, missing parents, invalid radii/coords

### M3 — SWC parser and I/O

- [x] Robust SWC reader
  - Skip comment lines (`#`), parse 7 columns: `n T x y z r parent`
  - Strong typing and error messages with line numbers
  - Accept path, file-like objects, and strings
- [x] Header annotations and reconnection
  - Parse lines like `# CYCLE_BREAK reconnect i j` into reconnection pairs
  - Validation of identical `(x, y, z, r)` available (configurable)
  - Reconnection pairs exposed on `SWCParseResult.reconnections`
- [ ] Validation layer (configurable strictness)
  - Enforce unique ids; check parent before child; allow out-of-order with fixup

### M4 — Centroid (skeleton) visualization

- [x] Build edge list from `SWCModel` suitable for `plotly.graph_objects.Scatter3d`
- [x] `plot_model(swc_model, ...) -> go.Figure`
  - Options: color by tag/depth/component, show markers vs lines, line width scaling by radius (optional)
  - Aspect ratio, axis labels, background theme presets
  - Tests (figure structure, traces present, basic property checks)

### M5 — Frustum geometry (frusta) and volumetric visualization

- [x] `Frustum` data structure
  - Oriented frustum between points `a` and `b` with radii `r_a`, `r_b`
  - Stable local frame construction for mesh generation
  - Optional end caps; degenerate handling (very short frusta, zero radius, etc.)

- [x] Mesh batching utilities
  - Generate vertices and faces for entire model
  - One `Mesh3d` per model (batched) vs per-frustum trade-offs
  - Color mapping by frustum id/tag or by external scalar array
  - Performance passes for moderate-sized morphologies
- [x] Add uniform radius scaling for volumetric mesh (`plot_frusta(radius_scale=...)`)
- [x] Overlay centroid + frusta (`plot_frusta_with_centroid`)
- [x] Interactive radius slider (`plot_frusta_slider`)
- [x] Master plotting entry point `plot_model(...)` (centroid + frusta + slider + points)
- [x] `PointSet` geometry (low-res spheres) and integration into `plot_model`
- [ ] Geometry tests (vertex counts, invariants, edge cases)

### M6 — Dynamics (time-dependent scalars on frusta)

- [ ] Data container for per-frustum time series `V_i(t)`
- [x] `animate_frusta_timeseries(frusta, values, ...)` for Plotly animations with slider and playback controls
- [x] Frustum ordering/remap utilities in `FrustaSet` (`edge_uvs`, `frustum_order_map`, `reordered`, `frustum_face_slices_map`) to align values with mesh order
- [ ] Example notebook with synthetic dynamics

### M7 — Examples and documentation

- [ ] Notebooks will be authored by the user; do not auto-create. Provide code snippets and recipe outlines in README/docstrings
- [ ] Document notebook outlines: centroid, volumetric, dynamics
- [ ] Add small sample SWC files under `data/` for user notebooks
- [x] Installation page (`docs/install.md`)
- [x] Visualization page with `plot_model`, slider, and `PointSet` (`docs/visualization.md`)
- [x] Update quick start to use `plot_model` (`docs/index.md`)
- [x] API reference via mkdocstrings (`docs/api.md`)
- [x] MkDocs nav + GitHub Pages workflow for docs

### M8 — Testing and quality

- [ ] Parser tests (happy path + failures)
- [ ] Reconnection tests: header parsing, merge invariants (equal `(x, y, z, r)`), union-of-edges, multi-pair groups, error cases
- [ ] Geometry tests (numerical stability, rotations)
- [ ] Visualization tests (figure JSON structure)
- [ ] Tests for `FrustaSet.scaled` (counts unchanged, geometry changes)
- [ ] Tests for `PointSet` (sphere vertex/face counts, `from_txt` parsing)
- [ ] Tests for `plot_model` slider frames and equal-axes layout
- [ ] CI green across supported Python versions

### M9 — Packaging and release

- [ ] Finalize metadata (license = MIT, authors, classifiers)
- [ ] Version v0.1.0 pre-release
- [ ] Publish examples and docs; consider Read the Docs or GitHub Pages

### M10 — Future enhancements (backlog)

- [ ] SWC+ support and annotations
- [ ] Morphometrics (branch order, path length, Sholl analysis)
- [ ] Smoothing/resampling along centerlines
- [ ] Export to common 3D formats (e.g., glTF)
- [ ] Import from NeuroMorpho or other repositories

## References

- [SWC specification (NeuronLand)](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
- [INCF SWC page](https://www.incf.org/swc)
- [SWC+ extension (future consideration)](https://neuroinformatics.nl/swcPlus/)

## Development notes

- This project uses `uv` for environment and dependency management.
- Typical workflow:
  - `uv venv` — create a virtual env
  - `uv add <deps>` — add dependencies
  - `uv run <cmd>` — run scripts (tests, examples)
- Plotly is chosen for interactive 3D rendering in notebooks; NetworkX underpins the graph model.
