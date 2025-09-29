# swcviz TODO and roadmap

This document tracks the plan for building `swcviz`, a Python library for 3D visualization of neuronal morphologies in SWC format, optimized for Jupyter with Plotly and a graph-based core using NetworkX.

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
  - `swcviz/` package with modules:
    - [x] `io.py` (SWC reader/validator: `parse_swc`, `SWCRecord`, `SWCParseResult`)
    - [x] `model.py` (`SWCModel` DiGraph; `GeneralModel` Graph; `_graph_attributes`; `print_attributes`)
    - [x] `geometry.py` (`Segment` frustum construction, `PointSet` spheres, helper math)
    - [x] `viz.py` (centroid, volumetric, `plot_model`, slider, overlay points)
    - [x] `config.py` (global Plotly config + `apply_layout`, equal-axes enforcement)
    - [ ] `animation.py` (time-dependent scalar visualization)
  - `data/` sample SWC files (small, clearly licensed)
  - `notebooks/` examples for Jupyter (user-authored; do not auto-create notebooks)
  - `tests/` unit tests
- [ ] Initialize packaging with `pyproject.toml`
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
  - Node attrs: `t` (type), `x`, `y`, `z` (floats), `r` (float), optional `meta`
  - Directed edges: parent ➔ child; support forest (multiple roots)
- [x] Implement `GeneralModel(networkx.Graph)` for visualization and reconnection support
  - Undirected graph built with header reconnections (`CYCLE_BREAK reconnect i j`)
  - Constructors: `from_parse_result(...)`, `from_swc_file(...)`
  - Merge policy: union-find over reconnect pairs (requires identical `(x, y, z, r)`), provenance via `merged_ids`, `lines`
- [x] Graph metrics and printing helpers (`_graph_attributes`, `SWCModel.print_attributes`, `GeneralModel.print_attributes`)
- [ ] Convenience methods
  - `from_swc(path_or_buffer)` classmethod
  - `to_dataframe()` for easy inspection
  - `segments()` iterator yielding parent-child pairs and attributes
  - `to_general_model(reconnect: bool = True)` to apply header-based reconnections
  - `roots()`, `components()`, `depths()` utilities
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
  - Transitive merges supported in `GeneralModel` via union-find
  - Reconnection pairs exposed on `SWCParseResult.reconnections`
- [ ] Validation layer (configurable strictness)
  - Enforce unique ids; check parent before child; allow out-of-order with fixup

### M4 — Centroid (skeleton) visualization

- [x] Build edge list from `GeneralModel` suitable for `plotly.graph_objects.Scatter3d`
- [x] `plot_centroid(general_model, ...) -> go.Figure`
  - Options: color by type/depth/component, show markers vs lines, line width scaling by radius (optional)
  - Aspect ratio, axis labels, background theme presets
  - Tests (figure structure, traces present, basic property checks)

### M5 — Segment geometry (frusta) and volumetric visualization

- [x] `Segment` data structure
  - Oriented frustum between points `a` and `b` with radii `r_a`, `r_b`
  - Stable local frame construction for mesh generation
  - Optional end caps; degenerate handling (very short segments, zero radius)

- [x] Mesh batching utilities
  - Generate vertices and faces for entire model
  - One `Mesh3d` per model (batched) vs per-segment trade-offs
  - Color mapping by segment id/type or by external scalar array
  - Performance passes for moderate-sized morphologies
- [x] Add uniform radius scaling for volumetric mesh (`plot_frusta(radius_scale=...)`)
- [x] Overlay centroid + frusta (`plot_frusta_with_centroid`)
- [x] Interactive radius slider (`plot_frusta_slider`)
- [x] Master plotting entry point `plot_model(...)` (centroid + frusta + slider + points)
- [x] `PointSet` geometry (low-res spheres) and integration into `plot_model`
- [ ] Geometry tests (vertex counts, invariants, edge cases)

### M6 — Dynamics (time-dependent scalars on segments)

- [ ] Data container for per-segment time series `V_i(t)`
- [ ] `animate_segments(model, values, times, ...)` for Plotly animations
- [ ] Color scales, legend, and playback controls
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
- [ ] FAQ including `GeneralModel` reconnection semantics and usage tips

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
