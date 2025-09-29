# swctools

Jupyter-first toolbox for SWC parsing, modeling, analysis, geometry, and 3D visualization (NetworkX, NumPy, Plotly).

- Parse SWC into directed `SWCModel` and undirected `GeneralModel`
- Visualize skeletons (centroid) with `plot_centroid`
- Build volumetric frusta meshes with `FrustaSet`, or use the master `plot_model`

## Quick start

```python
from swctools import parse_swc, GeneralModel, FrustaSet, PointSet, plot_model, set_config

# Optional: global viz settings (equal axes enforced by default)
set_config(width=800, height=600)

swc = """
# CYCLE_BREAK reconnect 2 3
1 1 0 0 0 1 -1
2 3 2 0 0 0.5 1
3 3 2 0 0 0.5 1
4 3 3 0 0 0.4 2
""".strip()

gm = GeneralModel.from_swc_file(swc, strict=True, validate_reconnections=True)
fr = FrustaSet.from_general_model(gm, sides=16, end_caps=False)

# Optional overlay points (as small spheres)
ps = PointSet.from_points([(0,0,0), (3,0,0)], base_radius=0.05)

# One-call visualization
fig = plot_model(gm=gm, frusta=fr, show_centroid=True, point_set=ps, radius_scale=0.8)
fig.show()

# Interactive radius slider (0..1)
fig_slider = plot_model(gm=gm, frusta=fr, slider=True, min_scale=0.0, max_scale=1.0, steps=21)
fig_slider.show()
```

See the Visualization and API reference pages for details.
