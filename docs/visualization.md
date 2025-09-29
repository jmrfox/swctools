# Visualization

This page summarizes the visualization entry points and options. Visualization is one part of the broader `swctools` toolbox for SWC parsing, modeling, analysis, and geometry.

## Global configuration

Use `set_config(...)` to set defaults for all figures:

```python
from swctools import set_config
set_config(width=800, height=600, force_equal_axes=True, template="plotly_white")
```

- `force_equal_axes` ensures identical units along x/y/z (applies `aspectmode="data"`).
- Other options: `scene_aspectmode` (if you disable `force_equal_axes`), margins, legend.

## Centroid (skeleton)

```python
from swctools import plot_centroid
fig = plot_centroid(gm, show_nodes=True)
fig.show()
```

## Volumetric frusta

```python
from swctools import plot_frusta, FrustaSet
fr = FrustaSet.from_general_model(gm, sides=16, end_caps=False)
fig = plot_frusta(fr, radius_scale=0.8)
fig.show()
```

## Overlay centroid + frusta

```python
from swctools import plot_frusta_with_centroid
fig = plot_frusta_with_centroid(gm, fr, radius_scale=1.0)
fig.show()
```

## Master function: plot_model

```python
from swctools import plot_model

fig = plot_model(
    gm=gm,
    frusta=fr,              # optional; built from gm if omitted
    show_frusta=True,
    show_centroid=True,
    radius_scale=0.8,       # static scale for radii
)
fig.show()
```

### Interactive slider for radius scale

```python
fig = plot_model(
    gm=gm,
    frusta=fr,
    slider=True,            # enables a 0..1 slider by default
    min_scale=0.0,
    max_scale=1.0,
    steps=21,
)
fig.show()
```

## Overlay arbitrary points as spheres (PointSet)

```python
from swctools import PointSet

# Build from a list of xyz points
ps = PointSet.from_points([(0, 0, 0), (3, 0, 0)], base_radius=0.05)
# or from a text file with "x y z" per line (comments starting with '#')
ps = PointSet.from_txt("points.txt", base_radius=0.05)

fig = plot_model(gm=gm, frusta=fr, point_set=ps, point_size=1.5, point_color="crimson")
fig.show()
```

Notes:

- `point_size` scales the `base_radius` uniformly.
- Spheres are low-res by default (`stacks=6`, `slices=12`) to keep rendering fast.
