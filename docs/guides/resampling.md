# Resampling grids

Resampling is a **first-class, reusable** operation in geohalo — not just a hidden step
inside `reduce`. If you want the refined (or coarsened) field itself, `resample_grid`
gives it to you, backed by the same [mean-preserving](../concepts/downscaling.md) sparse
transform.

## One call

```python
import geohalo as ghl

fine = ghl.resample_grid(da, target_resolution=0.05, iterations=3)
```

It works in **either direction** — a smaller `target_resolution` refines, a larger one
coarsens — and mean-preservation is exact wherever geometrically possible (always when
refining; coarsening can't preserve a source cell that has no target child).

`resample_grid` accepts an `xr.DataArray` or an `xr.Dataset` (every spatial data variable
is resampled, the rest pass through) and preserves all non-spatial dims.

## Separating build from apply

`resample_grid` builds a `Resampler` and applies it in one shot. To reuse the transform
across many slices, build it once and apply with `resample_grid_with_matrix`:

```python
import geohalo as ghl

cache = ghl.LocalCache("./.geohalo-cache")
t_lat, t_lon = ghl.geometry.target_coords_from_resolution(da.latitude.values, da.longitude.values, 0.05)

resampler = cache.get_or_compute_resampler(
    da.latitude.values, da.longitude.values, t_lat, t_lon, iterations=3,
)
fine = ghl.resample_grid_with_matrix(da, resampler)
```

The `Resampler` is value-independent and [cacheable](caching.md) — built once per
`(source grid, target grid, iterations)`.

## Choosing `iterations`

`iterations` controls how far the [mean-preserving correction](../concepts/downscaling.md)
reaches across the grid. Every value preserves each parent cell's mean exactly; higher
counts trade build cost for smoothness.

| `iterations` | Character                                    | Cost              |
| ------------ | -------------------------------------------- | ----------------- |
| `1` (default) | classic operator, blocky but exact          | cheapest          |
| `2`–`3`      | visibly smoother, still sub-second to build  | moderate          |
| higher       | smoother still; transform fills in further   | grows with reach  |

```python
coarse = ghl.resample_grid(da, target_resolution=0.5)          # coarsen, iters=1
smooth = ghl.resample_grid(da, target_resolution=0.05, iterations=4)
```

## Cost note: materialised vs fused

`resample_grid` materialises the full transform \(\mathbf{T}\), because you asked for the
**field**. That matrix can be large for a big refinement (hundreds of MB at fine target
resolutions and high iteration counts).

If the refined field is only a stepping stone to **per-polygon values**, you don't need
\(\mathbf{T}\) at all — go through `reduce(..., target_resolution=…)`, which
[fuses the resample into the stencil](../concepts/reduce-operator.md) and never builds
the fine grid:

```python
# materialises the fine field (you want the grid)
fine = ghl.resample_grid(da, target_resolution=0.05, iterations=3)

# never materialises the fine field (you want polygon values)
out = ghl.reduce(da, geoms, target_resolution=0.05, resample_iterations=3)
```

## Handling descending latitudes

ECMWF and many GRIB products ship latitudes **descending** (90 → −90). geohalo's
`Resampler.compute` and `FactoredResampler.compute` canonicalise source latitudes
to ascending order when building their matrices. The xarray apply helpers map
matrix columns to the input's stored latitude order, so you do not need to flip
anything yourself or copy a descending grid into ascending order.
Ascending and descending versions of the same source grid share one cache entry.

`resample_grid_with_matrix` checks both latitude and longitude coordinates against
the resampler's source grid and raises `ValueError` on a mismatch, even when the
grid shapes are identical. Target coordinates keep the order supplied when the
resampler was built.

If you multiply `Resampler.transform_matrix` directly, or call
`FactoredResampler.apply_flat`, flatten source values with **ascending latitude**
and longitude in `source_lon` order. This corrects the convention in 1.1.0, where
matrices built from descending latitudes expected descending values, causing the
xarray helpers to flip north and south.

## Application memory

For large source grids, the xarray helpers apply the transform one batch slice at
a time. Small contiguous grids use bounded batches. This avoids copying or
casting every source slice at once while preserving the matrix's arithmetic
precision. The dense target output still needs to fit in memory, alongside
temporary storage for one source/target slice and cached matrix indices for
descending latitudes. Lazy inputs are still loaded in full before application.
