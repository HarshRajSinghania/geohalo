# Changelog

## Unreleased

- Fix north/south reversal when resampling grids with descending source latitudes
  ([#4](https://github.com/campiohe/geohalo/issues/4)). `Resampler.compute` and
  `FactoredResampler.compute` now store source latitudes ascending and build their
  matrices in that order. The xarray helpers accept either latitude orientation.
- Share resampler cache entries between ascending and descending versions of the
  same source grid. Existing ascending-source entries remain valid; old
  descending-source entries are no longer used.
- Reject mismatched source coordinates in `resample_grid_with_matrix` with a
  `ValueError`.

Compatibility: code that directly multiplies `Resampler.transform_matrix` or calls
`FactoredResampler.apply_flat` must now flatten source values in ascending latitude
order, with longitude in `source_lon` order. Target coordinate order is unchanged.
