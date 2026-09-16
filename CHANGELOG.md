# Changelog

## Unreleased

- Add `geohalo.geometry.polygon_areas` for spherical polygon area on the same
  sphere and radius as `cell_areas` ([#21](https://github.com/campiohe/geohalo/issues/21)).
  Edges are treated as straight in longitude; the latitude integral is of
  `sin φ`, which is exact for a lat/lon box.

