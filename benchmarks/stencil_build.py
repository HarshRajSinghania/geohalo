"""Synthetic issue #7 benchmark: detailed polygons through GeoJSON versus WKB.

Run: ``uv run python -m benchmarks.stencil_build``. No downloads are required.
Both paths include sorting, geometry serialization, extraction, CSR assembly,
and hashing; polygon generation is outside the timed region.
"""

import argparse
import hashlib
import time

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from geohalo import EmptyOverlapError, Stencil
from geohalo.geometry import cell_areas, ensure_ascending_lats, grid_digest, require_regular_grid


def _previous_compute(lats, lons, geoms):
    """The previous Stencil.compute path, including its scalar WKB digest loop."""
    lats, _ = ensure_ascending_lats(lats)
    lons = np.asarray(lons, dtype=np.float64)
    require_regular_grid(lats, "latitude")
    require_regular_grid(lons, "longitude")
    order = np.argsort([repr(key) for key in geoms.index])
    ordered = geoms.iloc[order]
    n_lat, n_lon = lats.size, lons.size
    raster = NumPyRasterSource(
        np.zeros((n_lat, n_lon)),
        xmin=float(lons[0] - (lons[1] - lons[0]) / 2),
        xmax=float(lons[-1] + (lons[-1] - lons[-2]) / 2),
        ymin=float(lats[0] - (lats[1] - lats[0]) / 2),
        ymax=float(lats[-1] + (lats[-1] - lats[-2]) / 2),
    )
    features = [
        {"type": "Feature", "geometry": shapely.geometry.mapping(geom), "properties": {"i": i}}
        for i, geom in enumerate(ordered.to_numpy())
    ]
    frame = exact_extract(raster, features, ["cell_id", "coverage"], output="pandas", include_cols=[])
    areas = cell_areas(lats, lons)
    rows, cols, weights = [], [], []
    for i, key in enumerate(ordered.index):
        ids = np.asarray(frame.iloc[i]["cell_id"], dtype=np.int64)
        coverage = np.asarray(frame.iloc[i]["coverage"], dtype=np.float64)
        if ids.size == 0:
            raise EmptyOverlapError(key)
        row, col = n_lat - 1 - ids // n_lon, ids % n_lon
        weight = coverage * areas[row, col]
        if weight.sum() <= 0:
            raise EmptyOverlapError(key)
        rows.append(np.full(ids.size, i, dtype=np.int64))
        cols.append(row * n_lon + col)
        weights.append(weight)
    matrix = sp.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(ordered), n_lat * n_lon),
    )
    ordered_for_digest = geoms.iloc[np.argsort([repr(key) for key in geoms.index])]
    geom_hash = hashlib.sha256()
    geom_hash.update(repr(tuple(geoms.index.names)).encode())
    for key, geom in zip(ordered_for_digest.index, ordered_for_digest.to_numpy(), strict=True):
        geom_hash.update(repr(key).encode())
        geom_hash.update(shapely.to_wkb(geom))
    digest = hashlib.sha256()
    digest.update(grid_digest(lats, lons))
    digest.update(b"sph")
    digest.update(geom_hash.digest())
    return Stencil(matrix, ordered.index, lats, lons, digest.digest())


def _measure(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons", type=int, default=3000)
    parser.add_argument("--quad-segs", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.polygons, args.quad_segs, args.repeats) < 1:
        parser.error("polygons, quad-segs, and repeats must be positive")
    rng = np.random.default_rng(0)
    centres = np.column_stack([rng.uniform(-75, -35, args.polygons), rng.uniform(-30, 5, args.polygons)])
    geoms = gpd.GeoSeries(
        [shapely.Point(x, y).buffer(rng.uniform(0.2, 1.0), quad_segs=args.quad_segs) for x, y in centres],
        index=[f"zone{i:05d}" for i in range(args.polygons)], crs="EPSG:4326",
    )
    lat = np.linspace(90, -90, 721)
    lon = np.arange(1440) * 0.25 - 180
    vertices = int(shapely.get_num_coordinates(geoms.to_numpy()).sum())
    print(f"{len(geoms):,} polygons; {vertices:,} vertices; medians of {args.repeats} builds", flush=True)
    previous, before = _measure(lambda: _previous_compute(lat, lon, geoms), args.repeats)
    print(f"Previous GeoJSON build: {before:.3f} s", flush=True)
    current, after = _measure(lambda: Stencil.compute(lat, lon, geoms), args.repeats)
    print(f"Current WKB build:      {after:.3f} s ({before / after:.1f}x faster)", flush=True)
    np.testing.assert_array_equal(current.occupancy_matrix.indptr, previous.occupancy_matrix.indptr)
    np.testing.assert_array_equal(current.occupancy_matrix.indices, previous.occupancy_matrix.indices)
    np.testing.assert_array_equal(current.occupancy_matrix.data, previous.occupancy_matrix.data)
    np.testing.assert_array_equal(current.row_sums, previous.row_sums)
    if current.digest != previous.digest or not current.keys.equals(previous.keys):
        raise AssertionError("stencil keys or digest changed")
    print("CSR matrices, row sums, keys, and digests match exactly.")


if __name__ == "__main__":
    main()
