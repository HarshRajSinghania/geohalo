"""Synthetic issue #5 benchmark, with no external data or services.

Run: ``uv run python -m benchmarks.operator_memory``.
Allocation peaks exclude input/operator construction but include output and any
application caches created on the first call. Timings are warmed medians.
"""

import argparse
import time
import tracemalloc

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

from geohalo import ReduceOperator, Stencil, reduce_with_operator


def _previous_product(grid, operator):
    """The pre-fix application path: sort data, flatten, and multiply in one batch."""
    if grid.latitude[0] > grid.latitude[-1]:
        grid = grid.sortby("latitude")
    values = grid.transpose("step", "latitude", "longitude").to_numpy()
    flat = values.reshape(values.shape[0], -1)
    return (flat @ operator.matrix.T) / operator.row_sums


def _measure(fn, repeats):
    peaks = []
    for _ in range(2):
        tracemalloc.start()
        try:
            result = np.asarray(fn())
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return result, peaks, np.median(times) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")

    lat = np.linspace(-90, 90, 721)
    lon = np.arange(1440) * 0.25 - 180
    polygons = gpd.GeoSeries(
        [shapely.box(x, y, x + 3, y + 3) for x in range(-75, -35, 4) for y in range(-30, 5, 4)],
        crs="EPSG:4326",
    )
    stencil = Stencil.compute(lat, lon, polygons)
    values = np.random.default_rng(0).random((args.steps, lat.size, lon.size), dtype=np.float32)
    print(f"Input: {values.nbytes / 2**20:.2f} MiB float32; {args.steps} slices; {stencil.occupancy_matrix.nnz} nnz")
    print("| Latitude | Method | First peak MiB | Warm peak MiB | Median ms | Max abs difference |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for descending in (False, True):
        operator = ReduceOperator.compute(stencil, lat, lon)
        grid = xr.DataArray(
            values, dims=("step", "latitude", "longitude"),
            coords={"latitude": lat[::-1] if descending else lat, "longitude": lon},
        )
        order = "descending" if descending else "ascending"
        expected, peaks, ms = _measure(lambda g=grid, o=operator: _previous_product(g, o), args.repeats)
        print(f"| {order} | Previous | {peaks[0] / 2**20:.3f} | {peaks[1] / 2**20:.3f} | {ms:.2f} | — |")
        actual, peaks, ms = _measure(lambda g=grid, o=operator: reduce_with_operator(g, o), args.repeats)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        difference = float(np.max(np.abs(actual - expected)))
        print(f"| {order} | Current | {peaks[0] / 2**20:.3f} | {peaks[1] / 2**20:.3f} | {ms:.2f} | {difference:.1e} |")


if __name__ == "__main__":
    main()
