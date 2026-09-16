"""Issue #6: actual Zarr v3 chunk reads for full-grid and restricted reductions.

Run ``uv run python -m benchmarks.chunk_reads``. Uses an in-memory Zarr store;
there are no external datasets or network requests. Building the synthetic data,
store, and operators is excluded from the measurements.
"""

import argparse
import gc
import time
import tracemalloc

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from zarr.storage import MemoryStore, WrapperStore

from geohalo import ReduceOperator, RestrictedOperator, Stencil, reduce_with_operator, reduce_with_restricted_operator


class CountingStore(WrapperStore):
    def __init__(self, store, reads=None):
        super().__init__(store)
        self.reads = [] if reads is None else reads

    def _with_store(self, store):
        return type(self)(store, self.reads)

    def _record(self, key, value):
        if key.startswith("field/c/"):
            self.reads.append((key, 0 if value is None else len(value)))
        return value

    async def get(self, key, prototype, byte_range=None):
        return self._record(key, await super().get(key, prototype, byte_range))

    def get_sync(self, key, *, prototype=None, byte_range=None):
        return self._record(key, super().get_sync(key, prototype=prototype, byte_range=byte_range))

    async def _get_many(self, requests):
        for key, prototype, byte_range in requests:
            yield key, await self.get(key, prototype, byte_range)


def _measure(store, apply, repeats):
    times = []
    for track_memory in [False] * repeats + [True]:
        with xr.open_zarr(store, chunks=None, consolidated=False) as ds:
            store.reads.clear()
            gc.collect()
            if track_memory:
                tracemalloc.start()
            start = time.perf_counter()
            result = apply(ds.field)
            elapsed = time.perf_counter() - start
            if track_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            else:
                times.append(elapsed)
            reads = len(store.reads)
            unique = len({key for key, _ in store.reads})
            received = sum(size for _, size in store.reads)
            if reads != unique:
                raise AssertionError("a data chunk was fetched more than once")
    return result, float(np.median(times)), reads, received, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    lat = np.linspace(90, -90, 721)
    lon = np.arange(1440) * 0.25 - 180
    values = np.random.default_rng(42).random((args.steps, lat.size, lon.size), dtype=np.float32)
    grid = xr.DataArray(
        values, dims=("step", "latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon}, name="field",
    )
    boxes = [shapely.box(-60 + i, -20, -58 + i, -18) for i in range(10)]
    boxes.extend(shapely.box(-95 + i, 40, -93 + i, 42) for i in range(5))
    geoms = gpd.GeoSeries(boxes, index=[f"zone{i}" for i in range(len(boxes))], crs="EPSG:4326")
    stencil = Stencil.compute(lat, lon, geoms)
    operator = ReduceOperator.compute(stencil, lat, lon)
    store = CountingStore(MemoryStore())
    grid.to_dataset().to_zarr(
        store, encoding={"field": {"chunks": (6, 64, 64)}}, zarr_format=3, consolidated=False,
    )
    del grid, values
    with xr.open_zarr(store, chunks=None, consolidated=False) as ds:
        plan = RestrictedOperator.from_grid(operator, ds.field)
    cells = sum((r.stop - r.start) * (c.stop - c.start) for r, c in plan.windows)
    print(
        f"{args.steps} float32 steps; 721x1440 grid; 15 polygons; {len(plan.windows)} windows; "
        f"{cells:,} / {lat.size * lon.size:,} cells per step read", flush=True,
    )
    expected, *full = _measure(store, lambda grid: reduce_with_operator(grid, operator), args.repeats)
    actual, *restricted = _measure(
        store, lambda grid: reduce_with_restricted_operator(grid, plan), args.repeats,
    )
    xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for name, (seconds, reads, received, peak) in (("Full grid", full), ("Restricted", restricted)):
        print(
            f"{name}: {seconds:.4f} s; {reads} chunk reads; {received / 2**20:.3f} MiB payload; "
            f"{peak / 2**20:.3f} MiB extra peak", flush=True,
        )
    print(f"Maximum absolute result difference: {float(abs(actual - expected).max()):.3g}")
    print("Times are uninstrumented medians; peaks use a separate tracemalloc run, not process RSS.")


if __name__ == "__main__":
    main()
