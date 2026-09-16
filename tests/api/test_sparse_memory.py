"""Numerical and allocation regressions for applying sparse grid operators."""

import tracemalloc
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
import xarray as xr

from geohalo import LocalCache, ReduceOperator, Resampler, Stencil, reduce_with_operator, resample_grid_with_matrix


def _operator(n_lat, n_lon):
    rng = np.random.default_rng(42)
    # Include negative coefficients, as occur in fused resampling operators.
    matrix = sp.csr_matrix(
        (rng.uniform(-0.25, 1.0, 128), (np.repeat(np.arange(4), 32), rng.integers(n_lat * n_lon, size=128))),
        shape=(4, n_lat * n_lon),
    )
    return ReduceOperator(
        matrix=matrix, row_sums=np.asarray(matrix.sum(axis=1)).ravel(), keys=pd.Index(list("abcd")),
        source_lat=np.linspace(-80, 80, n_lat), source_lon=np.linspace(-170, 170, n_lon),
        iterations=1, digest=b"test",
    )


@pytest.mark.parametrize("shape", [(4, 5), (101, 103)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "fortran"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_reduce_matches_original_product(shape, layout, dtype, descending, how):
    op = _operator(*shape)
    values = np.random.default_rng(0).uniform(1, 10, size=(2, 3, *shape)).astype(dtype)
    if layout == "strided":
        values = np.ascontiguousarray(values.transpose(2, 0, 3, 1)).transpose(1, 3, 0, 2)
    elif layout == "fortran":
        values = np.asfortranarray(values)
    lat = op.source_lat[::-1] if descending else op.source_lat
    grid = xr.DataArray(
        values, dims=("member", "step", "lat", "lon"),
        coords={"member": [0, 1], "step": [0, 1, 2], "lat": lat, "lon": op.source_lon, "model": "test"},
        name="temperature", attrs={"units": "K"},
    )
    original = grid.sortby("lat").to_numpy().reshape(6, -1) @ op.matrix.T
    if how == "mean":
        original /= op.row_sums
    actual = reduce_with_operator(grid, op, how=how, lat_dim="lat", lon_dim="lon")
    np.testing.assert_allclose(actual.values, original.reshape(2, 3, 4), rtol=1e-12, atol=1e-12)
    assert actual.dtype == np.float64
    assert actual.dims == ("member", "step", "geom")
    assert actual.attrs == grid.attrs
    assert actual.name == grid.name
    assert actual.model == grid.model


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_reduce_large_batch_memory_is_below_one_source_slice(descending, layout):
    op = _operator(721, 1440)
    values = np.ones((24, 721, 1440), dtype=np.float32)  # ~95 MiB, allocated before tracing
    if layout == "strided":
        values = values[:, :, ::-1]
    lat = op.source_lat[::-1] if descending else op.source_lat
    grid = xr.DataArray(
        values, dims=("step", "latitude", "longitude"),
        coords={"latitude": lat, "longitude": op.source_lon},
    )
    # Include the first call's preparation as well as steady-state application.
    for _ in range(2):
        tracemalloc.start()
        try:
            actual = reduce_with_operator(grid, op)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        np.testing.assert_allclose(actual.values, 1.0, rtol=1e-12)
        assert peak < values[0].nbytes


@pytest.mark.parametrize("descending", [False, True])
def test_resample_large_batch_memory_is_bounded_by_output_and_slice(descending):
    lat, lon = np.linspace(-80, 80, 201), np.linspace(-170, 170, 251)
    resampler = Resampler.compute(lat, lon, lat, lon)
    values = np.random.default_rng(2).random((24, lat.size, lon.size), dtype=np.float32)
    source_lat = lat[::-1] if descending else lat
    grid = xr.DataArray(
        values, dims=("step", "latitude", "longitude"),
        coords={"latitude": source_lat, "longitude": lon},
    )
    for _ in range(2):
        tracemalloc.start()
        try:
            actual = resample_grid_with_matrix(grid, resampler)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        expected = values[:, ::-1] if descending else values
        np.testing.assert_array_equal(actual.values, expected)
        # Allow per-slice float64 input/output and sparse orientation indices,
        # but no extra allocation proportional to the number of batch slices.
        assert peak < actual.nbytes + 8 * values[0].nbytes


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("descending", [False, True])
def test_resample_large_strided_grid_matches_original(iterations, descending):
    lat, lon = np.linspace(-80, 80, 101), np.linspace(-170, 170, 103)
    target_lat, target_lon = lat[::2], lon[::2]
    resampler = Resampler.compute(lat, lon, target_lat, target_lon, iterations=iterations)
    values = np.random.default_rng(1).random((2, 3, lat.size, lon.size), dtype=np.float32)
    values = np.asfortranarray(values)
    grid = xr.DataArray(
        values, dims=("member", "step", "latitude", "longitude"),
        coords={"latitude": lat[::-1] if descending else lat, "longitude": lon},
    )
    expected = grid.sortby("latitude").to_numpy().reshape(6, -1) @ resampler.transform_matrix.T
    actual = resample_grid_with_matrix(grid, resampler)
    np.testing.assert_allclose(
        actual.values, expected.reshape(2, 3, target_lat.size, target_lon.size), rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("descending", [False, True])
def test_reduce_large_operator_touching_every_source_cell(descending):
    op = _operator(101, 103)
    columns = np.arange(op.matrix.shape[1])
    matrix = sp.csr_matrix(
        (np.linspace(1.0, 2.0, columns.size), (columns % 4, columns)), shape=op.matrix.shape,
    )
    op = replace(op, matrix=matrix, row_sums=np.asarray(matrix.sum(axis=1)).ravel())
    values = np.random.default_rng(4).random((3, 101, 103), dtype=np.float32)
    grid = xr.DataArray(
        values, dims=("step", "latitude", "longitude"),
        coords={"latitude": op.source_lat[::-1] if descending else op.source_lat, "longitude": op.source_lon},
    )
    expected = (grid.sortby("latitude").to_numpy().reshape(3, -1) @ matrix.T) / op.row_sums
    actual = reduce_with_operator(grid, op)
    np.testing.assert_allclose(actual.values, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", [(4, 5), (101, 103)])
@pytest.mark.parametrize("batch_size", [0, 1, 2000])
def test_reduce_empty_single_and_many_slices(shape, batch_size):
    op = _operator(*shape)
    values = np.broadcast_to(np.float32(3.0), (batch_size, *shape)).copy()
    grid = xr.DataArray(
        values, dims=("step", "latitude", "longitude"),
        coords={"latitude": op.source_lat, "longitude": op.source_lon},
    )
    actual = reduce_with_operator(grid, op)
    assert actual.shape == (batch_size, 4)
    np.testing.assert_allclose(actual.values, 3.0, rtol=1e-12)


@pytest.mark.parametrize("matrix_dtype", [np.float32, np.int32])
def test_mean_preserves_dtype_promotion_for_custom_operator(matrix_dtype):
    op = _operator(4, 5)
    op = replace(op, matrix=op.matrix.astype(matrix_dtype))
    grid = xr.DataArray(
        np.ones((4, 5), dtype=matrix_dtype), dims=("latitude", "longitude"),
        coords={"latitude": op.source_lat, "longitude": op.source_lon},
    )
    expected = (grid.to_numpy().ravel() @ op.matrix.T) / op.row_sums
    actual = reduce_with_operator(grid, op)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.values, expected)


def test_apply_grid_rejects_wrong_shape():
    op = _operator(4, 5)
    with pytest.raises(ValueError, match="expected trailing source shape"):
        op.apply_grid(np.ones((5, 4)))


def test_reduce_operator_with_no_stored_coefficients():
    op = _operator(101, 103)
    op = replace(op, matrix=sp.csr_matrix(op.matrix.shape), row_sums=np.ones(4))
    actual = op.apply_grid(np.ones((101, 103)), descending=True)
    np.testing.assert_array_equal(actual, np.zeros(4))


def test_large_operator_cache_reload_and_mixed_dataset(tmp_path):
    lat, lon = np.linspace(-10, 10, 101), np.linspace(-10, 10, 103)
    geoms = gpd.GeoSeries([shapely.box(-2, -2, 2, 2)], index=["zone"])
    stencil = Stencil.compute(lat, lon, geoms)
    cache = LocalCache(tmp_path)
    first = cache.get_or_compute_reduce_operator(stencil, lat, lon)
    rng = np.random.default_rng(5)
    source = xr.Dataset(
        {
            "t2m": (("latitude", "longitude"), rng.random((101, 103), dtype=np.float32), {"units": "K"}),
            "tp": (("step", "latitude", "longitude"), rng.random((2, 101, 103)), {"units": "mm"}),
            "scalar": 5,
        },
        coords={"latitude": lat[::-1], "longitude": lon, "step": [0, 1], "model": "test"},
        attrs={"source": "synthetic"},
    )
    expected = reduce_with_operator(source, first)
    cached = cache.get_or_compute_reduce_operator(stencil, lat[::-1], lon)
    actual = reduce_with_operator(source, cached)
    xr.testing.assert_identical(actual, expected)
    assert actual.attrs == source.attrs
    assert actual.scalar == source.scalar
    for name in ("t2m", "tp"):
        values = source[name].sortby("latitude").to_numpy().reshape(-1, lat.size * lon.size)
        reference = (values @ first.matrix.T) / first.row_sums
        np.testing.assert_allclose(actual[name].to_numpy().reshape(-1, 1), reference, rtol=1e-12)
        assert actual[name].attrs == source[name].attrs
