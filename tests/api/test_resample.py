import numpy as np
import pytest
import xarray as xr

from geohalo.api import resample_grid, resample_grid_with_matrix
from geohalo.geometry import target_coords_from_resolution
from geohalo.resampler import Resampler


def _da(values, lats, lons, extra_dims=()):
    dims = (*extra_dims, "latitude", "longitude")
    return xr.DataArray(values, dims=dims, coords={"latitude": lats, "longitude": lons})


def test_with_matrix_shapes() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 6)
    t_lon = np.linspace(0.0, 2.0, 6)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=1)
    out = resample_grid_with_matrix(_da(np.arange(9.0).reshape(3, 3), s_lat, s_lon), r)
    assert out.sizes == {"latitude": 6, "longitude": 6}


def test_with_matrix_preserves_batch() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 6)
    t_lon = np.linspace(0.0, 2.0, 6)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=1)
    da = _da(np.zeros((4, 3, 3)), s_lat, s_lon, extra_dims=("member",))
    out = resample_grid_with_matrix(da, r)
    assert out.dims == ("member", "latitude", "longitude")
    assert out.sizes["member"] == 4


def test_resample_grid_mean_preserved_constant() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    out = resample_grid(_da(np.full((3, 3), 5.0), lats, lons), target_resolution=0.5, iterations=3)
    np.testing.assert_allclose(out.values, 5.0, atol=1e-9)


def test_resample_grid_dataset() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), np.full((3, 3), 2.0)),
         "u": (("latitude", "longitude"), np.full((3, 3), 9.0))},
        coords={"latitude": lats, "longitude": lons},
    )
    out = resample_grid(ds, target_resolution=0.5)
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"t", "u"}


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("builder", ["direct", "matrix_ascending", "matrix_descending"])
@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("iterations", [1, 3])
def test_resample_preserves_north_south(descending, builder, as_dataset, iterations) -> None:
    lats = np.arange(-10.0, 10.5, 1.0)
    lons = np.arange(5.0)
    ascending = _da(np.broadcast_to(lats[:, None], (lats.size, lons.size)), lats, lons)
    source = ascending.isel(latitude=slice(None, None, -1)) if descending else ascending
    if as_dataset:
        ascending = ascending.to_dataset(name="value")
        source = source.to_dataset(name="value")

    expected = resample_grid(ascending, 0.5, iterations=iterations)
    if builder == "direct":
        actual = resample_grid(source, 0.5, iterations=iterations)
    else:
        source_lat = lats[::-1] if builder == "matrix_descending" else lats
        target_lat, target_lon = target_coords_from_resolution(source_lat, lons, 0.5)
        resampler = Resampler.compute(source_lat, lons, target_lat, target_lon, iterations=iterations)
        actual = resample_grid_with_matrix(source, resampler)

    field = actual["value"] if as_dataset else actual
    np.testing.assert_allclose(field.isel(latitude=-1), 10.0, atol=1e-12)
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", ["latitude", "longitude"])
@pytest.mark.parametrize("mismatch", ["coordinates", "size"])
@pytest.mark.parametrize("as_dataset", [False, True])
def test_with_matrix_rejects_different_source_grid(dim, mismatch, as_dataset) -> None:
    lats = np.array([2.0, 1.0, 0.0])
    lons = np.array([0.0, 1.0, 2.0])
    source = _da(np.arange(9.0).reshape(3, 3), lats, lons)
    resampler = Resampler.compute(lats, lons, lats, lons)
    if mismatch == "coordinates":
        source = source.assign_coords({dim: source[dim] + 0.25})
    else:
        source = source.isel({dim: slice(1, None)})
    if as_dataset:
        source = source.to_dataset(name="value")

    with pytest.raises(ValueError, match="does not match the resampler's source grid"):
        resample_grid_with_matrix(source, resampler)


def test_with_matrix_requires_spatial_dims() -> None:
    coords = np.array([0.0, 1.0])
    resampler = Resampler.compute(coords, coords, coords, coords)
    source = xr.DataArray([1.0, 2.0], dims="latitude", coords={"latitude": coords})
    with pytest.raises(ValueError, match="missing required dims"):
        resample_grid_with_matrix(source, resampler)


def test_with_matrix_preserves_target_order_and_custom_dims() -> None:
    lats = np.array([2.0, 1.0, 0.0])
    lons = np.array([3.0, 4.0])
    target_lat = np.linspace(2.0, 0.0, 5)
    target_lon = np.linspace(4.0, 3.0, 3)
    source = xr.DataArray(
        np.arange(12.0).reshape(2, 3, 2), dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": lats, "lon": lons}, name="value", attrs={"units": "K"},
    ).transpose("lon", "time", "lat")
    resampler = Resampler.compute(lats, lons, target_lat, target_lon)
    actual = resample_grid_with_matrix(source, resampler, lat_dim="lat", lon_dim="lon")
    ascending = Resampler.compute(lats[::-1], lons, target_lat[::-1], target_lon[::-1])
    expected = resample_grid_with_matrix(source.sortby("lat"), ascending, lat_dim="lat", lon_dim="lon")
    xr.testing.assert_identical(actual, expected.isel(lat=slice(None, None, -1), lon=slice(None, None, -1)))
