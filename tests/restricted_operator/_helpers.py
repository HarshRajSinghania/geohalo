import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from geohalo import ReduceOperator


def make_operator(matrix=None, *, multiindex=False, n_lat=8, n_lon=10):
    if matrix is None:
        matrix = sp.csr_matrix(
            ([0.2, 0.5, 0.3, 0.4, 0.6, 1.2, -0.2], ([0, 0, 0, 1, 1, 2, 2], [1, 3, 23, 65, 78, 19, 41])),
            shape=(3, n_lat * n_lon),
        )
    labels = [f"zone{i}" for i in range(matrix.shape[0])]
    keys = (
        pd.MultiIndex.from_tuples([("region", label) for label in labels], names=["region", "zone"])
        if multiindex else pd.Index(labels, name="zone")
    )
    return ReduceOperator(
        matrix, np.arange(1, matrix.shape[0] + 1, dtype=float), keys,
        np.arange(n_lat, dtype=float), np.arange(n_lon, dtype=float), 1, b"test-operator",
    )


def make_grid(operator, *, descending=False, batch_shape=(3, 2), dtype=np.float32):
    shape = (*batch_shape, operator.source_lat.size, operator.source_lon.size)
    values = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(dtype)
    lat = operator.source_lat
    if descending:
        values, lat = values[..., ::-1, :], lat[::-1]
    batch_dims = tuple(f"dim{i}" for i in range(len(batch_shape)))
    coords = {dim: np.arange(size) for dim, size in zip(batch_dims, batch_shape, strict=True)}
    coords.update(latitude=lat, longitude=operator.source_lon, model="test")
    if batch_dims:
        coords["valid_time"] = (batch_dims[0], np.arange(batch_shape[0]) + 100)
    return xr.DataArray(
        values, dims=(*batch_dims, "latitude", "longitude"), coords=coords, name="t2m", attrs={"units": "K"},
    )
