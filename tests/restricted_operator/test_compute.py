from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

from geohalo import RestrictedOperator
from geohalo.restricted_operator import restricted_operator_digest
from tests.restricted_operator._helpers import make_grid, make_operator


@pytest.mark.parametrize("descending", [False, True])
def test_irregular_windows_cover_exactly_touched_chunks(descending):
    operator = make_operator()
    lat = operator.source_lat[::-1] if descending else operator.source_lat
    lat_chunks, lon_chunks = (3, 1, 4), (2, 3, 1, 4)
    plan = RestrictedOperator.compute(operator, lat, lat_chunks, lon_chunks)
    expected, actual = np.zeros((8, 10), dtype=int), np.zeros((8, 10), dtype=int)
    r_edges, c_edges = np.cumsum((0, *lat_chunks)), np.cumsum((0, *lon_chunks))
    row, col = np.divmod(np.unique(operator.matrix.indices), 10)
    if descending:
        row = 7 - row
    for r, c in zip(row, col, strict=True):
        ri, ci = np.searchsorted(r_edges, r, side="right") - 1, np.searchsorted(c_edges, c, side="right") - 1
        expected[r_edges[ri]:r_edges[ri + 1], c_edges[ci]:c_edges[ci + 1]] = 1
    for rows, cols in plan.windows:
        actual[rows, cols] += 1
    np.testing.assert_array_equal(actual, expected)  # no holes read, no overlapping windows
    stored = make_grid(operator, descending=descending, batch_shape=()).values
    gathered = np.concatenate([
        stored[rows, cols].ravel()[positions]
        for (rows, cols), positions in zip(plan.windows, plan.gathers, strict=True)
    ])
    expected_values = operator.matrix @ make_grid(operator, batch_shape=()).values.ravel()
    np.testing.assert_array_equal(plan.matrix @ gathered, expected_values)
    np.testing.assert_array_equal(plan.matrix.data, operator.matrix.data)
    np.testing.assert_array_equal(plan.matrix.indptr, operator.matrix.indptr)
    assert plan.matrix.shape == (3, 7)
    assert plan.digest == restricted_operator_digest(operator, lat, lat_chunks, lon_chunks)


@pytest.mark.parametrize(
    ("cells", "expected_windows"),
    [([0, 1, 4, 5], 1),  # filled rectangle
     ([0, 5, 10, 15], 4),  # diagonal: a bounding box would load 12 unrelated chunks
     ([0, 1, 2, 4, 6, 8, 9, 10], 4),  # ring with an untouched centre
     ([0, 1, 12, 13], 2)],  # identical spans separated by untouched rows
)
def test_rectangles_have_no_holes_or_overlaps(cells, expected_windows):
    matrix = sp.csr_matrix((np.ones(len(cells)), (np.zeros(len(cells), dtype=int), cells)), shape=(1, 16))
    operator = make_operator(matrix, n_lat=4, n_lon=4)
    plan = RestrictedOperator.compute(operator, operator.source_lat, 1, 1)
    mask = np.zeros((4, 4), dtype=int)
    for rows, cols in plan.windows:
        mask[rows, cols] += 1
    np.testing.assert_array_equal(mask.ravel(), matrix.toarray()[0])
    assert len(plan.windows) == expected_windows


def test_explicit_zeros_do_not_cause_reads_or_modify_original():
    matrix = sp.csr_matrix(([0.0, 2.0], [0, 79], [0, 2]), shape=(1, 80))
    operator = make_operator(matrix)
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    assert plan.windows == ((slice(6, 8), slice(8, 10)),)
    np.testing.assert_array_equal(plan.matrix.toarray(), [[2.0]])
    assert operator.matrix.nnz == 2


def test_empty_matrix_has_no_windows():
    operator = make_operator(sp.csr_matrix((3, 80)))
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    assert plan.windows == plan.gathers == ()
    assert plan.matrix.shape == (3, 0)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, (4, 0, 4), (4, -1, 5), (4.0, 4), (4, True, 3), (4,), ()])
def test_invalid_chunks_rejected(bad):
    operator = make_operator()
    with pytest.raises(ValueError, match="chunks"):
        RestrictedOperator.compute(operator, operator.source_lat, bad, 2)


@pytest.mark.parametrize("bad", [np.arange(7), np.arange(8) + 1, np.arange(8).reshape(2, 4), np.full(8, np.nan)])
def test_wrong_latitude_rejected(bad):
    operator = make_operator()
    with pytest.raises(ValueError, match="source_lat"):
        RestrictedOperator.compute(operator, bad, 2, 2)


def test_digest_includes_orientation_layout_and_operator():
    operator = make_operator()
    lat = operator.source_lat
    digest = restricted_operator_digest(operator, lat, 3, 4)
    assert digest == restricted_operator_digest(operator, lat, (3, 3, 2), (4, 4, 2))
    assert digest != restricted_operator_digest(operator, lat[::-1], 3, 4)
    assert digest != restricted_operator_digest(operator, lat, 2, 4)
    assert digest != restricted_operator_digest(operator, lat, 3, 2)
    assert digest != restricted_operator_digest(replace(operator, digest=b"other"), lat, 3, 4)


@pytest.mark.parametrize("metadata", ["dask", "preferred", "encoded"])
def test_from_grid_infers_metadata(metadata):
    operator = make_operator()
    grid = make_grid(operator, descending=True)
    if metadata == "dask":
        grid = grid.chunk({"latitude": 3, "longitude": 4})
        grid.encoding["chunks"] = (1, 1, 2, 2)  # Dask's actual layout takes precedence
    elif metadata == "preferred":
        grid.encoding["preferred_chunks"] = {"latitude": 3, "longitude": 4}
        grid.encoding["chunks"] = (1, 1, 2, 2)
    else:
        grid.encoding["chunks"] = (1, 1, 3, 4)
    plan = RestrictedOperator.from_grid(operator, grid)
    expected = RestrictedOperator.compute(operator, grid.latitude.values, 3, 4)
    assert plan.digest == expected.digest
    assert plan.windows == expected.windows


def test_from_grid_validates_metadata_and_coords():
    operator = make_operator()
    grid = make_grid(operator)
    with pytest.raises(TypeError, match="DataArray"):
        RestrictedOperator.from_grid(operator, grid.to_dataset())
    with pytest.raises(ValueError, match="missing required dims"):
        RestrictedOperator.from_grid(operator, grid.isel(latitude=0))
    with pytest.raises(ValueError, match="no spatial chunk metadata"):
        RestrictedOperator.from_grid(operator, grid)
    grid.encoding["chunks"] = (2, 2)
    with pytest.raises(ValueError, match="one entry per dimension"):
        RestrictedOperator.from_grid(operator, grid)
    grid.encoding["chunks"] = (1, 1, 2, 2)
    with pytest.raises(ValueError, match="source grid"):
        RestrictedOperator.from_grid(operator, grid.assign_coords(longitude=grid.longitude + 1))
