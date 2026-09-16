import numpy as np
import pytest

from geohalo.geometry import nearest_index
from geohalo.resampler import FactoredResampler, Resampler, resampler_digest


def _means_over_parents(target_vals, parent_lat, parent_lon, n_s_lat, n_s_lon):
    flat_parent = (parent_lat[:, None] * n_s_lon + parent_lon[None, :]).ravel()
    sums = np.bincount(flat_parent, weights=target_vals.ravel(), minlength=n_s_lat * n_s_lon)
    counts = np.bincount(flat_parent, minlength=n_s_lat * n_s_lon)
    return (sums / np.maximum(counts, 1)).reshape(n_s_lat, n_s_lon)


def test_identity_when_same_grid() -> None:
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0])
    r = Resampler.compute(lat, lon, lat, lon, iterations=1)
    np.testing.assert_allclose(r.transform_matrix.toarray(), np.eye(6), atol=1e-9)


def test_refine_mean_preservation_iter1() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 6)
    t_lon = np.linspace(0.0, 2.0, 6)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=1)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=(3, 3))
    y = (r.transform_matrix @ x.ravel()).reshape(6, 6)
    means = _means_over_parents(y, nearest_index(s_lat, t_lat), nearest_index(s_lon, t_lon), 3, 3)
    np.testing.assert_allclose(means, x, atol=1e-9)


def test_refine_mean_preservation_iter3() -> None:
    s_lat = np.array([0.0, 1.0, 2.0, 3.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 3.0, 12)
    t_lon = np.linspace(0.0, 2.0, 9)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=3)
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, size=(4, 3))
    y = (r.transform_matrix @ x.ravel()).reshape(12, 9)
    means = _means_over_parents(y, nearest_index(s_lat, t_lat), nearest_index(s_lon, t_lon), 4, 3)
    np.testing.assert_allclose(means, x, atol=1e-9)


def test_value_independence() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0])
    t_lat = np.linspace(0.0, 2.0, 5)
    t_lon = np.linspace(0.0, 1.0, 3)
    r1 = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=2)
    r2 = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=2)
    np.testing.assert_array_equal(r1.transform_matrix.toarray(), r2.transform_matrix.toarray())
    assert r1.digest == r2.digest


def test_coarsen_shape() -> None:
    s_lat = np.linspace(0.0, 4.0, 9)
    s_lon = np.linspace(0.0, 4.0, 9)
    t_lat = np.array([0.0, 2.0, 4.0])
    t_lon = np.array([0.0, 2.0, 4.0])
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=1)
    assert r.transform_matrix.shape == (9, 81)


def test_digest_changes_with_iterations() -> None:
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0])
    t_lat = np.linspace(0.0, 2.0, 5)
    t_lon = np.linspace(0.0, 1.0, 3)
    r1 = Resampler.compute(lat, lon, t_lat, t_lon, iterations=1)
    r2 = Resampler.compute(lat, lon, t_lat, t_lon, iterations=2)
    assert r1.digest != r2.digest


def test_invalid_iterations() -> None:
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=">= 1"):
        Resampler.compute(lat, lon, lat, lon, iterations=0)


@pytest.mark.parametrize("resampler_cls", [Resampler, FactoredResampler])
@pytest.mark.parametrize("iterations", [1, 3])
def test_source_latitudes_are_canonical(resampler_cls, iterations) -> None:
    lat = np.array([2.0, 1.0, 0.0])
    lon = np.array([0.0, 1.0])
    target_lat = np.linspace(2.0, 0.0, 5)
    target_lon = np.linspace(0.0, 1.0, 3)
    descending = resampler_cls.compute(lat, lon, target_lat, target_lon, iterations=iterations)
    ascending = Resampler.compute(lat[::-1], lon, target_lat, target_lon, iterations=iterations)

    np.testing.assert_array_equal(descending.source_lat, lat[::-1])
    np.testing.assert_array_equal(descending.target_lat, target_lat)
    assert descending.digest == ascending.digest
    # Direct matrix callers supply values in ascending source-latitude order.
    flat = np.arange(12.0).reshape(2, 6)
    actual = (
        descending.apply_flat(flat) if resampler_cls is FactoredResampler
        else flat @ descending.transform_matrix.T
    )
    np.testing.assert_allclose(actual, flat @ ascending.transform_matrix.T, atol=1e-12)
    np.testing.assert_array_equal(lat, [2.0, 1.0, 0.0])


def test_digest_canonicalises_only_source_latitudes() -> None:
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0])
    target_lat = np.linspace(0.0, 2.0, 5)
    target_lon = np.linspace(0.0, 1.0, 3)
    digest = resampler_digest(lat, lon, target_lat, target_lon, 1)
    assert resampler_digest(lat[::-1], lon, target_lat, target_lon, 1) == digest
    assert resampler_digest(lat, lon, target_lat[::-1], target_lon, 1) != digest
    assert resampler_digest(lat, lon, target_lat, target_lon[::-1], 1) != digest
    assert resampler_digest(lat, lon[::-1], target_lat, target_lon, 1) != digest
