"""Sparse tree construction must retain the previous row-composition semantics."""

import hashlib

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from geohalo import BiasTree, LocalCache
from geohalo.cache import _ser_tree


def _hierarchy(key_kind):
    rows = [
        ("a", "branch", 0.2), ("b", "branch", 1.1),
        ("branch", "middle", 3.7), ("c", "middle", 4.3),
        ("middle", "root", 0.17), ("d", "root", 2.6),
        ("e", "side", 0.4), ("f", "side", 0.7),
    ]

    def key(node):
        if key_kind == "multiindex":
            return ("region", node)
        if key_kind == "varying_arity":
            return (node,) if node in {"branch", "middle", "root", "side"} else ("leaf", node)
        if key_kind == "mixed":
            return ord(node) if len(node) == 1 else node
        return node

    children = [key(child) for child, _, _ in rows]
    index = (
        pd.MultiIndex.from_tuples(children, names=["kind", "child"])
        if key_kind == "multiindex" else pd.Index(children, name="child", tupleize_cols=False)
    )
    edges = pd.DataFrame(
        {"ancestor": [key(parent) for _, parent, _ in rows], "weight": [w for _, _, w in rows]},
        index=index,
    )
    # Independent expected ordering: leaves, then increasing maximum leaf distance.
    ordered = [
        node for level in [("a", "b", "c", "d", "e", "f"), ("branch", "side"), ("middle",), ("root",)]
        for node in sorted(map(key, level), key=repr)
    ]
    keys = (
        pd.MultiIndex.from_tuples(ordered, names=index.names)
        if key_kind == "multiindex" else pd.Index(ordered, name="child", tupleize_cols=False)
    )
    return edges, keys, 6


def _previous_matrix(edges, keys, n_leaves, how, weighted):
    """The old LIL row-by-row assembly, with independently specified node order."""
    children_of = {}
    for child, record in edges.iterrows():
        children_of.setdefault(record["ancestor"], []).append(
            (child, float(record["weight"]) if weighted else 1.0),
        )
    position = dict(zip(keys, range(len(keys)), strict=True))
    matrix = sp.lil_matrix((len(keys), n_leaves), dtype=np.float64)
    for j in range(n_leaves):
        matrix[j, j] = 1.0
    for node in keys[n_leaves:]:
        entries = children_of[node]
        total = sum(w for _, w in entries)
        composed = sp.csr_matrix((1, n_leaves), dtype=np.float64)
        for child, weight in entries:
            scale = weight / total if how == "mean" else weight
            composed = composed + matrix.getrow(position[child]) * scale
        matrix[position[node]] = composed
    return matrix.tocsr()


def _previous_digest(edges, how, weighted):
    digest = hashlib.sha256()
    digest.update(how.encode())
    names = list(edges.index.names) if isinstance(edges.index, pd.MultiIndex) else edges.index.name
    digest.update(repr(names).encode())
    rows = [
        (child, record["ancestor"], record["weight"]) if weighted else (child, record["ancestor"])
        for child, record in edges.iterrows()
    ]
    for row in sorted(rows, key=lambda row: tuple(repr(item) for item in row)):
        for item in row:
            digest.update(repr(item).encode())
    return digest.digest()


def _assert_csr_equal(actual, expected):
    assert isinstance(actual, sp.csr_matrix)
    assert actual.has_canonical_format
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("key_kind", ["flat", "multiindex", "varying_arity", "mixed"])
@pytest.mark.parametrize("shuffled", [False, True])
def test_matches_previous_build(how, weighted, key_kind, shuffled):
    edges, keys, n_leaves = _hierarchy(key_kind)
    if shuffled:
        edges = edges.sample(frac=1, random_state=42)
    expected = _previous_matrix(edges, keys, n_leaves, how, weighted)
    tree = BiasTree.compute(
        edges, parent_col="ancestor", weight_col="weight" if weighted else None, how=how,
    )
    _assert_csr_equal(tree.rollup_matrix, expected)
    pd.testing.assert_index_equal(tree.keys, keys)
    pd.testing.assert_index_equal(tree.leaf_keys, keys[:n_leaves])
    assert tree.digest == _previous_digest(edges, how, weighted)
    assert tree.how == how


@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("key_kind", ["flat", "multiindex"])
def test_previous_cache_payload_is_reused(tmp_path, monkeypatch, how, key_kind):
    edges, keys, n_leaves = _hierarchy(key_kind)
    matrix = _previous_matrix(edges, keys, n_leaves, how, weighted=True)
    digest = _previous_digest(edges, how, weighted=True)
    previous = BiasTree(matrix, keys, digest, how)
    cache = LocalCache(tmp_path)
    cache._store("tree", digest.hex()[:16], _ser_tree(previous))

    def no_build(*_args, **_kwargs):
        pytest.fail("existing cache entry should skip tree construction")

    monkeypatch.setattr(BiasTree, "compute", no_build)
    cached = cache.get_or_compute_tree(
        edges.iloc[::-1], parent_col="ancestor", weight_col="weight", how=how,
    )
    _assert_csr_equal(cached.rollup_matrix, matrix)
    pd.testing.assert_index_equal(cached.keys, keys)
    assert cached.digest == digest
    assert cached.how == how


def test_normalization_retains_python_float_sum():
    # Python >=3.12's compensated sum differs from naive np.bincount accumulation.
    edges = pd.DataFrame(
        {"ancestor": ["root"] * 5, "weight": [1e16, 1.0, 1.0, 1.0, 1.0]},
        index=pd.Index(list("abcde"), name="child"),
    )
    keys = pd.Index([*"abcde", "root"], name="child")
    expected = _previous_matrix(edges, keys, 5, "mean", weighted=True)
    actual = BiasTree.compute(edges, parent_col="ancestor", weight_col="weight")
    _assert_csr_equal(actual.rollup_matrix, expected)


@pytest.mark.parametrize("how", ["mean", "sum"])
def test_large_tree_uses_depth_bounded_sparse_products(monkeypatch, how):
    n_leaves, fanout = 2000, 10
    parents = [f"group{i // fanout}" for i in range(n_leaves)]
    groups = sorted(set(parents))
    edges = pd.DataFrame(
        {"parent": parents + ["root"] * len(groups) + ["root"]},
        index=[f"leaf{i}" for i in range(n_leaves)] + groups + ["loose"],
    )
    matmul = sp.csr_matrix.__matmul__
    products = []

    def count_product(left, right):
        products.append((left.shape, right.shape))
        return matmul(left, right)

    def no_lil(*_args, **_kwargs):
        pytest.fail("tree construction must not assemble individual LIL rows")

    monkeypatch.setattr(sp, "lil_matrix", no_lil)
    monkeypatch.setattr(sp.csr_matrix, "__matmul__", count_product)
    tree = BiasTree.compute(edges, how=how)
    assert len(products) == 2
    root = tree.rollup_matrix.getrow(tree.keys.get_loc("root")).toarray().ravel()
    expected = np.ones(n_leaves + 1)
    if how == "mean":
        expected /= len(groups) + 1
        expected[:n_leaves] /= fanout
    np.testing.assert_allclose(root, expected, rtol=1e-15)
    assert tree.rollup_matrix.nnz == 3 * n_leaves + 2


@pytest.mark.parametrize("how", ["mean", "sum"])
def test_deep_chain_is_nonrecursive(how):
    edges = pd.DataFrame({"parent": np.arange(1, 1101)}, index=np.arange(1100))
    tree = BiasTree.compute(edges, how=how)
    assert list(tree.keys) == list(range(1101))
    _assert_csr_equal(tree.rollup_matrix, sp.csr_matrix(np.ones((1101, 1))))


@pytest.mark.parametrize("weight", [1e-200, 1e200])
def test_extreme_products_terminate_and_match_previous(weight):
    edges = pd.DataFrame(
        {"ancestor": ["middle", "root"], "weight": [weight, weight]},
        index=pd.Index(["leaf", "middle"], name="child"),
    )
    keys = pd.Index(["leaf", "middle", "root"], name="child")
    with np.errstate(over="ignore", under="ignore"):
        expected = _previous_matrix(edges, keys, 1, "sum", weighted=True)
        actual = BiasTree.compute(edges, parent_col="ancestor", weight_col="weight", how="sum")
    _assert_csr_equal(actual.rollup_matrix, expected)


@pytest.mark.parametrize(
    "rows",
    [[("leaf", "a"), ("a", "b"), ("b", "a")],
     [("leaf", "root"), ("a", "b"), ("b", "a")],
     [("a", "a")], []],
)
def test_invalid_cycles_and_empty_edges_still_rejected(rows):
    edges = pd.DataFrame(rows, columns=["child", "parent"]).set_index("child")
    with pytest.raises(ValueError, match=r"cycle|leaf"):
        BiasTree.compute(edges)


@pytest.mark.parametrize("weight", [-1.0, 0.0, np.nan, np.inf, -np.inf, "1", 1 + 2j])
def test_invalid_weights_still_rejected(weight):
    edges = pd.DataFrame({"parent": ["root"], "weight": [weight]}, index=["leaf"])
    with pytest.raises(ValueError, match="positive finite"):
        BiasTree.compute(edges, weight_col="weight")
