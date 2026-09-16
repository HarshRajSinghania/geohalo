"""Synthetic issue #8 benchmark: row-by-row versus sparse fixed-point tree builds.

Run: ``uv run python -m benchmarks.bias_tree_build``. No downloads are required.
Both paths include validation, node ordering, sparse assembly, and hashing;
generating the hierarchy is outside the timed region.
"""

import argparse
import numbers
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from geohalo import BiasTree
from geohalo.bias_tree import _build_keys, _node_depth, tree_digest


def synthetic_edges(n_leaves, fanout):
    rng = np.random.default_rng(n_leaves)
    level = [f"leaf{i}" for i in range(n_leaves)]
    rows, depth = [], 0
    while len(level) > 1:
        parents = [f"n{depth}_{i // fanout}" for i in range(len(level))]
        rows.extend(
            (child, parent, float(rng.integers(1, 8)))
            for child, parent in zip(level, parents, strict=True)
        )
        level, depth = sorted(set(parents)), depth + 1
    # Leaves at different depths are common in geographic hierarchies.
    rows.extend((f"loose{i}", level[0], 2.0) for i in range(3))
    return pd.DataFrame(rows, columns=["child", "parent", "weight"]).set_index("child")


def _previous_compute(edges, how):
    """Previous BiasTree.compute, using the unchanged validation/key/digest helpers."""
    if not isinstance(edges, pd.DataFrame):
        raise TypeError("edges must be a pd.DataFrame")
    if not edges.index.is_unique:
        raise ValueError("edges.index has duplicates")
    if "parent" not in edges.columns:
        raise ValueError("parent column not found")
    children, parents, weights = list(edges.index), list(edges["parent"]), list(edges["weight"])
    for weight in weights:
        if not (isinstance(weight, numbers.Real) and np.isfinite(weight) and weight > 0):
            raise ValueError("edge weights must be positive finite")
    parent_of = dict(zip(children, parents, strict=True))
    children_of = {}
    for child, parent, weight in zip(children, parents, weights, strict=True):
        children_of.setdefault(parent, []).append((child, float(weight)))
    all_nodes = set(children) | set(parents)
    leaves = {node for node in all_nodes if node not in children_of}
    if not leaves:
        raise ValueError("edges must have at least one leaf")
    depth = _node_depth(parent_of, leaves)
    if len(depth) != len(all_nodes):
        raise ValueError("cycle detected")
    sorted_leaves = sorted(leaves, key=repr)
    internals = sorted(all_nodes - leaves, key=lambda node: (depth[node], repr(node)))
    nodes = sorted_leaves + internals
    position = {node: i for i, node in enumerate(nodes)}
    leaf_index = {node: i for i, node in enumerate(sorted_leaves)}
    n_leaves = len(sorted_leaves)
    matrix = sp.lil_matrix((len(nodes), n_leaves), dtype=np.float64)
    for leaf, j in leaf_index.items():
        matrix[position[leaf], j] = 1.0
    for node in internals:
        entries = children_of[node]
        total = sum(weight for _, weight in entries)
        composed = sp.csr_matrix((1, n_leaves), dtype=np.float64)
        for child, weight in entries:
            scale = weight / total if how == "mean" else weight
            composed = composed + matrix.getrow(position[child]) * scale
        matrix[position[node]] = composed
    return BiasTree(
        matrix.tocsr(), _build_keys(nodes, edges.index),
        tree_digest(edges, weight_col="weight", how=how), how,
    )


def _measure(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaves", type=int, default=2859, help="leaves before adding 3 at the root")
    parser.add_argument("--fanout", type=int, default=124)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.leaves < 2 or args.fanout < 2 or args.repeats < 1:
        parser.error("leaves and fanout must be at least 2; repeats must be positive")
    edges = synthetic_edges(args.leaves, args.fanout)
    n_leaves = args.leaves + 3
    n_internal = len(set(edges.index) | set(edges["parent"])) - n_leaves
    print(
        f"{n_leaves:,} leaves; {n_internal:,} internal nodes; medians of {args.repeats} builds",
        flush=True,
    )
    for how in ("mean", "sum"):
        previous, before = _measure(lambda how=how: _previous_compute(edges, how), args.repeats)
        print(f"{how}: previous {before:.3f} s", flush=True)
        current, after = _measure(
            lambda how=how: BiasTree.compute(edges, weight_col="weight", how=how), args.repeats,
        )
        print(f"{how}: current  {after:.3f} s ({before / after:.1f}x faster)", flush=True)
        np.testing.assert_array_equal(current.rollup_matrix.indptr, previous.rollup_matrix.indptr)
        np.testing.assert_array_equal(current.rollup_matrix.indices, previous.rollup_matrix.indices)
        np.testing.assert_array_equal(current.rollup_matrix.data, previous.rollup_matrix.data)
        pd.testing.assert_index_equal(current.keys, previous.keys)
        if current.digest != previous.digest or current.how != previous.how:
            raise AssertionError("tree digest or aggregation mode changed")
        print(f"{how}: CSR matrices, keys, and digests match exactly.", flush=True)


if __name__ == "__main__":
    main()
