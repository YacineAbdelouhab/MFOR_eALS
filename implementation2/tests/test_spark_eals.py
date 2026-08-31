"""The Spark parallelisation must reproduce the sequential oracle exactly (paper §4.2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from eals import interactions, numpy_reference as ref, spark_eals
from eals.weighting import item_confidence

from test_eals_numpy import toy_data


@pytest.fixture(scope="module")
def sc(spark_context):
    return spark_context


def pairs_of(data: interactions.Interactions) -> list[tuple[int, int]]:
    users = np.repeat(np.arange(data.n_users), np.diff(data.u_indptr))
    return [(int(u), int(i)) for u, i in zip(users, data.u_items)]


@pytest.mark.parametrize("n_parts", [1, 3, 7])
def test_spark_matches_numpy_reference(sc, n_parts):
    """Same seed, same result — regardless of how users are spread over partitions."""
    data = toy_data(n_users=25, n_items=18, seed=11)
    c = item_confidence(data.item_counts(), c0=32.0, alpha=0.4)
    K, lam, n_iters = 6, 0.01, 5

    expected = ref.fit(data, K=K, c=c, lam=lam, n_iters=n_iters, seed=2, track_objective=False)

    user_rdd, item_rdd = spark_eals.build_views(sc, pairs_of(data), n_parts=n_parts)
    got = spark_eals.fit(
        sc, user_rdd, item_rdd, data.n_users, data.n_items,
        K=K, c=c, lam=lam, n_iters=n_iters, seed=2, n_parts=n_parts,
    )

    assert np.allclose(got.P, expected.P, rtol=1e-9, atol=1e-11)
    assert np.allclose(got.Q, expected.Q, rtol=1e-9, atol=1e-11)


def test_partitioning_does_not_change_the_result(sc):
    """The invariant CLAUDE.md rests on, stated as its own test."""
    data = toy_data(n_users=30, n_items=20, seed=4)
    c = item_confidence(data.item_counts(), c0=64.0, alpha=0.5)
    pairs = pairs_of(data)

    runs = []
    for n_parts in (1, 2, 5):
        u, i = spark_eals.build_views(sc, pairs, n_parts=n_parts)
        runs.append(spark_eals.fit(sc, u, i, data.n_users, data.n_items, K=5, c=c, n_iters=4, seed=9, n_parts=n_parts))

    for other in runs[1:]:
        assert np.allclose(runs[0].P, other.P, rtol=1e-9, atol=1e-11)
        assert np.allclose(runs[0].Q, other.Q, rtol=1e-9, atol=1e-11)


def test_spark_objective_is_non_increasing(sc):
    data = toy_data(n_users=25, n_items=18, seed=6)
    c = item_confidence(data.item_counts(), c0=32.0, alpha=0.4)

    objs = []

    def track(it, P, Q):
        rhat = ref.predict_cache(data, P, Q)
        obj = ref.objective(data, P, Q, c, 0.01, rhat)
        objs.append(obj)
        return {"objective": obj}

    res = spark_eals.fit(
        sc, *spark_eals.build_views(sc, pairs_of(data), n_parts=3),
        n_users=data.n_users, n_items=data.n_items,
        K=6, c=c, lam=0.01, n_iters=8, seed=1, n_parts=3, on_iteration=track,
    )
    assert np.all(np.diff(np.array(objs)) <= 1e-8)
    assert all("t_iter" in row for row in res.history)


def test_gram_matches_dense(sc):
    """Distributed S^q must equal the single-shot GEMM."""
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(37, 5))
    c = rng.random(37)
    b_Q, b_c = sc.broadcast(Q), sc.broadcast(c)
    assert np.allclose(spark_eals._gram(sc, b_Q, b_c, 37, 4), ref.compute_Sq(Q, c))
    assert np.allclose(spark_eals._gram(sc, b_Q, None, 37, 4), ref.compute_Sp(Q))
