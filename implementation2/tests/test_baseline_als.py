"""MLlib ALS wrapper: shapes, timing fields, and that it actually learns something."""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import SparkSession

from eals import baseline_als, evaluation as ev


@pytest.fixture(scope="module")
def spark(spark_context):
    return SparkSession(spark_context)


def block_data(n_users=40, n_items=30, seed=0):
    """Two disjoint user/item blocks — any working MF must rank within-block items higher."""
    rng = np.random.default_rng(seed)
    users, items = [], []
    for u in range(n_users):
        block = u % 2
        pool = np.arange(block, n_items, 2)
        for i in rng.choice(pool, size=6, replace=False):
            users.append(u)
            items.append(int(i))
    return np.array(users), np.array(items)


def test_fit_returns_usable_factors(spark):
    users, items = block_data()
    out = baseline_als.fit(spark, users, items, n_users=40, n_items=30, K=8, max_iter=5, n_parts=2)

    assert out["P"].shape == (40, 8)
    assert out["Q"].shape == (30, 8)
    assert out["t_fit"] > 0
    assert out["t_per_iter"] == pytest.approx(out["t_fit"] / 5)
    assert np.isfinite(out["P"]).all() and np.isfinite(out["Q"]).all()
    assert np.abs(out["P"]).sum() > 0


def test_survives_enough_iterations_to_need_checkpointing(spark):
    """Regression test: without spark.setCheckpointDir, ALS's un-checkpointed RDD lineage
    grows every iteration and blows the JVM deserialization stack around maxIter~20-30
    (empirically hit at 30) — build_session must configure a checkpoint dir."""
    users, items = block_data(seed=8)
    out = baseline_als.fit(spark, users, items, n_users=40, n_items=30, K=8, max_iter=30, n_parts=2)
    assert np.isfinite(out["P"]).all() and np.isfinite(out["Q"]).all()


def test_learns_block_structure(spark, spark_context):
    """Held-out items from a user's own block should rank well above random (rank ~ N/2)."""
    users, items = block_data(seed=1)
    train_mask = np.ones(len(users), dtype=bool)
    train_mask[::6] = False  # hold out one interaction per user

    out = baseline_als.fit(
        spark, users[train_mask], items[train_mask], n_users=40, n_items=30, K=8, max_iter=15, n_parts=2
    )
    ranks = np.array([
        ev.rank_of_truth(out["Q"] @ out["P"][u], int(i), None)
        for u, i in zip(users[~train_mask], items[~train_mask])
    ])
    assert ranks.mean() < 15.0  # random would average ~15.5 over 30 items
