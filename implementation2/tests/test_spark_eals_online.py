"""Algorithm 2: the rank-1 cache maintenance must be exact, not an approximation."""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import SparkSession

from eals import spark_eals_online as online
from eals.numpy_reference import compute_Sp, compute_Sq
from eals.weighting import item_confidence

from test_eals_numpy import toy_data


def seed_state(seed=0, w_new=4.0):
    data = toy_data(n_users=15, n_items=12, seed=seed)
    c = item_confidence(data.item_counts(), c0=32.0, alpha=0.4)
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (data.n_users, 6))
    Q = rng.normal(0, 0.05, (data.n_items, 6))
    users = np.repeat(np.arange(data.n_users), np.diff(data.u_indptr))
    return online.OnlineState.from_offline_fit(P, Q, c, lam=0.01, w_new=w_new, train_users=users, train_items=data.u_items)


def test_from_offline_fit_seeds_caches_correctly():
    state = seed_state()
    assert np.allclose(state.Sq, compute_Sq(state.Q, state.c))
    assert np.allclose(state.Sp, compute_Sp(state.P))
    assert len(state.user_items[0]) > 0  # user 0 has training history in toy_data


def test_n_prior_interactions_matches_history_length():
    state = seed_state()
    for u in range(state.P.shape[0]):
        assert state.n_prior_interactions(u) == len(state.user_items.get(u, []))


def test_cache_rank1_update_matches_full_recomputation():
    """The identity Algorithm 2 relies on: after p_u (or q_i) changes, S^p (S^q) can be
    patched with a rank-1 update instead of a full O(NK²) rebuild — this must be exact."""
    state = seed_state(seed=3)
    u, i = 2, 5  # an existing interaction: exercises the "grow R_u with a repeat" path too
    state.apply_interaction(u, i, n_iters=1)

    assert np.allclose(state.Sp, compute_Sp(state.P), atol=1e-10)
    assert np.allclose(state.Sq, compute_Sq(state.Q, state.c), atol=1e-10)


def test_apply_interaction_grows_history_and_moves_score_up():
    state = seed_state(seed=5, w_new=8.0)
    u, i = 1, 9
    before_len = state.n_prior_interactions(u)
    before_score = state.scores_for(u)[i]

    state.apply_interaction(u, i, n_iters=3)

    assert state.n_prior_interactions(u) == before_len + 1
    assert i in state.user_items[u] and u in state.item_users[i]
    assert state.user_weights[u][-1] == pytest.approx(8.0)
    # A large w_new pulling toward r=1 should push the score for (u, i) up, not down.
    assert state.scores_for(u)[i] > before_score


def test_cold_start_new_user_has_no_prior_history():
    data = toy_data(n_users=10, n_items=8, seed=7)
    c = item_confidence(data.item_counts(), c0=16.0, alpha=0.4)
    rng = np.random.default_rng(1)
    P, Q = rng.normal(0, 0.05, (10, 4)), rng.normal(0, 0.05, (8, 4))
    state = online.OnlineState.from_offline_fit(P, Q, c, lam=0.01, w_new=4.0, train_users=np.array([0, 0, 1]), train_items=np.array([0, 1, 2]))

    assert state.n_prior_interactions(5) == 0  # never trained: cold-start user
    state.apply_interaction(5, 3, n_iters=2)
    assert state.user_items[5] == [3]
    assert np.isfinite(state.P[5]).all()


def test_repeated_interaction_appends_again_not_replaces():
    """A user re-reviewing the same item is a second, distinct online-update event
    (paper's Amazon footnote 5) — not deduplicated like the offline training matrix."""
    state = seed_state(seed=9)
    u, i = 0, state.user_items[0][0] if state.user_items.get(0) else 3
    before = len(state.user_items.get(u, []))
    state.apply_interaction(u, i, n_iters=1)
    state.apply_interaction(u, i, n_iters=1)
    assert len(state.user_items[u]) == before + 2


@pytest.fixture(scope="module")
def spark(spark_context):
    return SparkSession(spark_context)


def test_write_stream_batches_chunks_correctly(tmp_path):
    users = np.arange(25)
    items = np.arange(25) % 5
    ts = np.arange(25)
    n_batches = online.write_stream_batches(users, items, ts, tmp_path, rows_per_batch=10)

    assert n_batches == 3  # 10 + 10 + 5
    files = sorted((tmp_path).glob("batch_*.json"))
    assert len(files) == 3
    assert len(files[0].read_text().splitlines()) == 10
    assert len(files[2].read_text().splitlines()) == 5


def test_write_stream_batches_clears_stale_files(tmp_path):
    online.write_stream_batches(np.arange(20), np.arange(20) % 4, np.arange(20), tmp_path, rows_per_batch=5)
    assert len(list(tmp_path.glob("batch_*.json"))) == 4
    online.write_stream_batches(np.arange(6), np.arange(6) % 4, np.arange(6), tmp_path, rows_per_batch=5)
    assert len(list(tmp_path.glob("batch_*.json"))) == 2  # stale batches 2 and 3 removed, not left behind


def test_run_streaming_processes_every_event_in_order(spark, tmp_path):
    state = seed_state(seed=11, w_new=4.0)
    stream_dir, ckpt_dir = tmp_path / "stream", tmp_path / "ckpt"

    # A repeat: user 3 interacts with item 7 twice, two batches apart.
    users = np.array([1, 2, 3, 3, 4, 1])
    items = np.array([6, 7, 7, 7, 6, 8])
    ts = np.arange(6)
    online.write_stream_batches(users, items, ts, stream_dir, rows_per_batch=2)

    results = online.run_streaming(spark, state, stream_dir, ckpt_dir, n_iters_online=1)

    assert len(results) == 6
    assert [r["batch_id"] for r in results] == [0, 0, 1, 1, 2, 2]  # 3 batches of 2, in order
    assert [(r["user"], r["item"]) for r in results] == list(zip(users.tolist(), items.tolist()))
    assert all(r["rank"] >= 1 for r in results)

    # User 3's second interaction with item 7 (index 3) must see one more prior
    # interaction than the first (index 2) — the update in between actually happened.
    assert results[3]["prior_count"] == results[2]["prior_count"] + 1


def test_run_streaming_grows_state_history(spark, tmp_path):
    state = seed_state(seed=13)
    before_users = set(state.user_items)
    stream_dir, ckpt_dir = tmp_path / "stream2", tmp_path / "ckpt2"

    users, items, ts = np.array([0, 1, 2]), np.array([4, 5, 6]), np.array([100, 200, 300])
    online.write_stream_batches(users, items, ts, stream_dir, rows_per_batch=1)
    online.run_streaming(spark, state, stream_dir, ckpt_dir, n_iters_online=2)

    for u, i in zip(users.tolist(), items.tolist()):
        assert i in state.user_items[u]
    assert set(state.user_items) >= before_users  # nothing lost, only grown
