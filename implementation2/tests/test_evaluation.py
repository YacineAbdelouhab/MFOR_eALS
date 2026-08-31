"""Metrics and split protocols, checked against brute-force definitions."""

from __future__ import annotations

import numpy as np
import pytest

from eals import evaluation as ev


def test_rank_of_truth_matches_argsort():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    for gt in (0, 7, 49):
        by_score = np.argsort(-scores, kind="stable")
        expected = int(np.where(by_score == gt)[0][0]) + 1
        assert ev.rank_of_truth(scores, gt, None) == expected


def test_rank_of_truth_ignores_seen_items():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert ev.rank_of_truth(scores, gt=3, seen=None) == 4
    # Removing the two items above it promotes the ground truth to rank 2.
    assert ev.rank_of_truth(scores, gt=3, seen=np.array([0, 1])) == 2


def test_metrics_definition():
    ranks = np.array([1, 5, 11, 100])
    m = ev.metrics_from_ranks(ranks, top_k=10)
    assert m["HR@10"] == pytest.approx(0.5)
    expected = (1 / np.log2(2) + 1 / np.log2(6)) / 4
    assert m["NDCG@10"] == pytest.approx(expected)
    assert ev.metrics_from_ranks(np.array([1, 1, 1]), 10)["NDCG@10"] == pytest.approx(1.0)


def test_leave_one_out_holds_out_latest_per_user():
    users = np.array([0, 0, 0, 1, 1, 2])
    items = np.array([10, 11, 12, 20, 21, 30])
    ts = np.array([1, 3, 2, 5, 4, 9])
    tr_u, tr_i, te_u, te_i = ev.leave_one_out_split(users, items, ts)

    assert dict(zip(te_u, te_i)) == {0: 11, 1: 20}  # latest by timestamp
    assert 2 not in te_u  # single-interaction user is never held out
    assert len(tr_u) == 4
    assert sorted(zip(tr_u.tolist(), tr_i.tolist())) == [(0, 10), (0, 12), (1, 21), (2, 30)]


def test_chronological_split_is_time_ordered():
    rng = np.random.default_rng(1)
    n = 100
    ts = rng.permutation(n)
    users, items = rng.integers(0, 10, n), rng.integers(0, 20, n)
    tr_u, tr_i, te_u, te_i, te_ts = ev.chronological_split(users, items, ts, train_frac=0.9)
    assert len(te_u) == 10 and len(tr_u) == 90
    assert te_ts.min() > 89  # the held-out tail is strictly the most recent
    assert np.all(np.diff(te_ts) >= 0)


def test_evaluate_matches_local_computation(spark_context):
    rng = np.random.default_rng(3)
    M, N, K = 40, 30, 4
    P, Q = rng.normal(size=(M, K)), rng.normal(size=(N, K))
    test_u = rng.integers(0, M, 25)
    test_i = rng.integers(0, N, 25)

    got = ev.evaluate(spark_context, P, Q, test_u, test_i, top_ks=(5, 10), n_parts=3)
    ranks = np.array([ev.rank_of_truth(Q @ P[u], int(i), None) for u, i in zip(test_u, test_i)])

    assert got["mean_rank"] == pytest.approx(ranks.mean())
    for k in (5, 10):
        assert got[f"HR@{k}"] == pytest.approx(ev.metrics_from_ranks(ranks, k)[f"HR@{k}"])
        assert got[f"NDCG@{k}"] == pytest.approx(ev.metrics_from_ranks(ranks, k)[f"NDCG@{k}"])
