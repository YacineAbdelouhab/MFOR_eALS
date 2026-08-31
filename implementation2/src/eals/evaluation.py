"""Split protocols and ranking metrics"""

from __future__ import annotations

import numpy as np
from pyspark import SparkContext


def leave_one_out_split(
    users: np.ndarray, items: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """each user's latest interaction held out (train_u, train_i, test_u, test_i)."""
    order = np.lexsort((timestamps, users))
    u, i = users[order], items[order]
    is_last = np.empty(u.shape[0], dtype=bool)
    is_last[-1] = True
    is_last[:-1] = u[:-1] != u[1:]
    # A user with a single interaction would end up with no training history at all.
    counts = np.bincount(u, minlength=u.max() + 1)
    is_last &= counts[u] > 1
    return u[~is_last], i[~is_last], u[is_last], i[is_last]


def chronological_split(
    users: np.ndarray, items: np.ndarray, timestamps: np.ndarray, train_frac: float = 0.9
):
    """split by time, first 90% offline, last 10% streamed."""
    order = np.argsort(timestamps, kind="stable")
    cut = int(len(order) * train_frac)
    tr, te = order[:cut], order[cut:]
    return users[tr], items[tr], users[te], items[te], timestamps[te]


def rank_of_truth(scores: np.ndarray, gt: int, seen: np.ndarray | None) -> int:
    gt_score = scores[gt]
    higher = scores > gt_score
    if seen is not None and seen.size:
        higher[seen] = False
    higher[gt] = False
    return int(higher.sum()) + 1


def metrics_from_ranks(ranks: np.ndarray, top_k: int) -> dict[str, float]:
    hit = ranks <= top_k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    return {f"HR@{top_k}": float(hit.mean()), f"NDCG@{top_k}": float(ndcg.mean())}


def _rank_partition(b_P, b_Q, b_seen, exclude_seen: bool, batch: int = 512):
    def run(partition):
        P, Q = b_P.value, b_Q.value
        seen = b_seen.value if exclude_seen else None
        cases = list(partition)
        out = []
        for start in range(0, len(cases), batch):
            chunk = cases[start : start + batch]
            us = np.fromiter((u for u, _ in chunk), dtype=np.int64, count=len(chunk))
            gts = np.fromiter((i for _, i in chunk), dtype=np.int64, count=len(chunk))
            scores = P[us] @ Q.T  # one GEMM per batch rather than one dot per user
            for row, u, gt in zip(scores, us, gts):
                out.append(rank_of_truth(row, int(gt), seen.get(int(u)) if seen else None))
        return iter([np.asarray(out, dtype=np.int64)]) if out else iter([])

    return run


def evaluate(
    sc: SparkContext,
    P: np.ndarray,
    Q: np.ndarray,
    test_users: np.ndarray,
    test_items: np.ndarray,
    top_ks: tuple[int, ...] = (10, 100),
    seen_by_user: dict[int, np.ndarray] | None = None,
    exclude_seen: bool = False,
    n_parts: int = 8,
    max_cases: int | None = None,
    seed: int = 0,
) -> dict[str, float]:

    if max_cases is not None and max_cases < test_users.shape[0]:
        pick = np.random.default_rng(seed).choice(test_users.shape[0], max_cases, replace=False)
        test_users, test_items = test_users[pick], test_items[pick]

    b_P, b_Q = sc.broadcast(P), sc.broadcast(Q)
    b_seen = sc.broadcast(seen_by_user or {})
    cases = list(zip(test_users.tolist(), test_items.tolist()))
    ranks = np.concatenate(
        sc.parallelize(cases, n_parts).mapPartitions(_rank_partition(b_P, b_Q, b_seen, exclude_seen)).collect()
    )
    for b in (b_P, b_Q, b_seen):
        b.unpersist()

    result = {"n_cases": float(ranks.shape[0]), "mean_rank": float(ranks.mean())}
    for k in top_ks:
        result.update(metrics_from_ranks(ranks, k))
    return result
