from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pyspark import SparkContext
from pyspark.rdd import RDD

from eals.kernels import update_item, update_user


@dataclass
class SparkEALSResult:
    P: np.ndarray
    Q: np.ndarray
    history: list[dict] = field(default_factory=list)


def build_views(
    sc: SparkContext,
    pairs: list[tuple[int, int]] | RDD,
    n_parts: int,
) -> tuple[RDD, RDD]:
    """
    (u, i) pairs -> the user-grouped and item-grouped RDDs, both hash-partitioned.
    """
    rdd = sc.parallelize(pairs) if isinstance(pairs, list) else pairs
    rdd = rdd.cache()

    def pack(entries):
        items = np.fromiter(entries, dtype=np.int64)
        items.sort()
        ones = np.ones(items.shape[0], dtype=np.float64)
        return items, ones, ones  # (indices, w_ui, r_ui) — implicit feedback: both 1

    user_rdd = rdd.groupByKey(numPartitions=n_parts).mapValues(pack).cache()
    item_rdd = rdd.map(lambda t: (t[1], t[0])).groupByKey(numPartitions=n_parts).mapValues(pack).cache()
    user_rdd.count(), item_rdd.count()  # materialise before timing anything
    rdd.unpersist()
    return user_rdd, item_rdd


def _gram(sc: SparkContext, b_factors, b_weights, n_rows: int, n_parts: int) -> np.ndarray:
    edges = np.linspace(0, n_rows, n_parts + 1).astype(int)
    chunks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def partial(chunk):
        lo, hi = chunk
        F = b_factors.value[lo:hi]
        if b_weights is None:
            return F.T @ F
        return (F * b_weights.value[lo:hi, None]).T @ F

    return sc.parallelize(chunks, len(chunks)).map(partial).treeReduce(lambda a, b: a + b)


def _user_phase(b_P, b_Q, b_Sq, b_c, lam: float):
    def run(partition):
        P, Q, Sq, c = b_P.value, b_Q.value, b_Sq.value, b_c.value
        uids, rows = [], []
        for u, (items, w, r) in partition:
            p_u = P[u].copy()
            q_rows = Q[items]
            rhat = q_rows @ p_u
            update_user(p_u, rhat, q_rows, w, r, c[items], Sq, lam)
            uids.append(u)
            rows.append(p_u)
        # One array pair per partition, not one tuple per user
        return iter([(np.asarray(uids, dtype=np.int64), np.asarray(rows))]) if uids else iter([])

    return run


def _item_phase(b_P, b_Q, b_Sp, b_c, lam: float):
    def run(partition):
        P, Q, Sp, c = b_P.value, b_Q.value, b_Sp.value, b_c.value
        iids, rows = [], []
        for i, (users, w, r) in partition:
            q_i = Q[i].copy()
            p_rows = P[users]
            rhat = p_rows @ q_i
            update_item(q_i, rhat, p_rows, w, r, float(c[i]), Sp, lam)
            iids.append(i)
            rows.append(q_i)
        return iter([(np.asarray(iids, dtype=np.int64), np.asarray(rows))]) if iids else iter([])

    return run


def _scatter(base: np.ndarray, collected: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = base.copy()
    for ids, rows in collected:
        out[ids] = rows
    return out


def fit(
    sc: SparkContext,
    user_rdd: RDD,
    item_rdd: RDD,
    n_users: int,
    n_items: int,
    K: int,
    c: np.ndarray,
    lam: float = 0.01,
    n_iters: int = 10,
    seed: int = 0,
    n_parts: int | None = None,
    on_iteration=None,
) -> SparkEALSResult:
    from eals.numpy_reference import init_factors

    n_parts = n_parts or user_rdd.getNumPartitions()
    P, Q = init_factors(n_users, n_items, K, seed)
    b_c = sc.broadcast(c)
    b_P, b_Q = sc.broadcast(P), sc.broadcast(Q)
    result = SparkEALSResult(P=P, Q=Q)

    for it in range(n_iters):
        t0 = time.perf_counter()
        Sq = _gram(sc, b_Q, b_c, n_items, n_parts)
        t_sq = time.perf_counter()

        b_Sq = sc.broadcast(Sq)
        P = _scatter(P, user_rdd.mapPartitions(_user_phase(b_P, b_Q, b_Sq, b_c, lam)).collect())
        b_Sq.unpersist()
        b_P.unpersist()
        b_P = sc.broadcast(P)
        t_user = time.perf_counter()

        Sp = _gram(sc, b_P, None, n_users, n_parts)
        t_sp = time.perf_counter()

        b_Sp = sc.broadcast(Sp)
        Q = _scatter(Q, item_rdd.mapPartitions(_item_phase(b_P, b_Q, b_Sp, b_c, lam)).collect())
        b_Sp.unpersist()
        b_Q.unpersist()
        b_Q = sc.broadcast(Q)
        t_item = time.perf_counter()

        row = {
            "iter": it,
            "t_Sq": t_sq - t0,
            "t_user": t_user - t_sq,
            "t_Sp": t_sp - t_user,
            "t_item": t_item - t_sp,
            "t_iter": t_item - t0,
        }
        if on_iteration is not None:
            row.update(on_iteration(it, P, Q) or {})
        result.history.append(row)

    result.P, result.Q = P, Q
    b_P.unpersist()
    b_Q.unpersist()
    b_c.unpersist()
    return result
