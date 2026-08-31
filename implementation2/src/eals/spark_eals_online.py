from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyspark.sql.types import LongType, StructField, StructType

from eals.evaluation import rank_of_truth
from eals.kernels import update_item, update_user
from eals.numpy_reference import compute_Sp, compute_Sq

STREAM_SCHEMA = StructType(
    [StructField("user", LongType()), StructField("item", LongType()), StructField("ts", LongType())]
)


@dataclass
class OnlineState:
    P: np.ndarray
    Q: np.ndarray
    c: np.ndarray
    Sq: np.ndarray
    Sp: np.ndarray
    lam: float
    w_new: float
    user_items: dict[int, list[int]] = field(default_factory=dict)
    user_weights: dict[int, list[float]] = field(default_factory=dict)
    item_users: dict[int, list[int]] = field(default_factory=dict)
    item_weights: dict[int, list[float]] = field(default_factory=dict)

    @classmethod
    def from_offline_fit(cls, P, Q, c, lam, w_new, train_users, train_items) -> "OnlineState":
        state = cls(P=P.copy(), Q=Q.copy(), c=c, Sq=compute_Sq(Q, c), Sp=compute_Sp(P), lam=lam, w_new=w_new)
        for u, i in zip(train_users.tolist(), train_items.tolist()):
            state.user_items.setdefault(u, []).append(i)
            state.user_weights.setdefault(u, []).append(1.0)
            state.item_users.setdefault(i, []).append(u)
            state.item_weights.setdefault(i, []).append(1.0)
        return state

    def n_prior_interactions(self, u: int) -> int:
        return len(self.user_items.get(u, ()))

    def scores_for(self, u: int) -> np.ndarray:
        return self.Q @ self.P[u]

    def apply_interaction(self, u: int, i: int, n_iters: int = 1) -> None:
        self.user_items.setdefault(u, []).append(i)
        self.user_weights.setdefault(u, []).append(self.w_new)
        self.item_users.setdefault(i, []).append(u)
        self.item_weights.setdefault(i, []).append(self.w_new)

        for _ in range(n_iters):
            items = np.array(self.user_items[u], dtype=np.int64)
            w_u = np.array(self.user_weights[u], dtype=np.float64)
            r_u = np.ones(items.shape[0])
            q_rows = self.Q[items]
            rhat_u = q_rows @ self.P[u]
            p_old = self.P[u].copy()
            update_user(self.P[u], rhat_u, q_rows, w_u, r_u, self.c[items], self.Sq, self.lam)
            self.Sp += np.outer(self.P[u], self.P[u]) - np.outer(p_old, p_old)

            users = np.array(self.item_users[i], dtype=np.int64)
            w_i = np.array(self.item_weights[i], dtype=np.float64)
            r_i = np.ones(users.shape[0])
            p_rows = self.P[users]
            rhat_i = p_rows @ self.Q[i]
            q_old = self.Q[i].copy()
            update_item(self.Q[i], rhat_i, p_rows, w_i, r_i, float(self.c[i]), self.Sp, self.lam)
            self.Sq += float(self.c[i]) * (np.outer(self.Q[i], self.Q[i]) - np.outer(q_old, q_old))


def write_stream_batches(users: np.ndarray, items: np.ndarray, ts: np.ndarray, out_dir: str | Path, rows_per_batch: int) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("batch_*.json"):
        stale.unlink()

    n = users.shape[0]
    n_batches = 0
    base_time = time.time()
    for batch_idx, start in enumerate(range(0, n, rows_per_batch)):
        end = min(start + rows_per_batch, n)
        path = out_dir / f"batch_{batch_idx:06d}.json"
        with path.open("w") as fh:
            for u, i, t in zip(users[start:end].tolist(), items[start:end].tolist(), ts[start:end].tolist()):
                fh.write(json.dumps({"user": int(u), "item": int(i), "ts": int(t)}) + "\n")
        mtime = base_time + batch_idx
        os.utime(path, (mtime, mtime))
        n_batches += 1
    return n_batches


def run_streaming(spark, state: OnlineState, stream_dir: str | Path, checkpoint_dir: str | Path, n_iters_online: int = 1) -> list[dict]:
    results: list[dict] = []

    def process_batch(batch_df, batch_id):
        # Spark's file source does not guarantee row order within a micro-batch (confirmed:
        # order held incidentally under WSL2 but broke under Docker's file listing/scheduling)
        # — order explicitly, since Algorithm 2's sequential semantics depend on arrival order.
        pdf = batch_df.orderBy("ts").toPandas()
        for row in pdf.itertuples(index=False):
            u, i = int(row.user), int(row.item)
            prior_count = state.n_prior_interactions(u)
            rank = rank_of_truth(state.scores_for(u), i, None)
            results.append({"batch_id": int(batch_id), "user": u, "item": i, "ts": int(row.ts), "prior_count": prior_count, "rank": rank})
            state.apply_interaction(u, i, n_iters=n_iters_online)

    query = (
        spark.readStream.format("json")
        .schema(STREAM_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .load(str(stream_dir))
        .writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", str(checkpoint_dir))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    return results
