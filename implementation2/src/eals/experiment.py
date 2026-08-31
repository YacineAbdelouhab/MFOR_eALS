from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

from eals import baseline_als, evaluation as ev, spark_eals
from eals.weighting import item_confidence


@dataclass
class Dataset:
    users: np.ndarray
    items: np.ndarray
    ts: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.users.max()) + 1

    @property
    def n_items(self) -> int:
        return int(self.items.max()) + 1


def synthetic(n_users=5000, n_items=3000, per_user=20, n_groups=25, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    users, items = [], []
    group_items = [rng.choice(n_items, size=max(n_items // n_groups, per_user), replace=False) for _ in range(n_groups)]
    for u in range(n_users):
        pool = group_items[u % n_groups]
        picked = rng.choice(pool, size=min(per_user, pool.size), replace=False)
        users.extend([u] * picked.size)
        items.extend(picked.tolist())
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    ts = rng.integers(0, 10**6, users.shape[0]).astype(np.int64)
    return Dataset(users, items, ts)


def load_parquet(spark: SparkSession, path: str | Path) -> Dataset:
    from eals.data_prep import to_arrays

    return Dataset(*to_arrays(spark.read.parquet(str(path))))


def load_yelp_paper_ratings(path: str | Path) -> Dataset:
    from eals.data_prep import dedupe_latest, load_paper_ratings

    return Dataset(*dedupe_latest(load_paper_ratings(path)))


@dataclass
class OnlineSplit:
    offline: Dataset
    stream_users: np.ndarray 
    stream_items: np.ndarray
    stream_ts: np.ndarray
    n_users: int
    n_items: int


def load_yelp_online_split(path: str | Path, train_frac: float = 0.9) -> OnlineSplit:
    from eals.data_prep import dedupe_latest, load_paper_ratings

    df = load_paper_ratings(path)
    n_users, n_items = int(df["user"].max()) + 1, int(df["item"].max()) + 1

    # mergesort is stable: ties (same ts) keep the file's original row order, so a
    # user's own repeat reviews stay in a sensible relative order across the cut.
    df_sorted = df.sort_values("ts", kind="mergesort")
    cut = int(len(df_sorted) * train_frac)
    train_df, stream_df = df_sorted.iloc[:cut], df_sorted.iloc[cut:]

    off_u, off_i, off_ts = dedupe_latest(train_df)
    return OnlineSplit(
        offline=Dataset(off_u, off_i, off_ts),
        stream_users=stream_df["user"].to_numpy(np.int64),
        stream_items=stream_df["item"].to_numpy(np.int64),
        stream_ts=stream_df["ts"].to_numpy(np.int64),
        n_users=n_users,
        n_items=n_items,
    )


def subsample(data: Dataset, fraction: float, seed: int = 0) -> Dataset:
    if fraction >= 1.0:
        return data
    rng = np.random.default_rng(seed)
    keep = rng.random(data.users.shape[0]) < fraction
    return Dataset(data.users[keep], data.items[keep], data.ts[keep])


def item_counts(items: np.ndarray, n_items: int) -> np.ndarray:
    return np.bincount(items, minlength=n_items).astype(np.float64)


def run_eals(
    sc,
    data: Dataset,
    K: int,
    c0: float = 512.0,
    alpha: float = 0.4,
    lam: float = 0.01,
    n_iters: int = 20,
    n_parts: int = 8,
    seed: int = 0,
    eval_every: int = 0,
    eval_cases: int | None = 5000,
    top_ks: tuple[int, ...] = (10, 100),
) -> dict:
    tr_u, tr_i, te_u, te_i = ev.leave_one_out_split(data.users, data.items, data.ts)
    n_users, n_items = data.n_users, data.n_items
    # Weights come from the *training* popularity only — using test data here would leak.
    c = item_confidence(item_counts(tr_i, n_items), c0=c0, alpha=alpha)

    t0 = time.perf_counter()
    user_rdd, item_rdd = spark_eals.build_views(sc, list(zip(tr_u.tolist(), tr_i.tolist())), n_parts=n_parts)
    t_build = time.perf_counter() - t0

    def track(it, P, Q):
        if eval_every and (it + 1) % eval_every == 0:
            return ev.evaluate(sc, P, Q, te_u, te_i, top_ks=top_ks, n_parts=n_parts, max_cases=eval_cases)
        return {}

    res = spark_eals.fit(
        sc, user_rdd, item_rdd, n_users, n_items, K=K, c=c, lam=lam,
        n_iters=n_iters, seed=seed, n_parts=n_parts, on_iteration=track,
    )
    final = ev.evaluate(sc, res.P, res.Q, te_u, te_i, top_ks=top_ks, n_parts=n_parts)

    iters = res.history
    return {
        "algo": "eALS", "K": K, "c0": c0, "alpha": alpha, "lam": lam,
        "n_iters": n_iters, "n_parts": n_parts,
        "n_users": n_users, "n_items": n_items, "n_train": int(tr_u.shape[0]), "n_test": int(te_u.shape[0]),
        "t_build_views": t_build,
        "t_total": sum(r["t_iter"] for r in iters),
        "t_per_iter": float(np.median([r["t_iter"] for r in iters])),
        "t_user_phase": float(np.median([r["t_user"] for r in iters])),
        "t_item_phase": float(np.median([r["t_item"] for r in iters])),
        "t_caches": float(np.median([r["t_Sq"] + r["t_Sp"] for r in iters])),
        **final,
        "_history": iters,
    }


def fit_eals_for_online(
    sc, split: "OnlineSplit", K: int, c0: float = 512.0, alpha: float = 0.4,
    lam: float = 0.01, n_iters: int = 20, n_parts: int = 8, seed: int = 0,
):
    data = split.offline
    c = item_confidence(item_counts(data.items, split.n_items), c0=c0, alpha=alpha)
    user_rdd, item_rdd = spark_eals.build_views(sc, list(zip(data.users.tolist(), data.items.tolist())), n_parts=n_parts)
    res = spark_eals.fit(sc, user_rdd, item_rdd, split.n_users, split.n_items, K=K, c=c, lam=lam, n_iters=n_iters, seed=seed, n_parts=n_parts)
    return res.P, res.Q, c


def run_mllib_als(
    spark: SparkSession, data: Dataset, K: int, lam: float = 0.01,
    n_iters: int = 10, n_parts: int = 8, seed: int = 0, top_ks: tuple[int, ...] = (10, 100),
) -> dict:
    sc = spark.sparkContext
    tr_u, tr_i, te_u, te_i = ev.leave_one_out_split(data.users, data.items, data.ts)
    out = baseline_als.fit(
        spark, tr_u, tr_i, data.n_users, data.n_items, K=K,
        reg=lam, max_iter=n_iters, n_parts=n_parts, seed=seed,
    )
    final = ev.evaluate(sc, out["P"], out["Q"], te_u, te_i, top_ks=top_ks, n_parts=n_parts)
    return {
        "algo": "MLlib-ALS", "K": K, "lam": lam, "n_iters": n_iters, "n_parts": n_parts,
        "n_users": data.n_users, "n_items": data.n_items,
        "n_train": int(tr_u.shape[0]), "n_test": int(te_u.shape[0]),
        "t_total": out["t_fit"], "t_per_iter": out["t_per_iter"],
        **final,
    }


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in row.items() if not k.startswith("_")}

    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        new_cols = [k for k in row if k not in fieldnames]
        if new_cols:
            fieldnames = fieldnames + new_cols
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(existing_rows)
    else:
        fieldnames = list(row)
        with path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    with path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames, restval="", extrasaction="ignore").writerow(row)
