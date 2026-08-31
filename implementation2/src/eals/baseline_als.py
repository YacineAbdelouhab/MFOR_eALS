from __future__ import annotations

import time

import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession


def to_ratings_df(spark: SparkSession, users: np.ndarray, items: np.ndarray):
    rows = [(int(u), int(i), 1.0) for u, i in zip(users, items)]
    return spark.createDataFrame(rows, ["user", "item", "rating"])


def _factors_to_array(factor_df, n_rows: int, K: int) -> np.ndarray:
    """MLlib returns factors as a DataFrame of (id, features); rows it never saw stay at zero."""
    out = np.zeros((n_rows, K), dtype=np.float64)
    for row in factor_df.collect():
        out[row["id"]] = row["features"]
    return out


def fit(
    spark: SparkSession,
    users: np.ndarray,
    items: np.ndarray,
    n_users: int,
    n_items: int,
    K: int,
    reg: float = 0.01,
    alpha: float = 1.0,
    max_iter: int = 10,
    n_parts: int = 8,
    seed: int = 0,
) -> dict:
    df = to_ratings_df(spark, users, items).repartition(n_parts).cache()
    df.count()

    als = ALS(
        rank=K, maxIter=max_iter, regParam=reg, alpha=alpha,
        implicitPrefs=True, userCol="user", itemCol="item", ratingCol="rating",
        coldStartStrategy="drop", numUserBlocks=n_parts, numItemBlocks=n_parts, seed=seed,
    )
    t0 = time.perf_counter()
    model = als.fit(df)
    model.userFactors.count()  # fit is lazy up to this point
    model.itemFactors.count()
    elapsed = time.perf_counter() - t0

    P = _factors_to_array(model.userFactors, n_users, K)
    Q = _factors_to_array(model.itemFactors, n_items, K)
    df.unpersist()
    return {"P": P, "Q": Q, "t_fit": elapsed, "t_per_iter": elapsed / max_iter, "K": K, "max_iter": max_iter}
