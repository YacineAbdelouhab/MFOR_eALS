from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F


def load_reviews(spark: SparkSession, path: str | Path) -> DataFrame:
    df = spark.read.json(str(path))
    return (
        df.select(
            F.col("user_id").alias("user"),
            F.col("business_id").alias("item"),
            F.unix_timestamp(F.col("date")).cast("long").alias("ts"),
        )
        .dropna()
        # A user reviewing the same business twice is still a single implicit 1.
        .groupBy("user", "item")
        .agg(F.min("ts").alias("ts"))
    )


def k_core_filter(df: DataFrame, k: int = 10, max_rounds: int = 20) -> DataFrame:

    current = df
    previous = -1
    for _ in range(max_rounds):
        n = current.count()
        if n == previous:
            break
        previous = n
        keep_u = current.groupBy("user").count().filter(F.col("count") >= k).select("user")
        keep_i = current.join(keep_u, "user").groupBy("item").count().filter(F.col("count") >= k).select("item")
        current = current.join(keep_u, "user").join(keep_i, "item").cache()
    return current


def index_ids(df: DataFrame) -> DataFrame:

    def dense(col: str, name: str) -> DataFrame:
        return (
            df.select(col).distinct().rdd.map(lambda r: r[0]).zipWithIndex().toDF([col, name])
        )

    return (
        df.join(dense("user", "user_idx"), "user")
        .join(dense("item", "item_idx"), "item")
        .select("user_idx", "item_idx", "ts")
    )


def prepare(spark: SparkSession, raw_path: str | Path, out_path: str | Path, k: int = 10) -> DataFrame:
    out = index_ids(k_core_filter(load_reviews(spark, raw_path), k=k))
    out.write.mode("overwrite").parquet(str(out_path))
    return spark.read.parquet(str(out_path))


def to_arrays(df: DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pdf = df.select("user_idx", "item_idx", "ts").toPandas()
    return (
        pdf["user_idx"].to_numpy(np.int64),
        pdf["item_idx"].to_numpy(np.int64),
        pdf["ts"].to_numpy(np.int64),
    )


def load_paper_ratings(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", header=None, names=["user", "item", "rating", "ts"],
        dtype={"user": np.int64, "item": np.int64, "rating": np.float64, "ts": np.int64},
    )


def dedupe_latest(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = df.sort_values("ts").drop_duplicates(subset=["user", "item"], keep="first")
    return d["user"].to_numpy(np.int64), d["item"].to_numpy(np.int64), d["ts"].to_numpy(np.int64)


def describe(users: np.ndarray, items: np.ndarray) -> dict[str, float]:
    M, N, nnz = int(users.max()) + 1, int(items.max()) + 1, users.shape[0]
    return {
        "users": M,
        "items": N,
        "interactions": nnz,
        "sparsity_pct": 100.0 * (1.0 - nnz / (M * N)),
        "avg_per_user": nnz / M,
        "avg_per_item": nnz / N,
    }
