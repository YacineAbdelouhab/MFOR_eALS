from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

SRC_DIR = str(Path(__file__).resolve().parent.parent)


def build_session(
    app_name: str = "eals",
    n_cores: int | str = "*",
    driver_memory: str = "8g",
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    existing = os.environ.get("PYTHONPATH", "")
    if SRC_DIR not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [SRC_DIR, existing]))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    builder = (
        SparkSession.builder.master(f"local[{n_cores}]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        # P and Q are collected to the driver each iteration; at K=1024 they run to tens of MB.
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    checkpoint_dir = Path(SRC_DIR).parent / ".spark-checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir))
    return spark
