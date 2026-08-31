from __future__ import annotations

import os
import sys

import pytest
from pyspark import SparkConf, SparkContext


@pytest.fixture(scope="session")
def sc() -> SparkContext:
    if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
        pytest.skip("local Spark on Windows requires HADOOP_HOME/bin/winutils.exe")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    context = SparkContext.getOrCreate(
        SparkConf()
        .setMaster("local[2]")
        .setAppName("pyspark-eals-rdd-tests")
        .set("spark.ui.enabled", "false")
        .set("spark.default.parallelism", "2")
    )
    context.setLogLevel("ERROR")
    context.setCheckpointDir("checkpoints/tests-rdd")
    yield context
    context.stop()

