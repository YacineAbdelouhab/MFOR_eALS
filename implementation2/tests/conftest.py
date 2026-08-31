"""One Spark session for the whole suite — starting a JVM per test file is the slow part."""

from __future__ import annotations

import pytest

from eals.spark_utils import build_session


@pytest.fixture(scope="session")
def spark_context():
    spark = build_session("tests", n_cores=4, driver_memory="2g")
    yield spark.sparkContext
    spark.stop()
