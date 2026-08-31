from __future__ import annotations

from pathlib import Path


def test_source_does_not_use_dataframe_apis() -> None:
    source_root = Path(__file__).parents[1] / "src"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))
    forbidden = ("pyspark.sql", "SparkSession", "DataFrame", "applyInPandas", "import pandas")
    assert all(token not in source for token in forbidden)

