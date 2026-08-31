"""Data preparation, on a synthetic file shaped like the Yelp review dump."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from eals import data_prep

REAL_YELP_RATING = Path(__file__).resolve().parent.parent / "data" / "raw" / "yelp.rating"


@pytest.fixture(scope="module")
def spark(spark_context):
    return SparkSession(spark_context)


def write_reviews(path, rows):
    with open(path, "w") as fh:
        for user, item, date in rows:
            fh.write(json.dumps({"user_id": user, "business_id": item, "date": date, "stars": 4}) + "\n")
    return str(path)


def test_load_deduplicates_repeat_reviews(spark, tmp_path):
    path = write_reviews(
        tmp_path / "r.json",
        [("u1", "b1", "2015-01-01 10:00:00"), ("u1", "b1", "2016-01-01 10:00:00"), ("u1", "b2", "2015-06-01 10:00:00")],
    )
    got = {(r["user"], r["item"]): r["ts"] for r in data_prep.load_reviews(spark, path).collect()}
    assert len(got) == 2  # the repeat review collapses to one implicit 1
    assert got[("u1", "b1")] < got[("u1", "b2")]  # earliest timestamp kept


def test_k_core_filter_reaches_fixed_point(spark, tmp_path):
    """A single pass is not enough: dropping items can push a user back under the threshold."""
    rows = []
    # 3 dense users x 3 dense items -> everyone has 3 interactions.
    for u in range(3):
        for i in range(3):
            rows.append((f"dense_u{u}", f"dense_b{i}", "2015-01-01 10:00:00"))
    # A user propped up only by items that will themselves be removed.
    for i in range(3):
        rows.append(("weak_u", f"rare_b{i}", "2015-01-01 10:00:00"))

    path = write_reviews(tmp_path / "r.json", rows)
    kept = data_prep.k_core_filter(data_prep.load_reviews(spark, path), k=3).collect()

    users = {r["user"] for r in kept}
    assert users == {"dense_u0", "dense_u1", "dense_u2"}
    assert all(r["item"].startswith("dense_b") for r in kept)
    assert len(kept) == 9


def test_index_ids_are_dense_and_bijective(spark, tmp_path):
    rows = [(f"u{u}", f"b{i}", "2015-01-01 10:00:00") for u in range(4) for i in range(4)]
    path = write_reviews(tmp_path / "r.json", rows)
    indexed = data_prep.index_ids(data_prep.load_reviews(spark, path))
    users, items, ts = data_prep.to_arrays(indexed)

    assert sorted(set(users.tolist())) == list(range(4))
    assert sorted(set(items.tolist())) == list(range(4))
    assert len(set(zip(users.tolist(), items.tolist()))) == 16
    assert ts.shape == users.shape


def test_describe_matches_manual_counts():
    users = np.array([0, 0, 1, 2])
    items = np.array([0, 1, 1, 3])
    d = data_prep.describe(users, items)
    assert d["users"] == 3 and d["items"] == 4 and d["interactions"] == 4
    assert d["sparsity_pct"] == pytest.approx(100 * (1 - 4 / 12))


def test_load_paper_ratings_parses_tsv(tmp_path):
    path = tmp_path / "toy.rating"
    path.write_text("0\t0\t4.0\t100\n0\t1\t3.0\t200\n1\t0\t5.0\t150\n")
    df = data_prep.load_paper_ratings(path)
    assert list(df.columns) == ["user", "item", "rating", "ts"]
    assert df.shape == (3, 4)
    assert df["user"].dtype == np.int64
    assert df.loc[0, "ts"] == 100


def test_dedupe_latest_keeps_earliest_and_collapses_repeats():
    df = pd.DataFrame({
        "user": [0, 0, 0, 1],
        "item": [5, 5, 6, 7],
        "rating": [3.0, 4.0, 2.0, 5.0],
        "ts": [200, 100, 300, 50],  # user 0 reviewed item 5 twice: ts=200 then ts=100
    })
    users, items, ts = data_prep.dedupe_latest(df)
    pairs = dict(zip(zip(users.tolist(), items.tolist()), ts.tolist()))
    assert pairs == {(0, 5): 100, (0, 6): 300, (1, 7): 50}  # earliest ts kept for the repeat


@pytest.mark.skipif(not REAL_YELP_RATING.exists(), reason="data/raw/yelp.rating not present")
def test_real_yelp_rating_matches_paper_table_2():
    df = data_prep.load_paper_ratings(REAL_YELP_RATING)
    assert df.shape[0] == 731_671
    assert df["user"].nunique() == 25_677
    assert df["item"].nunique() == 25_815
    assert df["user"].min() == 0 and df["item"].min() == 0
    assert df["user"].max() == 25_676 and df["item"].max() == 25_814  # dense 0-indexed

    users, items, ts = data_prep.dedupe_latest(df)
    d = data_prep.describe(users, items)
    assert d["users"] == 25_677 and d["items"] == 25_815
    assert d["sparsity_pct"] == pytest.approx(99.89, abs=0.01)
    assert users.shape[0] < 731_671  # the ~30k repeat reviews were collapsed
