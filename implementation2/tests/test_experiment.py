"""End-to-end pipeline on synthetic data: split -> fit -> evaluate -> CSV."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from pyspark.sql import SparkSession

from eals import experiment as xp


@pytest.fixture(scope="module")
def spark(spark_context):
    return SparkSession(spark_context)


def test_synthetic_has_planted_structure():
    data = xp.synthetic(n_users=200, n_items=100, per_user=8, n_groups=10, seed=0)
    assert data.n_users == 200
    assert data.users.shape == data.items.shape == data.ts.shape
    # Users of the same group must overlap far more than users of different groups.
    by_user = {u: set(data.items[data.users == u].tolist()) for u in (0, 10, 1)}
    assert len(by_user[0] & by_user[10]) > len(by_user[0] & by_user[1])


def test_subsample_thins_without_reindexing():
    data = xp.synthetic(n_users=100, n_items=50, per_user=6, seed=1)
    half = xp.subsample(data, 0.5, seed=0)
    assert 0.4 < half.users.shape[0] / data.users.shape[0] < 0.6
    assert half.users.max() <= data.users.max()


def test_eals_end_to_end_beats_random(spark_context):
    """The planted blocks must be recovered: mean rank well below N/2."""
    data = xp.synthetic(n_users=300, n_items=200, per_user=10, n_groups=15, seed=2)
    row = xp.run_eals(spark_context, data, K=16, c0=64.0, alpha=0.4, n_iters=8, n_parts=4, eval_cases=None)

    assert row["algo"] == "eALS"
    assert row["n_test"] > 0
    assert 0.0 <= row["HR@10"] <= 1.0
    assert row["mean_rank"] < 100.0 / 2  # random baseline over 200 items is ~100
    assert row["HR@100"] > row["HR@10"] or row["HR@10"] == 1.0
    assert row["t_per_iter"] > 0 and len(row["_history"]) == 8


def test_eals_accuracy_improves_with_iterations(spark_context):
    data = xp.synthetic(n_users=300, n_items=200, per_user=10, n_groups=15, seed=3)
    row = xp.run_eals(
        spark_context, data, K=16, c0=64.0, alpha=0.4, n_iters=9, n_parts=4,
        eval_every=3, eval_cases=None,
    )
    tracked = [h["HR@10"] for h in row["_history"] if "HR@10" in h]
    assert len(tracked) == 3
    assert tracked[-1] >= tracked[0]


def test_mllib_baseline_runs_and_is_comparable(spark):
    data = xp.synthetic(n_users=200, n_items=120, per_user=8, n_groups=12, seed=4)
    row = xp.run_mllib_als(spark, data, K=16, n_iters=8, n_parts=4)
    assert row["algo"] == "MLlib-ALS"
    assert row["mean_rank"] < 60.0
    assert row["t_per_iter"] > 0


def write_rating_file(path, rows):
    with open(path, "w") as fh:
        for u, i, r, t in rows:
            fh.write(f"{u}\t{i}\t{r}\t{t}\n")
    return str(path)


def test_load_yelp_online_split_dedupes_offline_but_not_stream(tmp_path):
    # 10 rows -> 90% cut = index 9: rows[:9] offline, rows[9:] stream (already ts-sorted).
    rows = [(u, u % 3, 4.0, 100 + u) for u in range(9)]  # 9 distinct offline interactions
    rows.append((0, 0, 5.0, 109))  # a repeat of an offline pair, landing in the stream
    path = write_rating_file(tmp_path / "toy.rating", rows)

    split = xp.load_yelp_online_split(path, train_frac=0.9)

    assert split.offline.users.shape[0] == 9  # nothing collapsed: all 9 were distinct pairs
    assert split.stream_users.shape[0] == 1
    assert (split.stream_users[0], split.stream_items[0]) == (0, 0)  # the repeat, kept raw
    assert split.n_users == 9 and split.n_items == 3  # from the whole file, not just offline


def test_load_yelp_online_split_dedupes_within_offline_slice(tmp_path):
    # 10 rows, cut=int(10*0.9)=9 -> offline = ts 100..108 (9 rows, both (0,0) copies
    # included), stream = the single ts=109 row.
    rows = [
        (0, 0, 4.0, 100), (0, 0, 3.0, 101),  # same pair twice, both land in the offline head
        (1, 1, 4.0, 102), (2, 2, 4.0, 103), (3, 3, 4.0, 104),
        (4, 4, 4.0, 105), (5, 5, 4.0, 106), (6, 6, 4.0, 107),
        (7, 7, 4.0, 108), (8, 8, 4.0, 109),
    ]
    path = write_rating_file(tmp_path / "toy.rating", rows)
    split = xp.load_yelp_online_split(path, train_frac=0.9)

    pairs = list(zip(split.offline.users.tolist(), split.offline.items.tolist()))
    assert pairs.count((0, 0)) == 1  # the repeat collapsed to one training entry
    assert len(pairs) == 8  # 9 offline rows, one pair deduped away -> 8 distinct
    assert split.stream_users.shape[0] == 1 and split.stream_users[0] == 8  # only ts=109 left


def test_load_yelp_online_split_cold_start_user_has_no_offline_history(tmp_path):
    rows = [(u, 0, 4.0, 100 + u) for u in range(9)]
    rows.append((9, 0, 4.0, 109))  # user 9 only ever appears in the stream
    path = write_rating_file(tmp_path / "toy.rating", rows)
    split = xp.load_yelp_online_split(path, train_frac=0.9)

    assert 9 not in split.offline.users
    assert split.n_users == 10  # still sized for user 9, even with zero offline history
    assert split.stream_users[0] == 9


def test_fit_eals_for_online_sizes_factors_to_full_id_range(spark_context):
    data = xp.synthetic(n_users=60, n_items=40, per_user=6, n_groups=6, seed=0)
    split = xp.OnlineSplit(offline=data, stream_users=np.array([]), stream_items=np.array([]), stream_ts=np.array([]), n_users=80, n_items=50)
    P, Q, c = xp.fit_eals_for_online(spark_context, split, K=4, n_iters=2, n_parts=2)

    assert P.shape == (80, 4)  # sized for split.n_users, not just data.n_users (60)
    assert Q.shape == (50, 4)
    assert c.shape == (50,)


def test_append_csv_writes_header_once(tmp_path):
    path = tmp_path / "r.csv"
    xp.append_csv(path, {"algo": "eALS", "K": 8, "_history": [1, 2]})
    xp.append_csv(path, {"algo": "eALS", "K": 16, "_history": [3]})
    rows = list(csv.DictReader(path.open()))
    assert [r["K"] for r in rows] == ["8", "16"]
    assert "_history" not in rows[0]  # private fields stay out of the results file


def test_append_csv_handles_disjoint_schemas_without_misalignment(tmp_path):
    """The bug this guards against: a naive per-call DictWriter(fieldnames=list(row))
    writes each row in its own key order, so once two callers' key sets differ the
    values land under the wrong header columns."""
    path = tmp_path / "r.csv"
    xp.append_csv(path, {"algo": "eALS", "K": 16, "c0": 512.0, "alpha": 0.4, "lam": 0.01})
    xp.append_csv(path, {"algo": "MLlib-ALS", "K": 16, "lam": 0.02})  # no c0/alpha at all
    xp.append_csv(path, {"algo": "eALS", "K": 32, "c0": 256.0, "alpha": 0.5, "lam": 0.03})

    rows = list(csv.DictReader(path.open()))
    assert rows[0] == {"algo": "eALS", "K": "16", "c0": "512.0", "alpha": "0.4", "lam": "0.01"}
    assert rows[1] == {"algo": "MLlib-ALS", "K": "16", "c0": "", "alpha": "", "lam": "0.02"}
    assert rows[2] == {"algo": "eALS", "K": "32", "c0": "256.0", "alpha": "0.5", "lam": "0.03"}


def test_append_csv_widens_header_when_a_later_row_adds_columns(tmp_path):
    path = tmp_path / "r.csv"
    xp.append_csv(path, {"algo": "MLlib-ALS", "K": 16})  # written first, fewer columns
    xp.append_csv(path, {"algo": "eALS", "K": 32, "c0": 512.0})  # introduces c0 afterward

    with path.open() as fh:
        header = next(csv.reader(fh))
    assert header == ["algo", "K", "c0"]  # new column appended, existing order preserved
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["c0"] == ""  # backfilled for the row written before the column existed
    assert rows[1]["c0"] == "512.0"
