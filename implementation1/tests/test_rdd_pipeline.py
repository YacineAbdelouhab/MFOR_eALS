from __future__ import annotations

import numpy as np

from pyspark_eals_rdd.config import EALSConfig
from pyspark_eals_rdd.data import prepare_interactions, temporal_leave_one_out, universe_stats
from pyspark_eals_rdd.model import RDDDistributedEALS
from pyspark_eals_rdd.online import RDDOnlineUpdater


def events(sc):
    rows = []
    for user_id in range(6):
        for position in range(4):
            rows.append(
                (
                    user_id,
                    (user_id * 2 + position) % 8,
                    float(1 + position % 2),
                    float(1_700_000_000 + position * 100),
                    f"{user_id}-{position}",
                )
            )
    return sc.parallelize(rows, 2)


def test_split_and_offline_online(sc) -> None:
    train_events, test_events = temporal_leave_one_out(events(sc), min_interactions=3, partitions=2)
    assert test_events.map(lambda row: (row[0], 1)).reduceByKey(lambda a, b: a + b).values().filter(
        lambda count: count != 1
    ).isEmpty()
    config = EALSConfig(
        rank=3,
        max_iterations=2,
        partitions=2,
        checkpoint_interval=0,
        convergence_tolerance=0.0,
        seed=9,
    )
    interactions = prepare_interactions(train_events, config).cache()
    universe = universe_stats(
        interactions,
        sc.parallelize(range(10), 2),
        sc.parallelize(range(12), 2),
    )
    assert universe.registered_users == 10
    assert universe.registered_items == 12
    assert universe.active_users == 6
    model = RDDDistributedEALS(config).fit(interactions)
    assert model.RDD_users.count() == 6
    assert len(model.training_history) == 2
    assert model.training_history[-1]["objective"] <= model.training_history[0]["objective"]
    score = model.score(sc.parallelize([(0, 0)])).first()[2]
    assert np.isfinite(score)

    update = sc.parallelize([(0, 7, 1.0, 1_800_000_000.0, "online-1")])
    result = RDDOnlineUpdater(model).update_batch(interactions, update, batch_id=1)
    assert result.metrics["events"] == 1.0
    assert result.model.RDD_users.count() == 6
