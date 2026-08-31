from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pyspark import RDD, StorageLevel

from pyspark_eals_rdd.data import Interaction, RawEvent, merge_history
from pyspark_eals_rdd.model import (
    EALSRDDModel,
    ItemFactors,
    RDDDistributedEALS,
    UserFactors,
    compute_cache,
    deterministic_factor,
)


@dataclass
class RDDOnlineResult:
    model: EALSRDDModel
    history: RDD[Interaction]
    metrics: dict[str, float]


class RDDOnlineUpdater:
    def __init__(self, model: EALSRDDModel):
        self.model = model
        self.trainer = RDDDistributedEALS(model.config)
        self.trainer.partitions = model.config.partitions or model.sc.defaultParallelism

    def _cache_or_zero(self, rdd, extractor) -> np.ndarray:
        if rdd.isEmpty():
            return np.zeros((self.model.config.rank, self.model.config.rank), dtype=np.float64)
        return compute_cache(rdd, self.model.config.rank, extractor)

    def _all_users(self, touched: RDD[tuple[int, None]]) -> UserFactors:
        new_ids = touched.subtractByKey(self.model.RDD_users)
        # Les valeurs de configuration sont extraites AVANT la closure. Sinon la
        # lambda capture self, donc self.model, donc self.model.sc, et Spark
        # refuse de serialiser un SparkContext vers les executeurs (SPARK-5063).
        rank = self.model.config.rank
        seed = self.model.config.seed
        scale = self.model.config.factor_scale
        new_factors = new_ids.map(
            lambda paire: (paire[0], deterministic_factor(paire[0], rank, seed, scale))
        )
        return self.model.RDD_users.union(new_factors).partitionBy(self.trainer.partitions)

    def _all_items(self, touched: RDD[tuple[int, None]]) -> ItemFactors:
        new_ids = touched.subtractByKey(self.model.RDD_items)
        item_count = max(int(self.model.metadata.get("items", 1)), 1)
        default_missing = self.model.config.c0 / item_count
        # Meme precaution que dans _all_users, on ne capture pas self.
        rank = self.model.config.rank
        seed = self.model.config.seed + 1
        scale = self.model.config.factor_scale
        new_factors = new_ids.map(
            lambda paire: (
                paire[0],
                (deterministic_factor(paire[0], rank, seed, scale), default_missing, 0.0),
            )
        )
        return self.model.RDD_items.union(new_factors).partitionBy(self.trainer.partitions)

    def update_batch(
        self,
        history: RDD[Interaction],
        new_events: RDD[RawEvent],
        *,
        batch_id: int | None = None,
    ) -> RDDOnlineResult:
        started = time.perf_counter()
        event_count = new_events.count()
        if event_count == 0:
            return RDDOnlineResult(
                self.model,
                history,
                {"events": 0.0, "users": 0.0, "items": 0.0, "seconds": 0.0},
            )
        merged = merge_history(history, new_events, self.model.config).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        merged.count()
        touched_users = new_events.map(lambda evenement: (evenement[0], None)).reduceByKey(
            lambda left, _right: left, self.trainer.partitions
        ).persist(StorageLevel.MEMORY_AND_DISK)
        touched_items = new_events.map(lambda evenement: (evenement[1], None)).reduceByKey(
            lambda left, _right: left, self.trainer.partitions
        ).persist(StorageLevel.MEMORY_AND_DISK)
        user_count, item_count = touched_users.count(), touched_items.count()

        all_users = self._all_users(touched_users).persist(StorageLevel.MEMORY_AND_DISK)
        all_items = self._all_items(touched_items).persist(StorageLevel.MEMORY_AND_DISK)
        all_users.count()
        all_items.count()

        user_history = merged.map(
            lambda interaction: (interaction[0], interaction)
        ).join(touched_users).values().map(lambda paire: paire[0])
        updated_users = self.trainer._update_users(
            user_history, all_users, all_items, self.model.sq
        ).persist(StorageLevel.MEMORY_AND_DISK)
        updated_users.count()
        old_touched_users = self.model.RDD_users.join(touched_users).mapValues(
            lambda paire: paire[0]
        )
        sp = (
            self.model.sp
            - self._cache_or_zero(old_touched_users, lambda p_u: (p_u, 1.0))
            + self._cache_or_zero(updated_users, lambda p_u: (p_u, 1.0))
        )
        RDD_users = self.model.RDD_users.subtractByKey(touched_users).union(
            updated_users
        ).partitionBy(self.trainer.partitions).persist(StorageLevel.MEMORY_AND_DISK)
        RDD_users.count()

        item_history = merged.map(
            lambda interaction: (interaction[1], interaction)
        ).join(touched_items).values().map(lambda paire: paire[0])
        updated_items = self.trainer._update_items(
            item_history, RDD_users, all_items, sp
        ).persist(StorageLevel.MEMORY_AND_DISK)
        updated_items.count()
        old_touched_items = self.model.RDD_items.join(touched_items).mapValues(
            lambda paire: paire[0]
        )
        sq = (
            self.model.sq
            - self._cache_or_zero(old_touched_items, lambda item: (item[0], item[1]))
            + self._cache_or_zero(updated_items, lambda item: (item[0], item[1]))
        )
        RDD_items = self.model.RDD_items.subtractByKey(touched_items).union(
            updated_items
        ).partitionBy(self.trainer.partitions).persist(StorageLevel.MEMORY_AND_DISK)
        RDD_items.count()

        metadata = dict(self.model.metadata)
        metadata["last_online_batch_id"] = batch_id
        metadata["online_updates"] = int(metadata.get("online_updates", 0)) + 1
        metadata["users"] = RDD_users.count()
        metadata["items"] = RDD_items.count()
        next_model = EALSRDDModel(
            sc=self.model.sc,
            config=self.model.config,
            RDD_users=RDD_users,
            RDD_items=RDD_items,
            sq=sq,
            sp=sp,
            training_history=self.model.training_history,
            metadata=metadata,
        )
        self.model = next_model
        all_users.unpersist()
        all_items.unpersist()
        updated_users.unpersist()
        updated_items.unpersist()
        touched_users.unpersist()
        touched_items.unpersist()
        return RDDOnlineResult(
            next_model,
            merged,
            {
                "events": float(event_count),
                "users": float(user_count),
                "items": float(item_count),
                "seconds": float(time.perf_counter() - started),
            },
        )

