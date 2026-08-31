from __future__ import annotations

import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Iterable, TypeAlias

from pyspark import RDD, SparkContext

from pyspark_eals_rdd.config import EALSConfig

RawEvent: TypeAlias = tuple[int, int, float, float, str | None]
Interaction: TypeAlias = tuple[int, int, float, float, float, float]


@dataclass(frozen=True)
class DatasetStats:
    users: int
    items: int
    interactions: int
    density: float


@dataclass(frozen=True)
class UniverseStats:
    """Sépare l'univers métier complet de la partie observée dans R."""

    registered_users: int
    registered_items: int
    active_users: int
    active_items: int
    interactions: int
    catalog_density: float


def _timestamp(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def parse_json_event(line: str) -> RawEvent:
    row = json.loads(line)
    user_id, item_id = int(row["user_id"]), int(row["item_id"])
    strength = float(row.get("strength", 1.0))
    if strength <= 0 or not math.isfinite(strength):
        raise ValueError("strength must be finite and positive")
    return user_id, item_id, strength, _timestamp(row.get("event_ts")), row.get("event_id")


def parse_csv_event(line: str, delimiter: str = ",") -> RawEvent:
    fields = next(csv.reader([line], delimiter=delimiter))
    if len(fields) < 2:
        raise ValueError("CSV needs at least user_id,item_id")
    strength = float(fields[2]) if len(fields) > 2 and fields[2] else 1.0
    if strength <= 0 or not math.isfinite(strength):
        raise ValueError("strength must be finite and positive")
    return (
        int(fields[0]),
        int(fields[1]),
        strength,
        _timestamp(fields[3] if len(fields) > 3 else None),
        fields[4] if len(fields) > 4 and fields[4] else None,
    )


def parse_paper_rating_event(line: str, delimiter: str = "\t") -> RawEvent:
    """Lit un fichier ``*.rating`` publié avec l'article eALS.

    La note explicite du troisième champ est volontairement ignorée : comme dans
    le protocole du papier, chaque ligne devient un signal implicite de force 1.
    """
    fields = next(csv.reader([line], delimiter=delimiter))
    if len(fields) < 2:
        raise ValueError("paper-rating needs at least user_id and item_id")
    return (
        int(fields[0]),
        int(fields[1]),
        1.0,
        _timestamp(fields[3] if len(fields) > 3 else None),
        None,
    )


def read_events(
    sc: SparkContext,
    path: str,
    *,
    input_format: str = "json",
    delimiter: str = ",",
    header: bool = False,
    min_partitions: int | None = None,
) -> RDD[RawEvent]:
    lines = sc.textFile(path, min_partitions or sc.defaultParallelism).filter(lambda x: bool(x.strip()))
    if header:
        first = lines.first()
        lines = lines.filter(lambda line: line != first)
    if input_format == "json":
        return lines.map(parse_json_event)
    if input_format == "csv":
        return lines.map(lambda line: parse_csv_event(line, delimiter))
    if input_format == "paper-rating":
        paper_delimiter = delimiter if delimiter != "," else "\t"
        return lines.map(lambda line: parse_paper_rating_event(line, paper_delimiter))
    raise ValueError("input_format must be json, csv or paper-rating")


def _deduplicate(events: RDD[RawEvent], partitions: int) -> RDD[RawEvent]:
    keyed = events.zipWithUniqueId().map(
        lambda pair: (
            ("event", pair[0][4]) if pair[0][4] is not None else ("row", pair[1]),
            pair[0],
        )
    )
    return keyed.reduceByKey(lambda left, _right: left, partitions).values()


def prepare_interactions(events: RDD[RawEvent], config: EALSConfig) -> RDD[Interaction]:
    partitions = config.partitions or events.context.defaultParallelism
    deduplicated = _deduplicate(events, partitions)
    aggregated = deduplicated.map(
        lambda row: ((row[0], row[1]), (float(row[2]), float(row[3])))
    ).reduceByKey(
        lambda left, right: (left[0] + right[0], max(left[1], right[1])), partitions
    )

    def finish(row: tuple[tuple[int, int], tuple[float, float]]) -> Interaction:
        (user_id, item_id), (strength, timestamp) = row
        confidence = config.positive_confidence_base + config.positive_confidence_scale * math.log1p(
            strength
        )
        return user_id, item_id, 1.0, confidence, strength, timestamp

    return aggregated.map(finish)


def merge_history(
    history: RDD[Interaction], new_events: RDD[RawEvent], config: EALSConfig
) -> RDD[Interaction]:
    partitions = config.partitions or history.context.defaultParallelism
    new_interactions = prepare_interactions(new_events, config)
    combined = history.union(new_interactions).map(
        lambda row: ((row[0], row[1]), (row[4], row[5]))
    ).reduceByKey(
        lambda left, right: (left[0] + right[0], max(left[1], right[1])), partitions
    )

    def finish(row: tuple[tuple[int, int], tuple[float, float]]) -> Interaction:
        (user_id, item_id), (strength, timestamp) = row
        confidence = config.positive_confidence_base + config.positive_confidence_scale * math.log1p(
            strength
        )
        return user_id, item_id, 1.0, confidence, strength, timestamp

    return combined.map(finish)


def _create_list(value: tuple[int, float, float]) -> list[tuple[int, float, float]]:
    return [value]


def _append_list(
    values: list[tuple[int, float, float]], value: tuple[int, float, float]
) -> list[tuple[int, float, float]]:
    values.append(value)
    return values


def _merge_lists(
    left: list[tuple[int, float, float]], right: list[tuple[int, float, float]]
) -> list[tuple[int, float, float]]:
    left.extend(right)
    return left


def temporal_leave_one_out(
    events: RDD[RawEvent], *, min_interactions: int = 3, partitions: int | None = None
) -> tuple[RDD[RawEvent], RDD[RawEvent]]:
    if min_interactions < 2:
        raise ValueError("min_interactions must be at least 2")
    count = partitions or events.context.defaultParallelism
    pairs = _deduplicate(events, count).map(
        lambda row: ((row[0], row[1]), (row[2], row[3]))
    ).reduceByKey(lambda a, b: (a[0] + b[0], max(a[1], b[1])), count)
    histories = pairs.map(
        lambda row: (row[0][0], (row[0][1], row[1][0], row[1][1]))
    ).combineByKey(_create_list, _append_list, _merge_lists, count)

    def split(row: tuple[int, list[tuple[int, float, float]]]) -> Iterable[tuple[str, RawEvent]]:
        user_id, values = row
        if len(values) < min_interactions:
            return []
        ordered = sorted(values, key=lambda value: (value[2], value[0]))
        result: list[tuple[str, RawEvent]] = []
        for item_id, strength, timestamp in ordered[:-1]:
            result.append(("train", (user_id, item_id, strength, timestamp, None)))
        item_id, strength, timestamp = ordered[-1]
        result.append(("test", (user_id, item_id, strength, timestamp, None)))
        return result

    tagged = histories.flatMap(split).persist()
    return tagged.filter(lambda row: row[0] == "train").values(), tagged.filter(
        lambda row: row[0] == "test"
    ).values()


def dataset_stats(interactions: RDD[Interaction]) -> DatasetStats:
    users = interactions.map(lambda row: row[0]).distinct().count()
    items = interactions.map(lambda row: row[1]).distinct().count()
    count = interactions.count()
    density = count / (users * items) if users and items else 0.0
    return DatasetStats(users, items, count, density)


def universe_stats(
    interactions: RDD[Interaction], user_ids: RDD[int], item_ids: RDD[int]
) -> UniverseStats:
    """Calcule M et N depuis les référentiels, jamais depuis R seulement."""
    registered_users = user_ids.distinct().count()
    registered_items = item_ids.distinct().count()
    active_users = interactions.map(lambda row: row[0]).distinct().count()
    active_items = interactions.map(lambda row: row[1]).distinct().count()
    count = interactions.count()
    density = (
        count / (registered_users * registered_items)
        if registered_users and registered_items
        else 0.0
    )
    return UniverseStats(
        registered_users,
        registered_items,
        active_users,
        active_items,
        count,
        density,
    )
