from __future__ import annotations

import argparse
import json

from pyspark_eals_rdd.jobs.common import create_context


def _mix(user_id: int, draw: int, seed: int) -> int:
    """Hachage deterministe, pour un tirage reproductible sans etat partage."""
    value = (user_id * 11_400_714_819_323_198_485 + draw * 1_099_511_628_211 + seed) & (
        (1 << 64) - 1
    )
    value ^= value >> 30
    value *= 13_764_885_681_467_668_597
    return value & ((1 << 64) - 1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate clustered JSON events with RDDs")
    result.add_argument("--output", required=True)
    result.add_argument("--users", type=int, default=1_000)
    result.add_argument("--items", type=int, default=500)
    result.add_argument("--events-per-user", type=int, default=20)
    result.add_argument("--clusters", type=int, default=10)
    result.add_argument("--partitions", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.users, args.items, args.events_per_user, args.clusters, args.partitions) <= 0:
        raise ValueError("all sizes must be positive")
    sc = create_context("eals-rdd-generate")
    cluster_width = max(1, args.items // args.clusters)

    def events_for_user(user_id: int):
        cluster = user_id % args.clusters
        for draw in range(args.events_per_user):
            if draw % 5 < 4:
                item_id = (cluster * cluster_width + _mix(user_id, draw, args.seed) % cluster_width)
            else:
                item_id = _mix(user_id, draw, args.seed + 1) % args.items
            item_id %= args.items
            yield json.dumps(
                {
                    "event_id": f"{user_id}-{draw}",
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_ts": 1_700_000_000 + draw * 86_400 + user_id % 86_400,
                    "strength": float(1 + _mix(user_id, item_id, draw) % 3),
                },
                separators=(",", ":"),
            )

    events = sc.range(0, args.users, numSlices=args.partitions).flatMap(events_for_user)
    count = events.count()
    events.saveAsTextFile(args.output)
    print(f"generated_events={count} output={args.output}")
    sc.stop()


if __name__ == "__main__":
    main()

