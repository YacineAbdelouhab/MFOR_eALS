from __future__ import annotations

import argparse

from pyspark import StorageLevel

from pyspark_eals_rdd.config import EALSConfig
from pyspark_eals_rdd.data import dataset_stats, prepare_interactions, read_events
from pyspark_eals_rdd.jobs.common import (
    add_config_argument,
    add_input_arguments,
    create_context,
)
from pyspark_eals_rdd.model import RDDDistributedEALS


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train eALS with Pair RDDs only")
    add_input_arguments(result)
    add_config_argument(result)
    result.add_argument("--model-output", required=True)
    result.add_argument("--history-output")
    result.add_argument("--checkpoint", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    config = EALSConfig.from_json(args.config)
    sc = create_context("eals-rdd-train", config)
    sc.setCheckpointDir(args.checkpoint)
    events = read_events(
        sc,
        args.input,
        input_format=args.input_format,
        delimiter=args.delimiter,
        header=args.header,
    )
    interactions = prepare_interactions(events, config).persist(StorageLevel.MEMORY_AND_DISK)
    stats = dataset_stats(interactions)
    print(
        f"Donnees : {stats.users} utilisateurs, {stats.items} items, "
        f"{stats.interactions} interactions"
    )
    print(f"\nIteration   fonction de cout   gain relatif   secondes")

    def afficher(ligne: dict) -> None:
        print(
            f"{int(ligne['iteration']):>8}   {ligne['objective']:>16.2f}   "
            f"{ligne['relative_improvement']:>11.5f}   {ligne['seconds']:>7.1f}",
            flush=True,
        )

    model = RDDDistributedEALS(config).fit(interactions, progression=afficher)
    model.metadata["density"] = stats.density
    model.save(args.model_output)
    if args.history_output:
        interactions.saveAsPickleFile(args.history_output)
    converge = model.metadata.get("converged", False)
    print(
        f"\nArret apres {len(model.training_history)} iterations, "
        f"convergence atteinte : {'oui' if converge else 'non'}"
    )
    print(f"Modele ecrit dans {args.model_output}")
    interactions.unpersist()
    sc.stop()


if __name__ == "__main__":
    main()

