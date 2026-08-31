"""Point d'entrée volontairement court pour une lecture de type examen."""

from __future__ import annotations

import argparse

from pyspark import StorageLevel

from pyspark_eals_rdd.config import EALSConfig
from pyspark_eals_rdd.data import prepare_interactions, read_events, universe_stats
from pyspark_eals_rdd.jobs.common import create_context
from pyspark_eals_rdd.model import RDDDistributedEALS


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="eALS simple avec les RDD du rapport")
    result.add_argument("--input", required=True)
    result.add_argument("--input-format", choices=("json", "csv", "paper-rating"), default="json")
    result.add_argument("--config", required=True)
    result.add_argument("--user-ids", help="fichier texte du référentiel utilisateur")
    result.add_argument("--item-ids", help="fichier texte du référentiel item")
    result.add_argument("--checkpoint", default="checkpoints/eals-simple")
    return result


def main() -> None:
    args = parser().parse_args()
    config = EALSConfig.from_json(args.config)
    sc = create_context("eals-rdd-simple", config)
    sc.setCheckpointDir(args.checkpoint)

    # RDD_interactions : RDD[(u, i, r_ui, w_ui, strength, timestamp)]
    RDD_events = read_events(sc, args.input, input_format=args.input_format)
    RDD_interactions = prepare_interactions(RDD_events, config).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    RDD_interactions.count()

    # M et N viennent des référentiels s'ils sont disponibles.
    RDD_user_ids = (
        sc.textFile(args.user_ids).map(int)
        if args.user_ids
        else RDD_interactions.map(lambda x: x[0]).distinct()
    )
    RDD_item_ids = (
        sc.textFile(args.item_ids).map(int)
        if args.item_ids
        else RDD_interactions.map(lambda x: x[1]).distinct()
    )
    stats = universe_stats(RDD_interactions, RDD_user_ids, RDD_item_ids)

    model = RDDDistributedEALS(config).fit(RDD_interactions)

    # Noms utilisés dans le rapport.
    RDD_users = model.RDD_users          # PairRDD[(u, p_u)]
    RDD_items = model.RDD_items          # PairRDD[(i, (q_i, c_i, f_i))]
    Sq, Sp = model.sq, model.sp              # tableaux K x K sur le driver

    print(f"M={stats.registered_users}, N={stats.registered_items}, K={config.rank}")
    print(f"actifs: users={RDD_users.count()}, items={RDD_items.count()}")
    print(f"caches: Sq={Sq.shape}, Sp={Sp.shape}")
    for row in model.training_history:
        print(
            f"it={int(row['iteration']):02d}  "
            f"L={row['objective']:.6f}  "
            f"amelioration={row['relative_improvement']:.3e}"
        )
    print(
        f"converged={model.metadata['converged']}  "
        f"stop_iteration={model.metadata['stop_iteration']}"
    )
    sc.stop()


if __name__ == "__main__":
    main()
