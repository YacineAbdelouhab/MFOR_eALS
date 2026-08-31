"""Demonstration de bout en bout : apprentissage offline puis mise a jour online.

Le script entraine eALS sur un petit jeu synthetique et affiche, iteration par
iteration, la valeur de la fonction de cout. Sa decroissance stricte est la
verification la plus directe que le portage MapReduce est correct : une
implementation fautive, qui mettrait a jour P et Q en parallele ou qui
utiliserait un cache perime, ferait remonter cette valeur.

Usage : python scripts/smoke_job.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from pyspark import SparkConf, SparkContext

from pyspark_eals_rdd.config import EALSConfig
from pyspark_eals_rdd.data import prepare_interactions
from pyspark_eals_rdd.model import RDDDistributedEALS, compute_cache
from pyspark_eals_rdd.online import RDDOnlineUpdater

RANK = 8
ITERATIONS = 6
USERS = 40
ITEMS = 25


def build_spark() -> SparkContext:
    """Cree un contexte Spark local.

    PYSPARK_PYTHON doit designer l'interpreteur courant. Sans cela Spark lance
    ses executeurs avec `python3`, qui peut ne pas avoir PySpark installe.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    conf = (
        SparkConf()
        .setMaster("local[2]")
        .setAppName("eals-demo")
        .set("spark.ui.enabled", "false")
        .set("spark.default.parallelism", "2")
    )
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")
    return sc


def synthetic_rows() -> list:
    """Genere des interactions ou chaque utilisateur suit un motif regulier.

    Le motif cree une structure que la factorisation doit retrouver, sinon la
    fonction de cout stagnerait.
    """
    return [
        (user, (user * 3 + step) % ITEMS, 1.0, float(step), f"{user}-{step}")
        for user in range(USERS)
        for step in range(6)
    ]


def main() -> None:
    sc = build_spark()
    try:
        config = EALSConfig(
            rank=RANK,
            max_iterations=ITERATIONS,
            partitions=2,
            checkpoint_interval=0,
            convergence_tolerance=0.0,
            positive_confidence_scale=0.0,
        )

        rows = synthetic_rows()
        history = prepare_interactions(sc.parallelize(rows, 2), config).cache()
        print(f"Jeu de donnees : {USERS} utilisateurs, {ITEMS} items, "
              f"{history.count()} interactions, K = {RANK}\n")

        # ---- Apprentissage offline -------------------------------------
        model = RDDDistributedEALS(config).fit(history)

        print("Iteration   fonction de cout   variation")
        previous = None
        decroissante = True
        for record in model.training_history:
            value = record["objective"]
            if previous is None:
                print(f"    {int(record['iteration']):2d}      {value:14.6f}          -")
            else:
                delta = value - previous
                decroissante &= delta <= 1e-9
                print(f"    {int(record['iteration']):2d}      {value:14.6f}   {delta:+12.6f}")
            previous = value
        print(f"\nDecroissance monotone : {'oui' if decroissante else 'NON'}\n")

        # ---- Mise a jour online ----------------------------------------
        # Une nouvelle interaction arrive. On ne reapprend pas le modele, on
        # met a jour les seuls p_u et q_i concernes, puis on corrige les caches.
        evenements = [
            (user, (user * 7 + 3) % ITEMS, 1.0, 99.0 + user, f"online-{user}")
            for user in range(20)
        ]
        nouveaux = sc.parallelize(evenements, 1)
        depart = time.perf_counter()
        online = RDDOnlineUpdater(model).update_batch(history, nouveaux, batch_id=1)
        duree = time.perf_counter() - depart

        # Le cache corrige doit coincider avec un recalcul complet.
        cache_recalcule = compute_cache(
            online.model.RDD_users, RANK, lambda value: (value, 1.0)
        )
        ecart = np.linalg.norm(online.model.sp - cache_recalcule)
        ecart /= max(np.linalg.norm(cache_recalcule), 1e-12)

        print(f"Mise a jour online : {len(evenements)} evenements en {duree:.1f} s")
        print("Ce temps est domine par l'ordonnancement Spark en mode local, pas")
        print("par le calcul. Le cout algorithmique par evenement, mesure sur les")
        print("fonctions de mise a jour seules, est inferieur a la milliseconde (voir le rapport).")
        print(f"Ecart relatif du cache Sp apres correction : {ecart:.2e}")
        print(f"Cache exact a la precision machine : {'oui' if ecart < 1e-10 else 'NON'}")
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
