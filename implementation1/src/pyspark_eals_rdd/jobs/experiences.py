"""Les trois experiences du rapport, une commande chacune.

Chaque experience affiche son tableau sur la sortie standard et ecrit le
detail dans un fichier JSON. Les reglages sont fixes ici, le lanceur n'a que
le fichier d'entree a fournir.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

from pyspark import StorageLevel

from pyspark_eals_rdd.config import EALSConfig
from pyspark_eals_rdd.data import (
    dataset_stats,
    prepare_interactions,
    read_events,
    temporal_leave_one_out,
)
from pyspark_eals_rdd.evaluate import metriques_catalogue_complet
from pyspark_eals_rdd.jobs.common import add_input_arguments, create_context
from pyspark_eals_rdd.model import RDDDistributedEALS
from pyspark_eals_rdd.online import RDDOnlineUpdater

# Les reglages du rapport. La tolerance nulle desactive l'arret anticipe, le
# budget d'iterations est donc respecte a la lettre.
CONFIG_RAPPORT: dict[str, Any] = {
    "rank": 16,
    "regularization": 0.01,
    "max_iterations": 20,
    "inner_sweeps": 1,
    "alpha": 0.4,
    "c0": 512.0,
    "positive_confidence_base": 1.0,
    "positive_confidence_scale": 0.0,
    "convergence_tolerance": 0.0,
    "convergence_patience": 2,
    "seed": 42,
    "factor_scale": 0.01,
    "partitions": 0,
    "checkpoint_interval": 2,
    "max_factor_norm": 100.0,
    "initialisation": "aleatoire",
}


def parseur(description: str, sortie: str) -> argparse.ArgumentParser:
    resultat = argparse.ArgumentParser(description=description)
    add_input_arguments(resultat)
    resultat.set_defaults(input_format="paper-rating")
    resultat.add_argument("--sortie", default=sortie)
    resultat.add_argument("--iterations", type=int, default=20)
    resultat.add_argument("--k", type=int, default=100)
    resultat.add_argument("--checkpoint", default="/tmp/eals-checkpoints")
    return resultat


def configurer(**remplacements: Any) -> EALSConfig:
    valeurs = dict(CONFIG_RAPPORT)
    valeurs.update(remplacements)
    return EALSConfig.from_dict(valeurs)


def preparer(sc, args) -> tuple:
    """Lit le fichier, coupe en apprentissage et test, affiche les tailles."""
    evenements = read_events(
        sc,
        args.input,
        input_format=args.input_format,
        delimiter=args.delimiter,
        header=args.header,
    ).persist(StorageLevel.MEMORY_AND_DISK)
    evenements_train, evenements_test = temporal_leave_one_out(evenements)
    train = prepare_interactions(evenements_train, configurer()).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    stats = dataset_stats(train)
    print(
        f"Donnees : {stats.users} utilisateurs, {stats.items} items, "
        f"{stats.interactions} interactions d'apprentissage",
        flush=True,
    )
    return train, evenements_test.persist(StorageLevel.MEMORY_AND_DISK), stats


def entrainer(config: EALSConfig, train, etiquette: str):
    """Lance fit en affichant une ligne par iteration."""
    print(f"\n--- {etiquette}", flush=True)
    print("Iteration   fonction de cout   secondes", flush=True)

    def afficher(ligne: dict) -> None:
        print(
            f"{int(ligne['iteration']):>8}   {ligne['objective']:>16.2f}   "
            f"{ligne['seconds']:>8.1f}",
            flush=True,
        )

    debut = time.perf_counter()
    modele = RDDDistributedEALS(config).fit(train, progression=afficher)
    modele.metadata["fit_seconds"] = time.perf_counter() - debut
    return modele


def positifs(evenements_test, modele):
    """Les couples (u, i) de test dont u et i sont connus du modele."""
    par_utilisateur = evenements_test.map(lambda evenement: (evenement[0], evenement[1]))
    par_item = par_utilisateur.join(modele.RDD_users).map(lambda ligne: (ligne[1][0], ligne[0]))
    return par_item.join(modele.RDD_items).map(lambda ligne: (ligne[1][0], ligne[0])).persist(
        StorageLevel.MEMORY_AND_DISK
    )


def secondes_par_iteration(modele) -> float:
    duree = [ligne["seconds"] for ligne in modele.training_history]
    return sum(duree) / len(duree) if duree else 0.0


def tableau(titre: str, entetes: list[str], lignes: list[list[str]]) -> str:
    """Met en forme un tableau texte a colonnes alignees."""
    colonnes = [
        max(len(entetes[index]), *(len(ligne[index]) for ligne in lignes))
        for index in range(len(entetes))
    ]
    trait = "-+-".join("-" * largeur for largeur in colonnes)
    rendu = [
        "",
        titre,
        " | ".join(entete.ljust(largeur) for entete, largeur in zip(entetes, colonnes)),
        trait,
    ]
    for ligne in lignes:
        rendu.append(
            " | ".join(cellule.ljust(largeur) for cellule, largeur in zip(ligne, colonnes))
        )
    return "\n".join(rendu)


def ecrire(chemin: str, contenu: dict) -> None:
    with open(chemin, "w", encoding="utf-8") as flux:
        json.dump(contenu, flux, indent=2)
    print(f"\nDetail ecrit dans {chemin}", flush=True)


def lancer(nom: str, description: str, sortie: str, corps: Callable) -> None:
    args = parseur(description, sortie).parse_args()
    sc = create_context(nom, configurer())
    sc.setCheckpointDir(args.checkpoint)
    train, evenements_test, stats = preparer(sc, args)
    corps(args, train, evenements_test, stats)
    sc.stop()


# --------------------------------------------------------------- experience 1


def _qualite(args, train, evenements_test, stats) -> None:
    rangs = (8, 16, 32)
    modeles = {}
    for rang in rangs:
        config = configurer(rank=rang, max_iterations=args.iterations)
        modeles[rang] = entrainer(config, train, f"eALS K = {rang}")

    reference = modeles[16]
    test = positifs(evenements_test, reference)
    # Les f_i ne dependent pas de l'apprentissage, n'importe quel modele sert
    # de support au classement par popularite.
    mesures = {
        "popularite": metriques_catalogue_complet(
            reference, test, train, k=args.k, popularite=True
        )
    }
    for rang in rangs:
        mesures[f"eALS K={rang}"] = metriques_catalogue_complet(
            modeles[rang], test, train, k=args.k
        )

    lignes = [
        [
            "Popularite",
            f"{mesures['popularite'][f'hit_rate@{args.k}']:.4f}",
            f"{mesures['popularite'][f'ndcg@{args.k}']:.4f}",
            "-",
            "-",
        ]
    ]
    for rang in rangs:
        modele = modeles[rang]
        cle = f"eALS K={rang}"
        lignes.append(
            [
                f"eALS K = {rang}",
                f"{mesures[cle][f'hit_rate@{args.k}']:.4f}",
                f"{mesures[cle][f'ndcg@{args.k}']:.4f}",
                f"{secondes_par_iteration(modele):.1f}",
                f"{modele.metadata['fit_seconds']:.0f}",
            ]
        )

    print(
        tableau(
            f"Tableau 5.1 - Qualite des recommandations apres {args.iterations} iterations",
            ["Methode", f"HR@{args.k}", f"NDCG@{args.k}", "s / iteration", "Total (s)"],
            lignes,
        )
    )
    ecrire(
        args.sortie,
        {
            "donnees": {
                "utilisateurs": stats.users,
                "items": stats.items,
                "interactions": stats.interactions,
            },
            "iterations": args.iterations,
            "k": args.k,
            "mesures": mesures,
            "historiques": {
                f"K={rang}": modeles[rang].training_history for rang in rangs
            },
        },
    )


def main_qualite() -> None:
    lancer(
        "eals-xp1-qualite",
        "Experience 1, qualite des recommandations selon K",
        "/data/xp1.json",
        _qualite,
    )


# --------------------------------------------------------------- experience 2


def _initialisation(args, train, evenements_test, stats) -> None:
    modeles = {}
    for mode in ("aleatoire", "svd"):
        config = configurer(rank=16, max_iterations=args.iterations, initialisation=mode)
        modeles[mode] = entrainer(config, train, f"initialisation {mode}")

    test = positifs(evenements_test, modeles["aleatoire"])
    mesures = {
        mode: metriques_catalogue_complet(modeles[mode], test, train, k=args.k)
        for mode in modeles
    }

    def colonne(mode: str) -> list[str]:
        modele = modeles[mode]
        return [
            f"{modele.metadata['init_seconds']:.1f}",
            f"{modele.metadata['fit_seconds']:.0f}",
            f"{modele.training_history[-1]['objective']:.0f}",
            f"{mesures[mode][f'hit_rate@{args.k}']:.4f}",
            f"{mesures[mode][f'ndcg@{args.k}']:.4f}",
        ]

    intitules = [
        "Temps d'initialisation (s)",
        "Temps total, initialisation comprise (s)",
        f"Fonction de cout apres {args.iterations} iterations",
        f"HR@{args.k}",
        f"NDCG@{args.k}",
    ]
    gauche, droite = colonne("aleatoire"), colonne("svd")
    lignes = [[intitules[rang], gauche[rang], droite[rang]] for rang in range(len(intitules))]

    print(
        tableau(
            "Tableau 5.2 - Comparaison des deux initialisations, K = 16",
            ["Mesure", "Aleatoire", "SVD randomisee"],
            lignes,
        )
    )
    ecrire(
        args.sortie,
        {
            "iterations": args.iterations,
            "k": args.k,
            "mesures": mesures,
            "init_seconds": {
                mode: modeles[mode].metadata["init_seconds"] for mode in modeles
            },
            "historiques": {mode: modeles[mode].training_history for mode in modeles},
        },
    )


def main_initialisation() -> None:
    lancer(
        "eals-xp2-initialisation",
        "Experience 2, interet de l'initialisation par SVD randomisee",
        "/data/xp2.json",
        _initialisation,
    )


# --------------------------------------------------------------- experience 3


def _online(args, train, evenements_test, stats) -> None:
    config = configurer(rank=16, max_iterations=args.iterations)
    modele = entrainer(config, train, "eALS K = 16")

    nouvelles = evenements_test.context.parallelize(evenements_test.take(args.nouvelles))
    resultat = RDDOnlineUpdater(modele).update_batch(train, nouvelles, batch_id=0)
    secondes = resultat.metrics["seconds"]
    ajoutees = int(resultat.metrics["events"])

    lignes = [
        ["Moteur", "Spark local"],
        ["Interactions ajoutees", str(ajoutees)],
        ["Temps total de mise a jour (s)", f"{secondes:.2f}"],
        ["Temps moyen par interaction (ms)", f"{1000.0 * secondes / max(ajoutees, 1):.1f}"],
    ]
    print(
        tableau(
            f"Tableau 5.3 - Traitement de {ajoutees} nouvelles interactions",
            ["Mesure", "Valeur"],
            lignes,
        )
    )
    ecrire(
        args.sortie,
        {
            "iterations": args.iterations,
            "online": resultat.metrics,
            "historique": modele.training_history,
        },
    )
    resultat.history.unpersist()


def main_online() -> None:
    analyseur = parseur(
        "Experience 3, cout de la mise a jour online", "/data/xp3.json"
    )
    analyseur.add_argument("--nouvelles", type=int, default=100)
    args = analyseur.parse_args()
    sc = create_context("eals-xp3-online", configurer())
    sc.setCheckpointDir(args.checkpoint)
    train, evenements_test, stats = preparer(sc, args)
    _online(args, train, evenements_test, stats)
    sc.stop()
