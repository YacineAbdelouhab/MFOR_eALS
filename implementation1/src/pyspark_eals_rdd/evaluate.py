from __future__ import annotations

import math

import numpy as np
from pyspark import RDD


def _set_create(item_id: int) -> set[int]:
    return {item_id}


def _set_add(values: set[int], item_id: int) -> set[int]:
    values.add(item_id)
    return values


def _set_merge(left: set[int], right: set[int]) -> set[int]:
    left.update(right)
    return left


def metriques_catalogue_complet(
    model,
    positives: RDD[tuple[int, int]],
    train: RDD[tuple],
    *,
    k: int = 100,
    popularite: bool = False,
) -> dict[str, float]:
    """HR@k et NDCG@k en classant contre tout le catalogue, comme le papier.

    Le protocole est un leave-one-out, chaque utilisateur possede donc
    exactement un item de test. Son rang se lit en comptant les items du
    catalogue mieux places que lui, ceux deja vus a l'apprentissage etant
    ecartes. Le DCG ideal vaut 1, le NDCG se reduit donc a 1 / log2(rang + 1).

    Aucun shuffle n'est necessaire pour le classement. La matrice Q pese
    quelques megaoctets, on la diffuse et chaque utilisateur est traite en
    memoire par un simple produit matrice-vecteur.
    """
    items = model.RDD_items.collect()
    Q = np.array([valeur[0] for _identifiant, valeur in items], dtype=np.float64)
    frequences = np.array([valeur[2] for _identifiant, valeur in items], dtype=np.float64)
    colonne = {int(identifiant): rang for rang, (identifiant, _valeur) in enumerate(items)}

    diffuse_Q = model.sc.broadcast(Q)
    diffuse_frequences = model.sc.broadcast(frequences)
    diffuse_colonne = model.sc.broadcast(colonne)

    vus = train.map(lambda inter: (inter[0], inter[1])).combineByKey(
        _set_create, _set_add, _set_merge
    )
    entrees = positives.join(model.RDD_users).join(vus)

    def evalue(ligne: tuple) -> tuple[float, float, float]:
        _u, ((i_test, p_u), vus_u) = ligne
        positions = diffuse_colonne.value
        position_test = positions[int(i_test)]
        scores = diffuse_frequences.value if popularite else diffuse_Q.value @ p_u
        reference = scores[position_test]
        # Les ex aequo sont departages par la position dans le catalogue, ce que
        # ferait un tri stable. Sans cela la popularite, riche en egalites,
        # serait avantagee.
        meilleurs = scores > reference
        meilleurs |= (scores == reference) & (np.arange(scores.size) < position_test)
        for item_vu in vus_u:
            position = positions.get(int(item_vu))
            if position is not None:
                meilleurs[position] = False
        rang = 1 + int(meilleurs.sum())
        if rang > k:
            return 0.0, 0.0, 1.0
        return 1.0, 1.0 / math.log2(rang + 1.0), 1.0

    totaux = entrees.map(evalue).fold(
        (0.0, 0.0, 0.0),
        lambda gauche, droite: tuple(a + b for a, b in zip(gauche, droite, strict=True)),
    )
    utilisateurs = totaux[2]
    return {
        f"hit_rate@{k}": totaux[0] / utilisateurs if utilisateurs else 0.0,
        f"ndcg@{k}": totaux[1] / utilisateurs if utilisateurs else 0.0,
        "evaluated_users": utilisateurs,
    }
