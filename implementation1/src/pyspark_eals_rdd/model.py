from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

import numpy as np
from pyspark import RDD, SparkContext, StorageLevel

from pyspark_eals_rdd.config import EALSConfig
from pyspark_eals_rdd.data import Interaction
from pyspark_eals_rdd.math_core import update_item_factor, update_user_factor

UserFactors: TypeAlias = RDD[tuple[int, np.ndarray]]
ItemValue: TypeAlias = tuple[np.ndarray, float, float]
ItemFactors: TypeAlias = RDD[tuple[int, ItemValue]]


def deterministic_factor(entity_id: int, rank: int, seed: int, scale: float) -> np.ndarray:
    mixed = (int(entity_id) * 11_400_714_819_323_198_485 + int(seed)) & ((1 << 64) - 1)
    return np.random.default_rng(mixed).uniform(-scale, scale, rank).astype(np.float64)


def persist_once(rdd: RDD, level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> RDD:
    """Persiste un RDD seulement s'il n'a pas deja un niveau de stockage.

    Spark leve SparkUnsupportedOperationException si on appelle persist() deux
    fois sur le meme RDD avec un niveau different de NONE.
    """
    if rdd.getStorageLevel() == StorageLevel(False, False, False, False, 1):
        rdd.persist(level)
    return rdd


def compute_cache(
    factors: RDD[tuple[int, Any]],
    rank: int,
    extractor: Callable[[Any], tuple[np.ndarray, float]],
) -> np.ndarray:
    """Agrège un cache K x K par arbre; aucune ligne de facteurs n'est collectée."""

    def seq(total: np.ndarray, row: tuple[int, Any]) -> np.ndarray:
        vector, weight = extractor(row[1])
        total += float(weight) * np.outer(vector, vector)
        return total

    def combine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left += right
        return left

    return factors.treeAggregate(
        np.zeros((rank, rank), dtype=np.float64), seq, combine
    )


def distributed_sum(factors: RDD[tuple[int, Any]], fn: Callable[[Any], float]) -> float:
    return float(
        factors.treeAggregate(
            0.0,
            lambda total, row: total + float(fn(row[1])),
            lambda left, right: left + right,
        )
    )


def _produit_avec_R(RDD_interactions, X, taille, cle, autre, partitions):
    """Calcule R X ou R^T X sans jamais materialiser R.

    cle vaut 0 pour la cle utilisateur, ce qui donne R X, et 1 pour la cle
    item, ce qui donne R^T X. Chaque interaction emet la ligne de X portee par
    l'autre entite, puis reduceByKey additionne. Le resultat est etroit, il
    tient sur le driver.
    """
    diffuse = RDD_interactions.context.broadcast(X)
    lignes = (
        RDD_interactions.map(lambda inter: (inter[cle], diffuse.value[inter[autre]]))
        .reduceByKey(lambda gauche, droite: gauche + droite, partitions)
        .collect()
    )
    resultat = np.zeros((taille, X.shape[1]), dtype=np.float64)
    for identifiant, vecteur in lignes:
        resultat[identifiant] = vecteur
    return resultat


def svd_randomisee(RDD_interactions, rank, partitions, seed, sur_echantillonnage=10):
    """SVD tronquee de rang K par esquisse aleatoire (Halko et al., 2011).

    La SVD tronquee donne le meilleur produit P Q^T de rang K au sens de la
    norme de Frobenius, ce qui en fait un point de depart naturel pour eALS.
    La version exacte demanderait R en dense. Celle-ci ne touche R que par
    deux produits avec des blocs de rank + 10 colonnes, soit deux shuffles.

    Renvoie deux tableaux denses, P de taille M x K et Q de taille N x K.
    """
    largeur = rank + sur_echantillonnage
    nb_users, nb_items = RDD_interactions.map(
        lambda inter: (inter[0], inter[1])
    ).reduce(lambda a, b: (max(a[0], b[0]), max(a[1], b[1])))
    nb_users, nb_items = nb_users + 1, nb_items + 1

    # Esquisse. Chaque colonne de Y est une combinaison aleatoire des colonnes
    # de R, donc penche vers les directions de grande valeur singuliere.
    Omega = np.random.default_rng(seed).normal(0.0, 1.0, (nb_items, largeur))
    Y = _produit_avec_R(RDD_interactions, Omega, nb_users, 0, 1, partitions)

    # Base orthonormee du sous-espace dominant. La QR porte sur M x largeur,
    # quelques megaoctets, elle se fait sur le driver.
    Base, _ = np.linalg.qr(Y)

    # Projection de R sur cette base, puis SVD exacte de la petite matrice.
    Ct = _produit_avec_R(RDD_interactions, Base, nb_items, 1, 0, partitions)
    Ub, valeurs, Vt = np.linalg.svd(Ct.T, full_matrices=False)
    U = Base @ Ub

    racine = np.sqrt(valeurs[:rank])
    return U[:, :rank] * racine, Vt[:rank].T * racine


def _item_and_one(interaction: Interaction) -> tuple[int, int]:
    """(u, i, r_ui, w_ui, ...) -> (i, 1). Le map du comptage de popularite."""
    u, i, r_ui, w_ui, *reste = interaction
    return (i, 1)


def _user_of(interaction: Interaction) -> int:
    """(u, i, r_ui, w_ui, ...) -> u."""
    u, i, r_ui, w_ui, *reste = interaction
    return u


def _interaction_by_item(interaction: Interaction) -> tuple[int, Any]:
    """(u, i, r_ui, w_ui, ...) -> (i, (u, r_ui, w_ui))."""
    u, i, r_ui, w_ui, *reste = interaction
    return (i, (u, r_ui, w_ui))


def _observation_by_user(ligne: Any) -> tuple[int, Any]:
    """(i, ((u, r_ui, w_ui), (q_i, c_i, f_i))) -> (u, (q_i, r_ui, w_ui, c_i))."""
    i, ((u, r_ui, w_ui), (q_i, c_i, f_i)) = ligne
    return (u, (q_i, r_ui, w_ui, c_i))


def _interaction_by_user(interaction: Interaction) -> tuple[int, Any]:
    """(u, i, r_ui, w_ui, ...) -> (u, (i, r_ui, w_ui))."""
    u, i, r_ui, w_ui, *reste = interaction
    return (u, (i, r_ui, w_ui))


def _observation_by_item(ligne: Any) -> tuple[int, Any]:
    """(u, ((i, r_ui, w_ui), p_u)) -> (i, (p_u, r_ui, w_ui))."""
    u, ((i, r_ui, w_ui), p_u) = ligne
    return (i, (p_u, r_ui, w_ui))


def _score_by_user(ligne: Any) -> tuple[int, Any]:
    """(i, ((u, r_ui, w_ui), (q_i, c_i, f_i))) -> (u, (r_ui, w_ui, q_i, c_i))."""
    i, ((u, r_ui, w_ui), (q_i, c_i, f_i)) = ligne
    return (u, (r_ui, w_ui, q_i, c_i))


def _correction(ligne: Any) -> float:
    """Terme de la loss pour une interaction observee, cache mis a part."""
    u, ((r_ui, w_ui, q_i, c_i), p_u) = ligne
    score = float(np.dot(p_u, q_i))
    return w_ui * (r_ui - score) ** 2 - c_i * score**2


def _pair_by_item(ligne: Any) -> tuple[int, Any]:
    """(u, (i, p_u)) -> (i, (u, p_u))."""
    u, (i, p_u) = ligne
    return (i, (u, p_u))


def _dot_score(ligne: Any) -> tuple[int, int, float]:
    """(i, ((u, p_u), (q_i, c_i, f_i))) -> (u, i, p_u . q_i)."""
    i, ((u, p_u), (q_i, c_i, f_i)) = ligne
    return (u, i, float(np.dot(p_u, q_i)))


def _create_user_observation(
    value: tuple[np.ndarray, float, float, float]
) -> tuple[list[np.ndarray], list[float], list[float], list[float]]:
    q, label, confidence, missing = value
    return [q], [label], [confidence], [missing]


def _append_user_observation(
    aggregate: tuple[list[np.ndarray], list[float], list[float], list[float]],
    value: tuple[np.ndarray, float, float, float],
) -> tuple[list[np.ndarray], list[float], list[float], list[float]]:
    q, label, confidence, missing = value
    aggregate[0].append(q)
    aggregate[1].append(label)
    aggregate[2].append(confidence)
    aggregate[3].append(missing)
    return aggregate


def _merge_user_observations(
    left: tuple[list[np.ndarray], list[float], list[float], list[float]],
    right: tuple[list[np.ndarray], list[float], list[float], list[float]],
) -> tuple[list[np.ndarray], list[float], list[float], list[float]]:
    for index in range(4):
        left[index].extend(right[index])
    return left


def _create_item_observation(
    value: tuple[np.ndarray, float, float]
) -> tuple[list[np.ndarray], list[float], list[float]]:
    p, label, confidence = value
    return [p], [label], [confidence]


def _append_item_observation(
    aggregate: tuple[list[np.ndarray], list[float], list[float]],
    value: tuple[np.ndarray, float, float],
) -> tuple[list[np.ndarray], list[float], list[float]]:
    p, label, confidence = value
    aggregate[0].append(p)
    aggregate[1].append(label)
    aggregate[2].append(confidence)
    return aggregate


def _merge_item_observations(
    left: tuple[list[np.ndarray], list[float], list[float]],
    right: tuple[list[np.ndarray], list[float], list[float]],
) -> tuple[list[np.ndarray], list[float], list[float]]:
    for index in range(3):
        left[index].extend(right[index])
    return left


@dataclass
class EALSRDDModel:
    sc: SparkContext
    config: EALSConfig
    RDD_users: UserFactors
    RDD_items: ItemFactors
    sq: np.ndarray
    sp: np.ndarray
    training_history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, candidates: RDD[tuple[int, int]]) -> RDD[tuple[int, int, float]]:
        by_item = candidates.join(self.RDD_users).map(_pair_by_item)
        return by_item.join(self.RDD_items).map(_dot_score)

    def save(self, path: str) -> None:
        self.RDD_users.saveAsPickleFile(f"{path}/user_factors")
        self.RDD_items.saveAsPickleFile(f"{path}/item_factors")
        payload = {
            "config": self.config.to_dict(),
            "sq": self.sq.tolist(),
            "sp": self.sp.tolist(),
            "training_history": self.training_history,
            "metadata": self.metadata,
        }
        self.sc.parallelize([json.dumps(payload)], 1).saveAsTextFile(f"{path}/metadata")

    @classmethod
    def load(cls, sc: SparkContext, path: str) -> "EALSRDDModel":
        payload = json.loads(sc.textFile(f"{path}/metadata").first())
        config = EALSConfig.from_dict(payload["config"])
        partitions = config.partitions or sc.defaultParallelism
        return cls(
            sc=sc,
            config=config,
            RDD_users=sc.pickleFile(f"{path}/user_factors").partitionBy(partitions),
            RDD_items=sc.pickleFile(f"{path}/item_factors").partitionBy(partitions),
            sq=np.asarray(payload["sq"], dtype=np.float64),
            sp=np.asarray(payload["sp"], dtype=np.float64),
            training_history=payload["training_history"],
            metadata=payload["metadata"],
        )


class RDDDistributedEALS:
    def __init__(self, config: EALSConfig):
        self.config = config
        self.partitions = 0

    def _partition(self, rdd: RDD[tuple[int, Any]]) -> RDD[tuple[int, Any]]:
        return rdd.partitionBy(self.partitions)

    def _initialize(self, RDD_interactions: RDD[Interaction]) -> tuple[UserFactors, ItemFactors]:
        total = RDD_interactions.count()
        if total == 0:
            raise ValueError("cannot train on an empty RDD")
        rank, seed, scale = self.config.rank, self.config.seed, self.config.factor_scale
        alpha, c0 = self.config.alpha, self.config.c0

        depart_svd = self.config.initialisation == "svd"
        if depart_svd:
            P_svd, Q_svd = svd_randomisee(
                RDD_interactions, rank, self.partitions, seed
            )
            vecteurs_u = RDD_interactions.context.broadcast(P_svd)
            vecteurs_i = RDD_interactions.context.broadcast(Q_svd)

        # f_i, le comptage par cle. reduceByKey somme dans chaque partition
        # avant le shuffle, contrairement a groupByKey.
        comptes = RDD_interactions.map(_item_and_one).reduceByKey(
            lambda a, b: a + b, self.partitions
        ).persist(StorageLevel.MEMORY_AND_DISK)

        # (f_i / |R|, (f_i / |R|) ** alpha), les deux quantites de (2).
        # Denominateur de (2), une somme sur tous les items, donc un scalaire.
        denominateur = comptes.values().map(
            lambda compte: (compte / total) ** alpha
        ).sum()
        if denominateur <= 0 or not math.isfinite(denominateur):
            raise FloatingPointError("invalid popularity denominator")

        def build_item(ligne: Any) -> tuple[int, ItemValue]:
            i, compte = ligne
            frequence = compte / total
            numerateur = frequence**alpha                  # numerateur de (2)
            q_i = (vecteurs_i.value[i] if depart_svd
                   else deterministic_factor(i, rank, seed + 1, scale))
            c_i = c0 * numerateur / denominateur           # equation (2)
            return (i, (q_i, c_i, frequence))

        def build_user(u: int) -> tuple[int, np.ndarray]:
            return (u, vecteurs_u.value[u] if depart_svd
                    else deterministic_factor(u, rank, seed, scale))

        RDD_items = comptes.map(build_item)
        RDD_users = RDD_interactions.map(_user_of).distinct(self.partitions).map(
            build_user
        )
        RDD_items = self._partition(RDD_items).persist(StorageLevel.MEMORY_AND_DISK)
        RDD_items.count()
        comptes.unpersist()
        return self._partition(RDD_users), RDD_items

    def _update_users(
        self,
        RDD_interactions: RDD[Interaction],
        RDD_users: UserFactors,
        RDD_items: ItemFactors,
        sq: np.ndarray,
    ) -> UserFactors:
        # Cle = item, pour joindre les q_i, puis cle = utilisateur, pour
        # regrouper les termes de la somme sur R_u.
        observations = (
            RDD_interactions.map(_interaction_by_item)
            .join(RDD_items)
            .map(_observation_by_user)
        )
        grouped = observations.combineByKey(
            _create_user_observation,
            _append_user_observation,
            _merge_user_observations,
            self.partitions,
        )
        config = self.config
        # S^q est diffuse une fois par executeur, et non serialise dans chaque
        # tache. Il est re-diffuse a chaque iteration puisque S^q change.
        cache = RDD_interactions.context.broadcast(np.asarray(sq, dtype=np.float64))

        def update(valeur: Any) -> np.ndarray:
            (q_obs, r_ui, w_ui, c_i), p_u = valeur
            return update_user_factor(
                p_u,
                np.stack(q_obs),
                r_ui,
                w_ui,
                c_i,
                cache.value,
                config.regularization,
                sweeps=config.inner_sweeps,
                max_norm=config.max_factor_norm,
            )

        return self._partition(grouped.join(RDD_users).mapValues(update))

    def _update_items(
        self,
        RDD_interactions: RDD[Interaction],
        RDD_users: UserFactors,
        RDD_items: ItemFactors,
        sp: np.ndarray,
    ) -> ItemFactors:
        # Symetrique de _update_users. Cle = utilisateur pour joindre les p_u,
        # puis cle = item pour regrouper les termes de la somme sur R_i.
        observations = (
            RDD_interactions.map(_interaction_by_user)
            .join(RDD_users)
            .map(_observation_by_item)
        )
        grouped = observations.combineByKey(
            _create_item_observation,
            _append_item_observation,
            _merge_item_observations,
            self.partitions,
        )
        config = self.config
        # Meme diffusion pour S^p.
        cache = RDD_interactions.context.broadcast(np.asarray(sp, dtype=np.float64))

        def update(valeur: Any) -> ItemValue:
            (p_obs, r_ui, w_ui), (q_i, c_i, f_i) = valeur
            return (
                update_item_factor(
                    q_i,
                    np.stack(p_obs),
                    r_ui,
                    w_ui,
                    c_i,
                    cache.value,
                    config.regularization,
                    sweeps=config.inner_sweeps,
                    max_norm=config.max_factor_norm,
                ),
                c_i,
                f_i,
            )

        return self._partition(grouped.join(RDD_items).mapValues(update))

    def objective(
        self,
        RDD_interactions: RDD[Interaction],
        RDD_users: UserFactors,
        RDD_items: ItemFactors,
        sq: np.ndarray,
    ) -> float:
        scored = (
            RDD_interactions.map(_interaction_by_item)
            .join(RDD_items)
            .map(_score_by_user)
            .join(RDD_users)
        )
        observed_correction = scored.map(_correction).sum()
        cache = RDD_interactions.context.broadcast(np.asarray(sq, dtype=np.float64))
        missing_all = distributed_sum(
            RDD_users, lambda p: float(p @ cache.value @ p)
        )
        regularizer = self.config.regularization * (
            distributed_sum(RDD_users, lambda p: float(p @ p))
            + distributed_sum(RDD_items, lambda value: float(value[0] @ value[0]))
        )
        return float(observed_correction + missing_all + regularizer)

    def fit(
        self,
        RDD_interactions: RDD[Interaction],
        progression: Callable[[dict[str, float]], None] | None = None,
    ) -> EALSRDDModel:
        """Apprend le modele.

        progression, si fournie, est appelee apres chaque iteration avec la
        ligne d'historique correspondante. Elle sert a suivre un long run.
        """
        self.partitions = self.config.partitions or RDD_interactions.context.defaultParallelism
        RDD_interactions = persist_once(RDD_interactions)
        interaction_count = RDD_interactions.count()
        debut_initialisation = time.perf_counter()
        RDD_users, RDD_items = self._initialize(RDD_interactions)
        persist_once(RDD_users)
        persist_once(RDD_items)
        user_count, item_count = RDD_users.count(), RDD_items.count()
        init_seconds = time.perf_counter() - debut_initialisation
        sq = compute_cache(RDD_items, self.config.rank, lambda value: (value[0], value[1]))
        sp = compute_cache(RDD_users, self.config.rank, lambda value: (value, 1.0))
        previous = self.objective(RDD_interactions, RDD_users, RDD_items, sq)
        history: list[dict[str, float]] = []
        stable_iterations = 0
        converged = False

        for iteration in range(1, self.config.max_iterations + 1):
            started = time.perf_counter()
            old_users = RDD_users
            RDD_users = self._update_users(RDD_interactions, old_users, RDD_items, sq).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            if self.config.checkpoint_interval and iteration % self.config.checkpoint_interval == 0:
                RDD_users.localCheckpoint()
            RDD_users.count()
            old_users.unpersist()
            sp = compute_cache(RDD_users, self.config.rank, lambda value: (value, 1.0))

            old_items = RDD_items
            RDD_items = self._update_items(RDD_interactions, RDD_users, old_items, sp).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            if self.config.checkpoint_interval and iteration % self.config.checkpoint_interval == 0:
                RDD_items.localCheckpoint()
            RDD_items.count()
            old_items.unpersist()
            sq = compute_cache(
                RDD_items, self.config.rank, lambda value: (value[0], value[1])
            )

            current = self.objective(RDD_interactions, RDD_users, RDD_items, sq)
            relative = (previous - current) / max(abs(previous), 1e-12)
            ligne = {
                "iteration": float(iteration),
                "objective": float(current),
                "relative_improvement": float(relative),
                "seconds": float(time.perf_counter() - started),
            }
            history.append(ligne)
            if progression is not None:
                progression(ligne)
            if 0 <= relative < self.config.convergence_tolerance:
                stable_iterations += 1
            else:
                stable_iterations = 0
            if stable_iterations >= self.config.convergence_patience:
                converged = True
                break
            previous = current

        RDD_interactions.unpersist()
        return EALSRDDModel(
            sc=RDD_interactions.context,
            config=self.config,
            RDD_users=RDD_users,
            RDD_items=RDD_items,
            sq=sq,
            sp=sp,
            training_history=history,
            metadata={
                "users": user_count,
                "items": item_count,
                "interactions": interaction_count,
                "init_seconds": init_seconds,
                "implementation": "rdd-only-fast-eals",
                "converged": converged,
                "stop_iteration": len(history),
                "convergence_tolerance": self.config.convergence_tolerance,
                "convergence_patience": self.config.convergence_patience,
            },
        )
