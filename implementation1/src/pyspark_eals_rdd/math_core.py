from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _vector(values: object, rank: int) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rank,) or not np.all(np.isfinite(result)):
        raise ValueError(f"expected a finite vector of shape ({rank},)")
    return result.copy()


def _clip(vector: FloatArray, maximum: float) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm > maximum:
        vector *= maximum / norm
    return vector


def update_user_factor(
    p_u: object,
    q_obs: object,
    r_ui: object,
    w_ui: object,
    c_i: object,
    sq: object,
    regularization: float,
    *,
    sweeps: int = 1,
    max_norm: float = 100.0,
) -> FloatArray:
    """Applique l'equation (4) aux K coordonnees de p_u, une par une.

    q_obs contient les q_i des items observes par u, un par ligne.
    """
    q_obs = np.asarray(q_obs, dtype=np.float64)
    if q_obs.ndim != 2:
        raise ValueError("q_obs must be a matrix")
    rank = q_obs.shape[1]
    p_u = _vector(p_u, rank)
    r_ui = np.asarray(r_ui, dtype=np.float64).reshape(-1)
    w_ui = np.asarray(w_ui, dtype=np.float64).reshape(-1)
    c_i = np.asarray(c_i, dtype=np.float64).reshape(-1)
    cache = np.asarray(sq, dtype=np.float64)
    if q_obs.shape[0] == 0:
        return p_u
    if not (q_obs.shape[0] == len(r_ui) == len(w_ui) == len(c_i)):
        raise ValueError("inconsistent user observation arrays")
    if cache.shape != (rank, rank) or np.any(w_ui <= 0) or np.any(c_i <= 0):
        raise ValueError("invalid user cache or weights")

    prediction = q_obs @ p_u                       # les r_ui chapeau, i dans R_u
    for _ in range(sweeps):
        for f in range(rank):
            q_f = q_obs[:, f]                      # coordonnee f de chaque q_i
            sans = prediction - p_u[f] * q_f       # prediction privee de f

            # numerateur de (4)
            num = float(np.sum((w_ui * r_ui - (w_ui - c_i) * sans) * q_f))
            num -= float(p_u @ cache[:, f] - p_u[f] * cache[f, f])

            # denominateur de (4)
            den = float(
                np.sum((w_ui - c_i) * q_f * q_f) + cache[f, f] + regularization
            )
            if den <= 1e-12 or not np.isfinite(den):
                raise FloatingPointError("invalid user coordinate denominator")

            ancien = p_u[f]
            p_u[f] = num / den
            prediction += (p_u[f] - ancien) * q_f  # mise a jour incrementale
    return _clip(p_u, max_norm)


def update_item_factor(
    q_i: object,
    p_obs: object,
    r_ui: object,
    w_ui: object,
    c_i: float,
    sp: object,
    regularization: float,
    *,
    sweeps: int = 1,
    max_norm: float = 100.0,
) -> FloatArray:
    """Applique l'equation (5) aux K coordonnees de q_i, une par une.

    p_obs contient les p_u des utilisateurs ayant interagi avec i.
    Seule difference avec (4), c_i multiplie le terme de cache.
    """
    p_obs = np.asarray(p_obs, dtype=np.float64)
    if p_obs.ndim != 2:
        raise ValueError("p_obs must be a matrix")
    rank = p_obs.shape[1]
    q_i = _vector(q_i, rank)
    r_ui = np.asarray(r_ui, dtype=np.float64).reshape(-1)
    w_ui = np.asarray(w_ui, dtype=np.float64).reshape(-1)
    c_i = float(c_i)
    cache = np.asarray(sp, dtype=np.float64)
    if p_obs.shape[0] == 0:
        return q_i
    if not (p_obs.shape[0] == len(r_ui) == len(w_ui)):
        raise ValueError("inconsistent item observation arrays")
    if cache.shape != (rank, rank) or np.any(w_ui <= 0) or c_i <= 0:
        raise ValueError("invalid item cache or weights")

    prediction = p_obs @ q_i
    for _ in range(sweeps):
        for f in range(rank):
            p_f = p_obs[:, f]
            sans = prediction - q_i[f] * p_f

            num = float(np.sum((w_ui * r_ui - (w_ui - c_i) * sans) * p_f))
            num -= c_i * float(q_i @ cache[:, f] - q_i[f] * cache[f, f])

            den = float(
                np.sum((w_ui - c_i) * p_f * p_f) + c_i * cache[f, f] + regularization
            )
            if den <= 1e-12 or not np.isfinite(den):
                raise FloatingPointError("invalid item coordinate denominator")

            ancien = q_i[f]
            q_i[f] = num / den
            prediction += (q_i[f] - ancien) * p_f
    return _clip(q_i, max_norm)


def weighted_cache(factors: object, weights: object | None = None) -> FloatArray:
    matrix = np.asarray(factors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("factors must be a matrix")
    if weights is None:
        return matrix.T @ matrix
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vector) != matrix.shape[0]:
        raise ValueError("weights length mismatch")
    return matrix.T @ (matrix * vector[:, None])

