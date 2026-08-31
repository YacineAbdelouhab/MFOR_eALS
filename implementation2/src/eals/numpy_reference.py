
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eals.interactions import Interactions
from eals.kernels import update_item, update_user


@dataclass
class EALSModel:
    P: np.ndarray  # (M, K)
    Q: np.ndarray  # (N, K)
    c: np.ndarray  # (N,) missing-data confidence per item, Eq. (8)
    lam: float
    history: list[dict] = field(default_factory=list)

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.P[users], self.Q[items])


def init_factors(n_users: int, n_items: int, K: int, seed: int = 0, scale: float = 0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, (n_users, K)), rng.normal(0.0, scale, (n_items, K))


def predict_cache(data: Interactions, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", P[np.repeat(np.arange(data.n_users), np.diff(data.u_indptr))], Q[data.u_items])


def objective(data: Interactions, P: np.ndarray, Q: np.ndarray, c: np.ndarray, lam: float, rhat: np.ndarray) -> float:

    observed = np.dot(data.w, (data.r - rhat) ** 2)
    Sq = compute_Sq(Q, c)
    # Σ_u Σ_{i∉R_u} c_i r̂² = Σ_u p_uᵀ S^q p_u − Σ_{(u,i)∈R} c_i r̂²  — Eq. (14)
    missing = np.sum((P @ Sq) * P) - np.dot(c[data.u_items], rhat**2)
    reg = lam * (np.sum(P**2) + np.sum(Q**2))
    return float(observed + missing + reg)


def compute_Sq(Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (Q * c[:, None]).T @ Q


def compute_Sp(P: np.ndarray) -> np.ndarray:
    return P.T @ P


def fit(
    data: Interactions,
    K: int,
    c: np.ndarray,
    lam: float = 0.01,
    n_iters: int = 10,
    seed: int = 0,
    track_objective: bool = True,
) -> EALSModel:
    P, Q = init_factors(data.n_users, data.n_items, K, seed)
    rhat = predict_cache(data, P, Q)
    model = EALSModel(P=P, Q=Q, c=c, lam=lam)

    for it in range(n_iters):
        Sq = compute_Sq(Q, c)
        for u in range(data.n_users):
            sl = data.user_slice(u)
            idx = data.u_items[sl]
            if idx.size == 0:
                continue
            update_user(P[u], rhat[sl], Q[idx], data.w[sl], data.r[sl], c[idx], Sq, lam)

        Sp = compute_Sp(P)
        for i in range(data.n_items):
            sl = data.item_slice(i)
            nz = data.i_nzid[sl]
            if nz.size == 0:
                continue
            # Item view is a gather into the shared r̂; scatter the updated values back.
            rhat_i = rhat[nz].copy()
            update_item(Q[i], rhat_i, P[data.i_users[sl]], data.w[nz], data.r[nz], c[i], Sp, lam)
            rhat[nz] = rhat_i

        if track_objective:
            model.history.append({"iter": it, "objective": objective(data, P, Q, c, lam, rhat)})

    return model
