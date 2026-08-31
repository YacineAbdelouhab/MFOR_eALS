from __future__ import annotations

import numpy as np


def update_user(
    p_u: np.ndarray,
    rhat_u: np.ndarray,
    q_rows: np.ndarray,
    w_u: np.ndarray,
    r_u: np.ndarray,
    c_u: np.ndarray,
    Sq: np.ndarray,
    lam: float,
) -> None:
    wr = w_u * r_u
    wmc = w_u - c_u 
    for f in range(p_u.shape[0]):
        qf = q_rows[:, f]
        rhat_u -= p_u[f] * qf 
        numer = np.dot(wr - wmc * rhat_u, qf) - (p_u @ Sq[:, f] - p_u[f] * Sq[f, f])
        denom = np.dot(wmc, qf * qf) + Sq[f, f] + lam
        p_u[f] = numer / denom
        rhat_u += p_u[f] * qf


def update_item(
    q_i: np.ndarray,
    rhat_i: np.ndarray,
    p_rows: np.ndarray,
    w_i: np.ndarray,
    r_i: np.ndarray,
    c_i: float,
    Sp: np.ndarray,
    lam: float,
) -> None:
    wr = w_i * r_i
    wmc = w_i - c_i
    for f in range(q_i.shape[0]):
        pf = p_rows[:, f]
        rhat_i -= q_i[f] * pf
        numer = np.dot(wr - wmc * rhat_i, pf) - c_i * (q_i @ Sp[:, f] - q_i[f] * Sp[f, f])
        denom = np.dot(wmc, pf * pf) + c_i * Sp[f, f] + lam
        q_i[f] = numer / denom
        rhat_i += q_i[f] * pf
