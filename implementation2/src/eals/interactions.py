from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interactions:
    n_users: int
    n_items: int
    u_indptr: np.ndarray 
    u_items: np.ndarray  
    i_indptr: np.ndarray  # (N+1,)
    i_users: np.ndarray  # (nnz,) user index of each nonzero, grouped by item
    i_nzid: np.ndarray  # (nnz,) position of that nonzero in the user-view arrays
    w: np.ndarray  # (nnz,) w_ui, user-view order
    r: np.ndarray  # (nnz,) r_ui, user-view order

    @property
    def nnz(self) -> int:
        return self.u_items.shape[0]

    def user_slice(self, u: int) -> slice:
        return slice(self.u_indptr[u], self.u_indptr[u + 1])

    def item_slice(self, i: int) -> slice:
        return slice(self.i_indptr[i], self.i_indptr[i + 1])

    def item_counts(self) -> np.ndarray:
        return np.diff(self.i_indptr).astype(np.float64)


def build(
    users: np.ndarray,
    items: np.ndarray,
    n_users: int,
    n_items: int,
    w: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> Interactions:
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    nnz = users.shape[0]

    order = np.lexsort((items, users))
    u_sorted, i_sorted = users[order], items[order]
    w_flat = (np.ones(nnz) if w is None else np.asarray(w, dtype=np.float64)[order]).astype(np.float64)
    r_flat = (np.ones(nnz) if r is None else np.asarray(r, dtype=np.float64)[order]).astype(np.float64)

    # Same nonzeros re-grouped by item; nzid records where each landed in the user view.
    item_order = np.lexsort((u_sorted, i_sorted))

    def indptr(keys: np.ndarray, n: int) -> np.ndarray:
        return np.concatenate(([0], np.bincount(keys, minlength=n).cumsum()))

    return Interactions(
        n_users=n_users,
        n_items=n_items,
        u_indptr=indptr(u_sorted, n_users),
        u_items=i_sorted,
        i_indptr=indptr(i_sorted, n_items),
        i_users=u_sorted[item_order],
        i_nzid=item_order,
        w=w_flat,
        r=r_flat,
    )
