"""Correctness of the sequential oracle: coordinate descent must never increase the objective."""

from __future__ import annotations

import numpy as np
import pytest

from eals import interactions, numpy_reference as ref
from eals.weighting import item_confidence, uniform_confidence


def toy_data(n_users=20, n_items=15, density=0.25, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), rng.integers(0, n_items, n_users)] = True  # no empty user
    mask[rng.integers(0, n_users, n_items), np.arange(n_items)] = True  # no empty item
    users, items = np.nonzero(mask)
    return interactions.build(users, items, n_users, n_items)


def test_dual_view_is_consistent():
    data = toy_data()
    assert data.u_indptr.shape == (data.n_users + 1,)
    assert data.i_indptr.shape == (data.n_items + 1,)
    assert data.u_indptr[-1] == data.i_indptr[-1] == data.nnz

    # Both views must describe the same set of (u, i) pairs.
    from_user = {
        (u, int(i))
        for u in range(data.n_users)
        for i in data.u_items[data.user_slice(u)]
    }
    from_item = {
        (int(u), i)
        for i in range(data.n_items)
        for u in data.i_users[data.item_slice(i)]
    }
    assert from_user == from_item

    # i_nzid must point at the same nonzero in the user view.
    for i in range(data.n_items):
        sl = data.item_slice(i)
        assert np.all(data.u_items[data.i_nzid[sl]] == i)


def test_prediction_cache_matches_dense():
    data = toy_data()
    P, Q = ref.init_factors(data.n_users, data.n_items, K=4, seed=1)
    rhat = ref.predict_cache(data, P, Q)
    users = np.repeat(np.arange(data.n_users), np.diff(data.u_indptr))
    assert np.allclose(rhat, np.einsum("ij,ij->i", P[users], Q[data.u_items]))


@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_objective_is_non_increasing(alpha):
    """The defining property of coordinate descent: each exact update can only lower L."""
    data = toy_data()
    c = item_confidence(data.item_counts(), c0=32.0, alpha=alpha)
    model = ref.fit(data, K=8, c=c, lam=0.01, n_iters=12, seed=3)
    obj = np.array([h["objective"] for h in model.history])
    assert np.all(np.diff(obj) <= 1e-8), f"objective increased: {obj}"
    assert obj[-1] < obj[0]


def test_objective_matches_brute_force():
    """Eq. (14)'s O(|R| + MK²) rewrite must equal the literal O(MNK) definition, Eq. (7)."""
    data = toy_data(n_users=12, n_items=9, seed=5)
    c = item_confidence(data.item_counts(), c0=16.0, alpha=0.5)
    P, Q = ref.init_factors(data.n_users, data.n_items, K=5, seed=7)
    rhat = ref.predict_cache(data, P, Q)
    fast = ref.objective(data, P, Q, c, lam=0.02, rhat=rhat)

    dense_r = np.zeros((data.n_users, data.n_items))
    dense_w = np.zeros((data.n_users, data.n_items))
    users = np.repeat(np.arange(data.n_users), np.diff(data.u_indptr))
    dense_r[users, data.u_items] = data.r
    dense_w[users, data.u_items] = data.w
    observed_mask = np.zeros((data.n_users, data.n_items), dtype=bool)
    observed_mask[users, data.u_items] = True

    pred = P @ Q.T
    brute = (dense_w * (dense_r - pred) ** 2)[observed_mask].sum()
    brute += (c[None, :] * pred**2)[~observed_mask].sum()
    brute += 0.02 * (np.sum(P**2) + np.sum(Q**2))
    assert fast == pytest.approx(brute, rel=1e-9)


def test_alpha_zero_reduces_to_uniform_weighting():
    """Eq. (8) with alpha = 0 is exactly Hu et al.'s w0 = c0 / N."""
    data = toy_data()
    c0 = 64.0
    assert np.allclose(
        item_confidence(data.item_counts(), c0=c0, alpha=0.0),
        uniform_confidence(data.n_items, c0=c0),
    )


def test_popularity_weighting_orders_by_frequency():
    counts = np.array([1.0, 10.0, 100.0])
    c = item_confidence(counts, c0=100.0, alpha=0.5)
    assert c[0] < c[1] < c[2]
    assert c.sum() == pytest.approx(100.0)
