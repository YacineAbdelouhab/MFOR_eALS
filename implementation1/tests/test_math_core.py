from __future__ import annotations

import numpy as np

from pyspark_eals_rdd.data import parse_paper_rating_event
from pyspark_eals_rdd.math_core import update_item_factor, update_user_factor, weighted_cache


def test_paper_rating_parser_uses_binary_implicit_feedback() -> None:
    assert parse_paper_rating_event("12\t98\t5\t1700000000") == (
        12,
        98,
        1.0,
        1_700_000_000.0,
        None,
    )


def test_user_update_matches_closed_form() -> None:
    rng = np.random.default_rng(7)
    all_items = rng.normal(scale=0.2, size=(7, 4))
    observed = np.array([0, 2, 5])
    q = all_items[observed]
    c_all = np.linspace(0.03, 0.09, len(all_items))
    c = c_all[observed]
    w = np.array([2.0, 3.0, 1.5])
    reg = 0.07
    sq = weighted_cache(all_items, c_all)
    expected = np.linalg.solve(
        sq + q.T @ (q * (w - c)[:, None]) + reg * np.eye(4), q.T @ w
    )
    actual = update_user_factor(np.zeros(4), q, np.ones(3), w, c, sq, reg, sweeps=100)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_item_update_matches_closed_form() -> None:
    rng = np.random.default_rng(11)
    all_users = rng.normal(scale=0.2, size=(8, 3))
    observed = np.array([1, 4, 6])
    p = all_users[observed]
    c, w, reg = 0.05, np.array([1.5, 2.5, 4.0]), 0.03
    sp = weighted_cache(all_users)
    expected = np.linalg.solve(
        c * sp + p.T @ (p * (w - c)[:, None]) + reg * np.eye(3), p.T @ w
    )
    actual = update_item_factor(np.zeros(3), p, np.ones(3), w, c, sp, reg, sweeps=100)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
