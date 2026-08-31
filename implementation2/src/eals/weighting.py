from __future__ import annotations

import numpy as np


def item_confidence(item_counts: np.ndarray, c0: float = 512.0, alpha: float = 0.4) -> np.ndarray:
    freq = item_counts.astype(np.float64) / max(item_counts.sum(), 1.0)
    powered = freq**alpha
    total = powered.sum()
    if total == 0.0:  # every item unseen (degenerate slice)
        return np.zeros_like(powered)
    return c0 * powered / total


def uniform_confidence(n_items: int, c0: float = 512.0) -> np.ndarray:
    return np.full(n_items, c0 / n_items, dtype=np.float64)
