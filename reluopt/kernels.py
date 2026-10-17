"""Importance sampling for row sketches."""

from __future__ import annotations

import numpy as np


def as_float32(values):
    """Convert a search sketch, rejecting overflow instead of returning NaNs."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("search sketch exceeds float32 range; rescale X and y")
    return result


def importance_sample(nu, row_norms, m, rng, mix=0.3):
    """Row sketch: indices + inverse-probability weights for unbiased sums.
    p_i ∝ (1-mix) * |nu_i| ||x_i|| / Z + mix / n  (mixture keeps weights bounded).
    Returns (idx, w) with w_i = 1/(m p_i)."""
    n = len(nu)
    if m >= n:
        return np.arange(n), np.ones(n)
    score = np.abs(np.asarray(nu)) * np.asarray(row_norms)
    Z = score.sum()
    if not np.isfinite(Z) or Z <= 0:
        p = np.full(n, 1.0 / n)
    else:
        p = (1.0 - mix) * score / Z + mix / n
    idx = rng.choice(n, size=m, replace=True, p=p)
    w = 1.0 / (m * p[idx])
    return idx, w
