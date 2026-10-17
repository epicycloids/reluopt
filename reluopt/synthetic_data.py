"""Synthetic two-layer ReLU regression data."""

from __future__ import annotations

import numpy as np


def pattern_of(X, u):
    return (X @ u >= 0).astype(np.int8)


def relu(z):
    return np.maximum(z, 0.0)


def make_planted(
    n: int,
    d: int,
    k: int,
    rng: np.random.Generator,
    noise: float = 0.0,
    normalize_rows: bool = False,
):
    """Planted teacher model.

    y = sum_{j=1}^k (X u_j^*)_+ * alpha_j^*  (+ noise), alpha_j^* > 0.

    Returns dict with X, y, teacher directions U_star (d x k),
    teacher coefficients alpha_star, teacher chambers S_star (list of {0,1}^n),
    and the clean signal y_clean.
    """
    X = rng.standard_normal((n, d))
    if normalize_rows:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    U = rng.standard_normal((d, k))
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    alpha = rng.uniform(0.7, 1.3, size=k)  # positive coefficients
    feats = relu(X @ U)  # n x k
    y_clean = feats @ alpha  # n
    y = y_clean.copy()
    if noise > 0:
        y = y + noise * rng.standard_normal(n)
    S_star = [pattern_of(X, U[:, j]) for j in range(k)]
    return dict(
        X=X,
        y=y,
        y_clean=y_clean,
        U_star=U,
        alpha_star=alpha,
        S_star=S_star,
        n=n,
        d=d,
        k=k,
    )
