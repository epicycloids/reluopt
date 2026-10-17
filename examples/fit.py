"""Fit a signed two-layer ReLU regression model."""
import numpy as np
from reluopt import solve_rcg

rng = np.random.default_rng(7)
X = rng.standard_normal((512, 4))
y = np.maximum(X[:, 0], 0) - 0.5 * np.maximum(X[:, 1], 0)
result = solve_rcg(X, y, lam=0.5, max_iter=20)
prediction = result["model"].predict(X)
print(f"objective={result['obj']:.6g}, active={result['n_active']}, stop={result['stop_reason']}")
print(f"training RMSE={np.sqrt(np.mean((prediction-y)**2)):.6g}")
