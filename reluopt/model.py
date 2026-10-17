"""A fitted scalar-output ReLU model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReLUModel:
    """Predict ``relu(X @ directions) @ coefficients`` in float64.

    The model has no implicit intercept. Append a constant input feature when
    fitting and predicting if the chosen objective needs one.
    """

    directions: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        if np.iscomplexobj(self.directions) or np.iscomplexobj(self.coefficients):
            raise ValueError("model parameters must be real")
        U = np.array(self.directions, dtype=np.float64, copy=True)
        beta = np.array(self.coefficients, dtype=np.float64, copy=True)
        if U.ndim != 2 or beta.shape != (U.shape[1],):
            raise ValueError("directions and coefficients have incompatible shapes")
        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(beta)):
            raise ValueError("model parameters must be finite")
        U.flags.writeable = False
        beta.flags.writeable = False
        object.__setattr__(self, "directions", U)
        object.__setattr__(self, "coefficients", beta)

    def predict(self, X, *, chunk_rows=65536):
        """Return predictions, using at most ``chunk_rows`` feature rows."""
        X = np.asarray(X)
        if np.iscomplexobj(X) or X.dtype.kind not in "biuf":
            raise ValueError("X must contain real numbers")
        if X.ndim != 2 or X.shape[1] != self.directions.shape[0]:
            raise ValueError("X has the wrong number of input features")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        if isinstance(chunk_rows, (bool, np.bool_)) or not isinstance(
            chunk_rows, (int, np.integer)
        ) or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        prediction = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, X.shape[0], chunk_rows):
            end = min(start + chunk_rows, X.shape[0])
            features = np.asarray(X[start:end], dtype=np.float64) @ self.directions
            np.maximum(features, 0, out=features)
            prediction[start:end] = features @ self.coefficients
        return prediction
