"""Projection from a signed frozen-mask block back to ReLU atoms."""
import numpy as np
from reluopt.consolidation import propose_consolidation
from reluopt.data_engine import ShardedData


def test_negative_output_coefficient_preserves_relu_orientation():
    X = np.array([[1.0], [-1.0]])
    engine = ShardedData(X, np.array([-1.0, 0.0]))
    proposed = propose_consolidation(engine, np.ones((1, 1)), np.array([-1.0]),
                                      lam=0.0, sketch_m=0, iters=3)
    np.testing.assert_array_equal(proposed, np.ones((1, 1)))
