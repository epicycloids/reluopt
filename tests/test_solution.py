"""Regression tests for returned models and feasible working-set changes."""

from __future__ import annotations

import numpy as np
import pytest

from reluopt.model import ReLUModel
from reluopt.restricted_master import AtomLasso
from reluopt.solver import (
    _compress_chambers,
    make_engine,
    solve_rcg,
    verify_solution,
)


SMALL_ORACLE = dict(
    n_random=4, n_subspace=2, n_perturb=0, sketch_m=0,
    iters_sketch=4, iters_full=1, top_polish=3,
)


def test_verification_uses_original_data_and_preserves_input_directions():
    # Float32 cancellation changes the ReLU feature by about 40%. Float64
    # reductions of those quantized features cannot repair that error.
    X = np.tile([1e8 + 16, 1e8], (4, 1))
    y = np.ones(4)
    U = (np.array([1.0, -1.0]) / np.sqrt(2))[:, None]
    before = U.copy()
    solution = verify_solution(
        make_engine(X, y, dtype="float32"), 0.1, U, cert_steps=0
    )
    prediction = np.maximum(X @ solution["U"], 0) @ solution["beta"]
    replay = 0.5 * np.sum((prediction - y) ** 2) + 0.1 * np.abs(solution["beta"]).sum()
    np.testing.assert_allclose(solution["obj"], replay, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(solution["nu"], y - prediction, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(U, before)
    assert solution["finite_master_converged"]
    assert solution["finite_master_kkt"] < 1e-8


def test_column_cap_preserves_incumbent():
    rng = np.random.default_rng(314)
    X = rng.normal(size=(48, 3))
    y = rng.normal(size=48)
    solution = solve_rcg(
        X, y, 0.1, max_iter=8, max_cols=2, certify_every=0,
        gated_every=0, final_verify=False, eps_heur=0,
        oracle_kwargs=SMALL_ORACLE,
    )
    assert solution["stop_reason"] == "column_budget"
    assert solution["n_cols"] <= 2
    assert np.all(np.diff(solution["history"]["obj"]) <= 0)
    predicted = solution["model"].predict(X, chunk_rows=7)
    objective = 0.5 * np.sum((y - predicted) ** 2) + 0.1 * np.abs(solution["beta"]).sum()
    np.testing.assert_allclose(solution["obj"], objective, atol=1e-12)


def test_feature_payload_budget():
    X = np.array([[1.0], [-1.0]])
    solution = solve_rcg(
        X, np.ones(2), 0.1, max_iter=4, certify_every=0, gated_every=0,
        final_verify=False, max_feature_bytes=0, oracle_kwargs=SMALL_ORACLE,
    )
    assert solution["stop_reason"] == "feature_budget"
    assert solution["max_cols"] == solution["n_cols"] == 0
    assert solution["feature_payload_bytes"] == 0
    np.testing.assert_array_equal(solution["model"].predict(X), np.zeros(2))


def test_chamber_merge_preserves_signed_representation():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    U = np.array([[1.0, 0.8], [0.0, 0.6]])
    engine = make_engine(X, np.array([-1.0, -0.6, -1.6]))
    master = AtomLasso(engine)
    master.set_atoms(U)
    beta, _, objective, _ = master.solve(0.01)
    assert np.count_nonzero(beta) == 2
    merged = _compress_chambers(master, engine, 0.01, objective)
    assert merged is not None
    assert merged.P == 1
    assert merged.beta[0] < 0
    _, _, merged_objective, _ = merged.solve(0.01)
    assert merged_objective <= objective


def test_model_predict_and_immutable_parameters():
    U = np.eye(2)
    beta = np.array([2.0, -1.0])
    model = ReLUModel(U, beta)
    U[:] = 0
    beta[:] = 0
    X = np.array([[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]])
    np.testing.assert_array_equal(model.predict(X, chunk_rows=1), [0.0, -4.0, 10.0])
    assert not model.directions.flags.writeable
    assert not model.coefficients.flags.writeable
    assert model.predict(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError):
        model.predict(np.ones((3, 4)))
    with pytest.raises(ValueError):
        model.predict(X, chunk_rows=0)
    with pytest.raises(ValueError):
        model.predict(X.astype(complex) + 1j)


@pytest.mark.parametrize("option", [
    {"lam": -1.0}, {"lam": np.nan}, {"max_iter": -1},
    {"max_cols": 0}, {"max_cols": 2.5}, {"max_feature_bytes": -1},
    {"patience": 0}, {"sketch_m": "bad"}, {"sketch_m": -2},
    {"time_budget": 0}, {"price_tol": -1.0}, {"eps_heur": np.inf},
])
def test_invalid_solver_options(option):
    kwargs = {"lam": 0.1, **option}
    with pytest.raises(ValueError):
        solve_rcg(np.ones((3, 1)), np.ones(3), **kwargs)


def test_invalid_device_is_not_silently_changed_to_cpu():
    with pytest.raises(ValueError, match="unknown device"):
        make_engine(np.ones((3, 1)), np.ones(3), device="typo")
