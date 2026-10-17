"""Finite-dictionary Lasso regressions, independent of the pricing heuristic."""

from itertools import product

import numpy as np

from reluopt.restricted_master import AtomLasso
from reluopt.solver import make_engine


def _master(A, y, dtype="float64"):
    engine = make_engine(A, y, dtype=dtype)
    master = AtomLasso(engine)
    master.set_atoms(np.eye(A.shape[1]))
    return master


def _objective(A, y, beta, lam):
    return 0.5 * np.sum((A @ beta - y) ** 2) + lam * np.abs(beta).sum()


def _enumerated_optimum(A, y, lam):
    """Enumerate all orthants/supports for a small, independent reference."""
    best = np.inf
    for signs in product((-1, 0, 1), repeat=A.shape[1]):
        signs = np.asarray(signs)
        active = np.flatnonzero(signs)
        beta = np.zeros(A.shape[1])
        if len(active):
            As = A[:, active]
            beta[active] = np.linalg.lstsq(
                As.T @ As, As.T @ y - lam * signs[active], rcond=None
            )[0]
            if np.any(beta[active] * signs[active] <= 0):
                continue
        corr = A.T @ (y - A @ beta)
        if np.any(np.abs(corr[signs == 0]) > lam + 1e-9):
            continue
        best = min(best, _objective(A, y, beta, lam))
    assert np.isfinite(best)
    return best


def _assert_solution(master, A, y, lam, expected=None, atol=2e-8):
    beta, dual_shards, objective, active = master.solve(lam)
    np.testing.assert_allclose(objective, _objective(A, y, beta, lam), atol=1e-10)
    np.testing.assert_allclose(master.eng.gather_vec(dual_shards), y - A @ beta)
    assert active == np.flatnonzero(beta).tolist()
    corr = A.T @ (y - A @ beta)
    violation = np.maximum(np.abs(corr) - lam, 0)
    nz = beta != 0
    violation[nz] = np.abs(corr[nz] - lam * np.sign(beta[nz]))
    assert np.max(violation, initial=0) < atol
    assert master.converged
    assert master.stopping_reason == "kkt_tolerance"
    np.testing.assert_allclose(master.kkt_violation, np.max(violation, initial=0), atol=1e-12)
    if expected is not None:
        np.testing.assert_allclose(objective, expected, atol=atol, rtol=1e-10)
    return beta, objective


def test_correlated_relu_regression():
    # Earlier sign-infeasible deletions returned 10.1687717 for this fixture.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 5))
    U = rng.standard_normal((5, 30))
    U[:, 1:] = U[:, :1] + 0.05 * U[:, 1:]
    U /= np.linalg.norm(U, axis=0)
    y = rng.standard_normal(16)
    lam = 10.0 ** rng.uniform(-3, 0)
    A = np.maximum(X @ U, 0)
    master = AtomLasso(make_engine(X, y))
    master.set_atoms(U)
    beta, objective = _assert_solution(master, A, y, lam, expected=7.191939991928879)
    # A feasible Lasso dual supplies an independent lower bound on the optimum.
    residual = y - A @ beta
    dual = residual * min(1.0, lam / np.max(np.abs(A.T @ residual)))
    dual_objective = y @ dual - 0.5 * dual @ dual
    assert 0 <= objective - dual_objective < 1e-8


def test_small_dictionaries_against_orthant_enumeration():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        A = np.maximum(rng.standard_normal((12, 5)), 0)
        y = rng.standard_normal(12)
        lam = 0.1
        expected = _enumerated_optimum(A, y, lam)
        _assert_solution(_master(A, y), A, y, lam, expected=expected)


def test_duplicate_zero_and_rescaled_features():
    a = np.array([1.0, 2.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 2.0])
    A = np.column_stack([a, a, np.zeros(4), 3 * a, b, b])
    y = np.array([2.0, 4.0, -1.0, -2.0])
    lam = 0.3
    # Only the scaled feature is optimal for the first orthogonal row block.
    prediction = np.r_[(2.0 - lam / 15) * a[:2], (-1.0 + lam / 5) * b[2:]]
    expected = 0.5 * np.sum((prediction - y) ** 2) + lam * (
        (2.0 - lam / 15) / 3 + 1.0 - lam / 5
    )
    beta, _ = _assert_solution(_master(A, y), A, y, lam, expected=expected)
    assert beta[2] == 0
    np.testing.assert_allclose(A @ beta, prediction)


def test_nearly_dependent_features():
    rng = np.random.default_rng(53)
    A = np.abs(rng.standard_normal((12, 4)))
    A[:, 1] = A[:, 0] + 1e-4 * np.abs(rng.standard_normal(12))
    y = rng.standard_normal(12)
    lam = 1e-4
    _assert_solution(_master(A, y), A, y, lam, expected=_enumerated_optimum(A, y, lam))


def test_zero_penalty_matches_least_squares_with_dependent_columns():
    rng = np.random.default_rng(23)
    A = np.abs(rng.standard_normal((10, 5)))
    A[:, 4] = A[:, 0] + A[:, 1]
    y = rng.standard_normal(10)
    reference = np.linalg.lstsq(A, y, rcond=None)[0]
    beta, objective = _assert_solution(_master(A, y), A, y, 0.0)
    np.testing.assert_allclose(A @ beta, A @ reference, atol=1e-8)
    np.testing.assert_allclose(objective, _objective(A, y, reference, 0.0), atol=1e-10)


def test_append_keep_and_lambda_warm_starts():
    rng = np.random.default_rng(11)
    A = np.maximum(rng.standard_normal((18, 6)), 0)
    y = rng.standard_normal(18)
    engine = make_engine(A, y)
    master = AtomLasso(engine)
    master.set_atoms(np.eye(6)[:, :3])
    master.solve(0.1)
    master.append_atoms(np.eye(6)[:, 3:])
    for lam in (0.1, 0.5, 0.02):
        _, objective = _assert_solution(master, A, y, lam)
        _, fresh_objective = _assert_solution(_master(A, y), A, y, lam)
        np.testing.assert_allclose(objective, fresh_objective, atol=1e-9)
    keep = [0, 2, 3, 5]
    master.keep(keep)
    _assert_solution(master, A[:, keep], y, 0.02,
                     expected=_enumerated_optimum(A[:, keep], y, 0.02))


def test_iteration_limit_reports_unsolved_state_and_preserves_objective():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.ones(2)
    master = _master(A, y)
    beta, _, objective, _ = master.solve(0.1, max_outer=0)
    np.testing.assert_array_equal(beta, 0)
    assert objective == 1.0
    assert not master.converged
    assert master.stopping_reason == "iteration_limit"
    assert master.n_iter == 0
    assert master.kkt_violation == 0.9
    _assert_solution(master, A, y, 0.1, expected=0.19)


def test_empty_dictionary_and_invalid_controls():
    master = AtomLasso(make_engine(np.zeros((3, 1)), np.ones(3)))
    master.set_atoms(np.zeros((1, 0)))
    _assert_solution(master, np.zeros((3, 0)), np.ones(3), 0.0, expected=1.5)
    for options in ({"lam": -1}, {"lam": np.nan}, {"lam": 1, "tol": 0},
                    {"lam": 1, "batch_add": 0}, {"lam": 1, "max_outer": -1}):
        try:
            master.solve(**options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid controls accepted: {options}")
