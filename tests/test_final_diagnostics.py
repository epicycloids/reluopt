"""Optional search diagnostics must not disable original-data verification."""

import numpy as np

import reluopt.solver as solver
from reluopt.model import ReLUModel


DIAGNOSTIC_FIELDS = (
    "rho_lb", "ub_plus", "ub_minus", "gap_cert", "gap_heur", "theta"
)


def _assert_original_data_fit(solution, X, y, lam):
    A = np.maximum(np.asarray(X, np.float64) @ solution["U"], 0)
    residual = np.asarray(y, np.float64) - A @ solution["beta"]
    expected_obj = 0.5 * residual @ residual + lam * np.abs(solution["beta"]).sum()
    np.testing.assert_allclose(solution["obj"], expected_obj, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(solution["nu"], residual, rtol=1e-13, atol=1e-14)
    corr = A.T @ residual
    violation = np.maximum(np.abs(corr) - lam, 0)
    active = solution["beta"] != 0
    violation[active] = np.abs(corr[active] - lam * np.sign(solution["beta"][active]))
    np.testing.assert_allclose(
        solution["finite_master_kkt"], violation.max(initial=0), atol=1e-12
    )
    assert solution["finite_master_converged"]
    assert solution["finite_master_kkt"] <= 1e-9 * max(1, lam)


def _forbidden_diagnostic(*args, **kwargs):
    raise AssertionError("optional final pricing/bounds must not run")


def test_verification_without_diagnostics_still_rebuilds_and_refits(monkeypatch):
    X = np.tile([1e8 + 16, 1e8], (4, 1))
    y = np.ones(4)
    U = (np.array([1.0, -1.0]) / np.sqrt(2))[:, None]
    engine = solver.make_engine(X, y, dtype="float32")
    monkeypatch.setattr(solver, "search_signed_atoms", _forbidden_diagnostic)
    monkeypatch.setattr(solver, "estimate_pricing_bounds", _forbidden_diagnostic)
    monkeypatch.setattr(solver, "dual_gap", _forbidden_diagnostic)
    solution = solver.verify_solution(
        engine, 0.1, U, beta_start=np.zeros(1), run_diagnostics=False
    )
    _assert_original_data_fit(solution, X, y, 0.1)
    assert not solution["diagnostics_computed"]
    assert all(solution[field] is None for field in DIAGNOSTIC_FIELDS)
    model = ReLUModel(solution["U"], solution["beta"])
    np.testing.assert_allclose(model.predict(X), y - solution["nu"])


def test_default_verification_still_computes_diagnostics(monkeypatch):
    calls = []

    def price(*args, **kwargs):
        calls.append("price")
        return {"best": 0.25}

    def bounds(*args, **kwargs):
        calls.append("bounds")
        return {"ub_plus": 0.35, "ub_minus": 0.4}

    monkeypatch.setattr(solver, "search_signed_atoms", price)
    monkeypatch.setattr(solver, "estimate_pricing_bounds", bounds)
    X, y = np.eye(2), np.array([1.0, -1.0])
    solution = solver.verify_solution(solver.make_engine(X, y), 0.1, np.eye(2))
    _assert_original_data_fit(solution, X, y, 0.1)
    assert calls == ["price", "bounds"]
    assert solution["diagnostics_computed"]
    assert solution["rho_lb"] == 0.25
    assert all(solution[field] is not None for field in DIAGNOSTIC_FIELDS)


def test_solve_disables_only_final_diagnostics(monkeypatch):
    rng = np.random.default_rng(71)
    X = rng.standard_normal((16, 4))
    y = 2 * np.maximum(X[:, 0], 0) - np.maximum(X[:, 1], 0)
    original_verify = solver.verify_solution
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs["run_diagnostics"])
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(solver, "verify_solution", verify)
    monkeypatch.setattr(solver, "estimate_pricing_bounds", _forbidden_diagnostic)
    solution = solver.solve_rcg(
        X, y, 0.1, max_iter=2, certify_every=0, gated_every=0,
        final_diagnostics=False,
        oracle_kwargs=dict(
            n_random=4, n_subspace=2, n_perturb=0, sketch_m=0,
            iters_sketch=4, iters_full=1, top_polish=3,
            spectral_method="randomized",
        ),
    )
    assert calls == [False]
    assert solution["n_cols"] > 0
    assert solution["verified_float64"]
    assert not solution["diagnostics_computed"]
    assert solution["certificate_status"] == "not_computed"
    assert all(solution[field] is None for field in DIAGNOSTIC_FIELDS)
    _assert_original_data_fit(solution, X, y, 0.1)
    np.testing.assert_allclose(solution["model"].predict(X), y - solution["nu"])


def test_empty_final_model_has_truthful_verification_without_search(monkeypatch):
    X, y = np.eye(2), np.array([1.0, 2.0])
    monkeypatch.setattr(solver, "estimate_pricing_bounds", _forbidden_diagnostic)
    solution = solver.solve_rcg(
        X, y, 0.1, max_iter=1, certify_every=0, gated_every=0,
        max_feature_bytes=0,
        oracle_kwargs=dict(
            n_random=2, n_subspace=1, n_perturb=0, sketch_m=0,
            iters_sketch=2, iters_full=1, top_polish=1,
        ),
    )
    assert solution["n_cols"] == 0
    assert solution["verified_float64"]
    assert not solution["diagnostics_computed"]
    assert all(solution[field] is None for field in DIAGNOSTIC_FIELDS)
    _assert_original_data_fit(solution, X, y, 0.1)
