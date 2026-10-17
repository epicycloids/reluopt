"""Warm-started fitting preserves the supplied network and its objective."""

import numpy as np
import pytest

import reluopt.solver as solver


SMALL_ORACLE = dict(
    n_random=4, n_subspace=2, n_perturb=0, sketch_m=0,
    iters_sketch=4, iters_full=1, top_polish=3,
)


def _fit(X, y, lam, U, beta=None, **options):
    return solver.solve_rcg(
        X, y, lam, initial_U=U, initial_beta=beta,
        max_iter=options.pop("max_iter", 0), final_diagnostics=False,
        certify_every=0, gated_every=0, oracle_kwargs=SMALL_ORACLE, **options,
    )


def _objective(X, y, lam, result):
    residual = y - np.maximum(X @ result["U"], 0) @ result["beta"]
    return 0.5 * residual @ residual + lam * np.abs(result["beta"]).sum()


def test_scaled_signed_network_is_normalized_and_refitted_without_search(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("coefficient-only fitting must not search")

    monkeypatch.setattr(solver, "search_signed_atoms", forbidden)
    monkeypatch.setattr(solver, "estimate_pricing_bounds", forbidden)
    X, y = np.array([[1.0], [-1.0]]), np.ones(2)
    U, beta = np.array([[2.0, -3.0, 0.0]]), np.array([0.25, -1.0, 99.0])
    before_U, before_beta = U.copy(), beta.copy()
    result = _fit(X, y, 0.25, U, beta)
    np.testing.assert_array_equal(U, before_U)
    np.testing.assert_array_equal(beta, before_beta)
    np.testing.assert_array_equal(result["U"], [[1.0, -1.0]])
    np.testing.assert_allclose(result["beta"], [0.75, 0.75], atol=1e-10)
    np.testing.assert_allclose(result["obj"], 0.4375, atol=1e-12)
    # Raw function is [0.5, -3], and its normalized path penalty is 3.5.
    np.testing.assert_allclose(result["initialization"]["obj"], 9.0)
    assert result["initialization"]["n_supplied"] == 3
    assert result["initialization"]["n_cols"] == 2
    assert result["initialization"]["refit_seconds"] > 0
    assert result["n_iter"] == 0
    assert result["finite_master_converged"]
    assert result["verified_float64"]
    assert not result["diagnostics_computed"]


def test_omitted_coefficients_seed_directions_and_keep_distinct_same_chamber_atoms():
    X = np.eye(2)
    U = np.array([[1.0, 0.2], [0.2, 1.0]])
    # Both directions have the same all-active chamber, but are different
    # features; their opposite coefficients must remain independently fitted.
    y = np.array([1.0, -1.0])
    result = _fit(X, y, 0.01, U)
    assert result["n_cols"] == 2
    assert result["n_active"] == 2
    assert result["beta"][0] > 0 > result["beta"][1]
    assert result["obj"] < 0.03
    np.testing.assert_allclose(result["initialization"]["obj"], 1.0)


def test_normalization_handles_large_and_small_finite_directions():
    X, y = np.array([[1.0], [-1.0]]), np.ones(2)
    U, beta = np.array([[1e308, -1e-308]]), np.array([1e-308, 1e308])
    result = _fit(X, y, 0, U, beta)
    np.testing.assert_allclose(result["model"].predict(X), y, atol=1e-14)
    np.testing.assert_allclose(result["initialization"]["obj"], 0, atol=1e-28)


@pytest.mark.parametrize("U,beta", [
    (np.empty((2, 0)), np.empty(0)),
    (np.zeros((2, 3)), np.array([3.0, 2.0, -5.0])),
])
def test_empty_initial_network_is_valid(U, beta):
    X, y = np.eye(2), np.array([1.0, -1.0])
    result = _fit(X, y, 0.1, U, beta)
    assert result["n_cols"] == result["n_active"] == result["n_iter"] == 0
    np.testing.assert_array_equal(result["model"].predict(X), np.zeros(2))
    np.testing.assert_allclose(result["obj"], 1.0)
    assert result["finite_master_converged"]


def test_cold_zero_iteration_returns_verified_zero_model():
    result = solver.solve_rcg(np.eye(2), np.ones(2), 0.1, max_iter=0)
    assert result["n_cols"] == result["n_iter"] == 0
    assert result["finite_master_converged"]
    assert result["initialization"] is None


def test_full_search_can_add_atoms_beyond_seed_chamber():
    X, y = np.array([[1.0], [-1.0]]), np.ones(2)
    seeded = _fit(X, y, 0.25, np.ones((1, 1)), np.array([0.5]))
    refined = _fit(
        X, y, 0.25, np.ones((1, 1)), np.array([0.5]), max_iter=4, max_cols=2,
    )
    assert refined["n_cols"] == 2
    np.testing.assert_allclose(refined["obj"], 0.4375, atol=1e-10)
    assert refined["obj"] < seeded["obj"]
    np.testing.assert_allclose(_objective(X, y, 0.25, refined), refined["obj"])


def test_full_seed_dictionary_and_expired_budget_preserve_refitted_incumbent():
    X, y = np.array([[1.0], [-1.0]]), np.ones(2)
    result = _fit(
        X, y, 0.25, np.array([[1.0, -1.0]]), np.array([3.0, -4.0]),
        max_cols=2, max_iter=4, time_budget=1e-12,
    )
    assert result["stop_reason"] == "time_budget"
    assert result["n_iter"] == 0
    assert result["n_cols"] == 2
    np.testing.assert_allclose(result["obj"], 0.4375, atol=1e-12)
    assert result["obj"] <= result["initialization"]["obj"]


def test_original_float64_seed_refit_survives_float32_feature_cancellation():
    X = np.tile([1e8 + 16, 1e8], (4, 1))
    y = np.ones(4)
    U = (np.array([1.0, -1.0]) / np.sqrt(2))[:, None]
    result = _fit(X, y, 0.1, U, np.ones(1), dtype_data="float32", final_verify=False)
    np.testing.assert_allclose(_objective(X, y, 0.1, result), result["obj"], atol=1e-12)
    assert result["finite_master_converged"]
    assert result["verified_float64"]
    assert result["obj"] <= result["initialization"]["obj"]


def test_coefficient_only_refit_does_not_build_unused_device_dictionary(monkeypatch):
    X, y = np.eye(2), np.ones(2)
    engine = solver.make_engine(X, y, dtype="float32")

    def forbidden(*args, **kwargs):
        raise AssertionError("coefficient-only fit needs no device dictionary")

    monkeypatch.setattr(engine, "build_atoms", forbidden)
    result = _fit(X, y, 0.1, np.eye(2), np.ones(2), engine=engine)
    np.testing.assert_allclose(result["beta"], 0.9, atol=1e-12)
    assert result["feature_payload_bytes"] == engine.feature_buffer_bytes(2)


def test_final_float64_comparison_restores_initial_fit_if_candidate_loses(monkeypatch):
    original = solver.verify_solution
    calls = 0

    def make_final_candidate_lose(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            result["beta"] = np.zeros_like(result["beta"])
        return result

    monkeypatch.setattr(solver, "verify_solution", make_final_candidate_lose)
    X, y = np.eye(2), np.ones(2)
    result = _fit(X, y, 0.1, np.eye(2), np.ones(2), max_iter=1)
    assert result["initialization"]["fallback_used"]
    np.testing.assert_allclose(result["beta"], 0.9, atol=1e-12)
    np.testing.assert_allclose(result["obj"], 0.19, atol=1e-12)
    np.testing.assert_allclose(_objective(X, y, 0.1, result), result["obj"])


@pytest.mark.parametrize("options", [
    {"initial_U": np.ones((3, 1))},
    {"initial_U": np.ones(2)},
    {"initial_U": np.array([[np.nan], [0.0]])},
    {"initial_U": np.eye(2), "initial_beta": np.ones(3)},
    {"initial_U": np.eye(2), "initial_beta": np.array([1.0, np.inf])},
    {"initial_U": np.eye(2).astype(complex)},
    {"initial_beta": np.ones(2)},
    {"initial_U": np.eye(2), "max_cols": 1},
    {"initial_U": np.eye(2), "max_feature_bytes": 0},
    {"initial_U": np.array([[1e308], [0.0]]), "initial_beta": np.array([10.0])},
])
def test_invalid_initialization_is_rejected_without_mutation(options):
    with pytest.raises(ValueError):
        solver.solve_rcg(np.eye(2), np.ones(2), 0.1, max_iter=0, **options)


def test_optional_cuda_warm_start_matches_float64_coefficient_refit():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    X, y = np.array([[1.0], [-1.0]]), np.ones(2)
    U, beta = np.array([[2.0, -3.0]]), np.array([0.25, -1.0])
    result = _fit(X, y, 0.25, U, beta, device="cuda", max_iter=2, max_cols=2)
    np.testing.assert_allclose(result["obj"], 0.4375, atol=1e-10)
    np.testing.assert_allclose(result["model"].predict(X), 0.75, atol=1e-8)
    assert result["verified_float64"]
