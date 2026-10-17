"""Objective and model contracts of optional external convex baselines."""

import importlib.util
import hashlib
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.convex_solvers import (
    ADELIE_DEFAULTS, SCNN_DEFAULTS, _AdelieBudgetStop, _config, _native_deadline,
    _patterns, _program_stats, _relu_parameters, _screen_coefficients,
    _signed_gap_scnn_optimizer, fit_adelie, fit_scnn,
)


def test_shared_patterns_use_half_width_and_remove_duplicates_and_zero():
    X = np.ones((12, 1))
    G, masks = _patterns(X, 20, 3)
    assert G.shape == (1, 1)
    np.testing.assert_array_equal(masks, np.ones((12, 1), dtype=bool))
    assert G[0, 0] > 0


def test_cone_blocks_map_to_actual_relu_and_identical_mean_objective():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    masks = np.array([[True, False], [False, True], [True, True]])
    blocks = np.array([[[2.0, -1.0], [-1.0, 2.0]], [[1.0, -0.5], [-0.5, 1.0]]])
    y = np.array([1.0, 1.0, 1.0])
    U, beta, stats = _program_stats(X, y, blocks, masks, 0.2)
    np.testing.assert_allclose(np.linalg.norm(U, axis=0), 1)
    prediction = np.maximum(X @ U, 0) @ beta
    expected = 0.5 * np.mean((prediction - y) ** 2) + 0.2 * np.abs(beta).sum()
    np.testing.assert_allclose(stats["objective_mean"], expected)
    np.testing.assert_allclose(stats["finite_program_objective_mean"], expected)
    assert stats["cone_violation_max"] == 0
    assert stats["prediction_mapping_max_error"] < 1e-14


def test_infeasible_blocks_are_reported_and_scored_as_actual_relu():
    X = np.array([[1.0], [-1.0]])
    masks = np.array([[True], [False]])
    blocks = np.array([[[-2.0]], [[0.0]]])
    U, beta, stats = _program_stats(X, np.zeros(2), blocks, masks, 0.1)
    np.testing.assert_array_equal(np.maximum(X @ U, 0) @ beta, [0.0, 2.0])
    assert stats["cone_violation_max"] == 2
    assert stats["prediction_mapping_max_error"] == 2


def test_zero_sum_vector_is_retained():
    U, beta = _relu_parameters(np.array([[[1.0, -1.0]], [[0.0, 0.0]]]))
    assert U.shape == (2, 1)
    np.testing.assert_allclose(beta, [np.sqrt(2)])


def test_partial_adelie_state_unpacks_screen_group_order():
    class State:
        screen_set = [2, 0]
        screen_begins = [0, 2]
        screen_beta = np.array([3.0, 4.0, 1.0, 2.0])

    np.testing.assert_array_equal(_screen_coefficients(State(), 2, 3), [1, 2, 0, 0, 3, 4])


def test_native_deadline_restores_signal_handler():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(_AdelieBudgetStop):
        with _native_deadline(time.perf_counter() + 0.001):
            time.sleep(0.02)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_unknown_adapter_options_rejected():
    for defaults in (SCNN_DEFAULTS, ADELIE_DEFAULTS):
        with pytest.raises(ValueError, match="unknown adapter options"):
            _config(defaults, {"intercept": True})


def test_adelie_solver_mode_is_explicit_and_validated():
    assert _config(ADELIE_DEFAULTS, {"solver_mode": "grpnet"})["solver_mode"] == "grpnet"
    assert _config(ADELIE_DEFAULTS, {"solver_mode": "pin"})["solver_mode"] == "pin"
    with pytest.raises(ValueError, match="solver_mode"):
        _config(ADELIE_DEFAULTS, {"solver_mode": "approximate"})


def test_optional_native_scnn_signed_duals_decrease_and_preserve_violation_norm():
    if importlib.util.find_spec("scnn") is None:
        pytest.skip("optional SCNN dependency not installed")
    import lab
    from scnn.private.interface import set_device
    from scnn.private.methods import AugmentedLagrangian
    from scnn.private.models import AL_MLP

    set_device("cpu", "float64", 0)
    X = lab.tensor([[1.0], [-1.0], [0.0]])
    model = AL_MLP(1, lab.tensor([[1.0], [0.0], [1.0]]), lab.tensor([[1.0]]), delta=3)
    model.weights = lab.ones_like(model.weights)
    criterion = SimpleNamespace(tol=1e-6)
    inner = SimpleNamespace(reset=lambda: None)
    outer = _signed_gap_scnn_optimizer(AugmentedLagrangian(use_delta_init=False))
    outer.step(model, criterion, inner, X, None)  # Native initialization.
    model.i_multipliers = lab.tensor([[[[0.5, 2.0, 0.25]]], [[[0.5, 2.0, 0.25]]]])
    previous_tol = criterion.tol
    outer.step(model, criterion, inner, X, None)
    expected = np.array([[[[0, 1, 0.25]]], [[[0, 1, 0.25]]]])
    np.testing.assert_allclose(lab.to_np(model.i_multipliers), expected)
    assert criterion.tol == previous_tol / outer.tau
    # Continued strict feasibility removes old multipliers; a boundary gap
    # leaves its multiplier unchanged, as required by complementarity.
    outer.step(model, criterion, inner, X, None)
    np.testing.assert_allclose(lab.to_np(model.i_multipliers)[..., :2], 0)
    np.testing.assert_allclose(lab.to_np(model.i_multipliers)[..., 2], 0.25)


@pytest.mark.parametrize("fit,module,mode", [
    (fit_scnn, "scnn", None), (fit_adelie, "adelie", "grpnet"),
    (fit_adelie, "adelie", "pin"),
])
def test_optional_native_constant_model_matches_soft_threshold(fit, module, mode):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"optional {module} dependency not installed")
    X, y = np.ones((12, 1)), np.full(12, 2.0)
    config = (dict(max_primal_iters=1000, max_dual_iters=200, tol=1e-10, constraint_tol=1e-12)
              if module == "scnn" else dict(solver_mode=mode))
    result = fit(X, y, 0.25, seed=3, width=20, time_budget=10, config=config)
    prediction = np.maximum(X @ result["U"], 0) @ result["beta"]
    np.testing.assert_allclose(prediction, 1.75, atol=2e-3)
    np.testing.assert_allclose(result["fit_info"]["objective_mean"], 0.46875, atol=1e-5)
    assert result["fit_info"]["cone_violation_max"] < 1e-5
    assert result["fit_info"]["model_kind"] == "relu_sampled_cones"


def test_optional_adelie_pin_and_grpnet_agree_without_centering_response():
    if importlib.util.find_spec("adelie") is None:
        pytest.skip("optional adelie dependency not installed")
    rng = np.random.default_rng(17)
    X = np.column_stack([rng.standard_normal((24, 2)), np.ones(24)])
    y = 1.2 + 0.8 * np.maximum(X[:, 0], 0) - 0.5 * np.maximum(-X[:, 1], 0)
    results = [fit_adelie(
        X, y, 0.05, seed=5, width=12, time_budget=10,
        config=dict(solver_mode=mode, tol=1e-10),
    ) for mode in ("grpnet", "pin")]
    for mode, result in zip(("grpnet", "pin"), results):
        info = result["fit_info"]
        assert info["native_solver_mode"] == mode
        assert info["status"] == "converged_native_criterion"
        assert info["cone_violation_max"] < 2e-5
        assert info["prediction_mapping_max_error"] < 2e-5
    assert results[0]["fit_info"]["pattern_sha256"] == results[1]["fit_info"]["pattern_sha256"]
    np.testing.assert_allclose(
        results[0]["fit_info"]["objective_mean"],
        results[1]["fit_info"]["objective_mean"], atol=1e-5, rtol=1e-5,
    )


def test_optional_native_solvers_agree_on_identical_sampled_cone_program():
    if any(importlib.util.find_spec(module) is None for module in ("scnn", "adelie")):
        pytest.skip("optional external convex solvers not installed")
    rng = np.random.default_rng(11)
    X = np.column_stack([rng.standard_normal((20, 2)), np.ones(20)])
    y = 0.8 * np.maximum(X[:, 0], 0) - 0.5 * np.maximum(-X[:, 1], 0)
    scnn = fit_scnn(X, y, 0.05, seed=5, width=12, time_budget=15,
                    config=dict(max_primal_iters=2000, max_dual_iters=200,
                                tol=1e-9, constraint_tol=1e-10))
    adelie = fit_adelie(X, y, 0.05, seed=5, width=12, time_budget=10,
                        config=dict(tol=1e-10))
    assert scnn["fit_info"]["pattern_sha256"] == adelie["fit_info"]["pattern_sha256"]
    assert scnn["fit_info"]["cone_violation_max"] < 2e-5
    np.testing.assert_allclose(
        scnn["fit_info"]["objective_mean"], adelie["fit_info"]["objective_mean"],
        atol=2e-4, rtol=2e-4,
    )


def _cuda_scnn_available():
    if importlib.util.find_spec("scnn") is None:
        pytest.skip("optional SCNN dependency not installed")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("raise_during_setup", [False, True])
def test_optional_scnn_restores_torch_state(device, grad_enabled, raise_during_setup, monkeypatch):
    if importlib.util.find_spec("scnn") is None:
        pytest.skip("optional SCNN dependency not installed")
    torch = pytest.importorskip("torch")
    if device == "cuda":
        _cuda_scnn_available()
    if raise_during_setup:
        import scnn.private.interface as interface
        native_setup = interface.set_device

        def failing_setup(*args, **kwargs):
            native_setup(*args, **kwargs)
            raise RuntimeError("injected setup failure")

        monkeypatch.setattr(interface, "set_device", failing_setup)
    with torch.set_grad_enabled(grad_enabled):
        previous_dtype = torch.get_default_dtype()
        dtype = "float64" if previous_dtype == torch.float32 else "float32"
        args = (np.ones((4, 1)), np.ones(4), 0.1)
        kwargs = dict(device=device, width=2, time_budget=0.1,
                      config=dict(dtype=dtype, max_primal_iters=2, max_dual_iters=2))
        if raise_during_setup:
            with pytest.raises(RuntimeError, match="injected setup failure"):
                fit_scnn(*args, **kwargs)
        else:
            fit_scnn(*args, **kwargs)
        assert torch.is_grad_enabled() == grad_enabled
        assert torch.get_default_dtype() == previous_dtype
        if grad_enabled:
            value = torch.tensor(2.0, requires_grad=True)
            value.square().backward()
            np.testing.assert_allclose(value.grad.item(), 4)


def test_optional_native_cuda_constant_model_matches_soft_threshold():
    _cuda_scnn_available()
    X, y = np.ones((12, 1)), np.full(12, 2.0)
    result = fit_scnn(
        X, y, 0.25, seed=3, device="cuda", width=20, time_budget=10,
        config=dict(dtype="float32", max_primal_iters=1000, max_dual_iters=200,
                    tol=1e-9, constraint_tol=1e-10),
    )
    np.testing.assert_allclose(np.maximum(X @ result["U"], 0) @ result["beta"], 1.75, atol=2e-3)
    np.testing.assert_allclose(result["fit_info"]["objective_mean"], 0.46875, atol=1e-5)
    assert result["fit_info"]["cone_violation_max"] < 1e-5


def test_optional_native_cuda_and_adelie_agree_on_sampled_cone_program():
    _cuda_scnn_available()
    if importlib.util.find_spec("adelie") is None:
        pytest.skip("optional adelie dependency not installed")
    rng = np.random.default_rng(11)
    X = np.column_stack([rng.standard_normal((20, 2)), np.ones(20)])
    y = 0.8 * np.maximum(X[:, 0], 0) - 0.5 * np.maximum(-X[:, 1], 0)
    scnn = fit_scnn(
        X, y, 0.05, seed=5, device="cuda", width=12, time_budget=15,
        config=dict(dtype="float32", max_primal_iters=2000, max_dual_iters=200,
                    tol=1e-8, constraint_tol=1e-8),
    )
    adelie = fit_adelie(X, y, 0.05, seed=5, width=12, time_budget=10,
                        config=dict(tol=1e-10))
    assert scnn["fit_info"]["pattern_sha256"] == adelie["fit_info"]["pattern_sha256"]
    np.testing.assert_allclose(
        scnn["fit_info"]["objective_mean"], adelie["fit_info"]["objective_mean"],
        atol=2e-4, rtol=2e-4,
    )


def test_optional_native_cuda_installs_host_patterns_near_float32_boundary():
    _cuda_scnn_available()
    seed, width = 19, 12
    gates = np.random.default_rng(seed).standard_normal((3, width // 2))
    middle = -(gates[0, 0] + gates[2, 0]) / gates[1, 0]
    perturbation = abs(float(np.spacing(np.float32(middle)))) / 16
    X = np.array([[1, middle + perturbation, 1], [1, middle - perturbation, 1],
                  [0, 0, 1], [1, 1, 1], [-1, -1, 1]], dtype=np.float64)
    assert np.any((X @ gates > 0) != (X.astype(np.float32) @ gates > 0))
    _, masks = _patterns(X, width, seed)
    expected = hashlib.sha256(np.packbits(masks, axis=0).tobytes()).hexdigest()
    result = fit_scnn(
        X, np.zeros(len(X)), 0.1, seed=seed, device="cuda", width=width,
        time_budget=5, config=dict(dtype="float32", max_primal_iters=2, max_dual_iters=2),
    )
    assert result["fit_info"]["pattern_sha256"] == expected
    assert result["fit_info"]["pattern_source"] == "original_input_float64"
