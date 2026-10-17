"""Small mathematical checks for the optional finite-width Torch baselines."""

import os

import numpy as np
import pytest

from benchmarks.torch_solvers import (
    _canonical_parameters,
    _penalized_loss,
    fit_torch,
)


def test_canonical_export_preserves_signed_relu_model_and_zero_atoms():
    rng = np.random.default_rng(31)
    X = rng.standard_normal((9, 4))
    W = rng.standard_normal((6, 4)).astype(np.float32)
    a = np.array([2.0, -3.0, 0.0, 1.0, -0.25, 0.5], dtype=np.float32)
    W[3] = 0
    U, beta = _canonical_parameters(W, a)
    expected = np.maximum(X @ W.astype(np.float64).T, 0) @ a.astype(np.float64)
    np.testing.assert_allclose(np.maximum(X @ U, 0) @ beta, expected, atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(U, axis=0), 1.0, atol=1e-15)
    assert U.dtype == beta.dtype == np.float64
    assert U.shape == (4, 4)
    assert np.count_nonzero(beta < 0) == 2


def test_balanced_parameter_penalty_matches_canonical_path_norm():
    rng = np.random.default_rng(32)
    W = rng.standard_normal((7, 3))
    a = rng.standard_normal(7)
    norm = np.linalg.norm(W, axis=1)
    scale = np.sqrt(np.abs(a) / norm)
    W *= scale[:, None]
    a /= scale
    _, beta = _canonical_parameters(W, a)
    np.testing.assert_allclose(
        0.5 * (np.square(W).sum() + np.square(a).sum()), np.abs(beta).sum(),
        atol=2e-15,
    )


def test_explicit_l2_gradient_has_correct_mean_loss_and_lambda_scaling():
    torch = pytest.importorskip("torch")
    X = torch.tensor([[1.0, 0.5], [0.2, 1.0], [-1.0, 0.1]], dtype=torch.float64)
    y = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    W = torch.tensor([[0.3, 0.8], [-0.6, 0.5]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
    lam = 0.13
    hidden = (X @ W.T).relu()
    prediction = hidden @ a
    loss = _penalized_loss(prediction, y, W, a, lam)
    loss.backward()
    error = (prediction.detach() - y) / len(y)
    expected_a = hidden.detach().T @ error + lam * a.detach()
    expected_W = ((error[:, None] * a.detach()) * (X @ W.detach().T > 0)).T @ X + lam * W.detach()
    torch.testing.assert_close(a.grad, expected_a)
    torch.testing.assert_close(W.grad, expected_W)
    expected_loss = 0.5 * (prediction.detach() - y).square().mean() + 0.5 * lam * (
        W.detach().square().sum() + a.detach().square().sum()
    )
    torch.testing.assert_close(loss.detach(), expected_loss)


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
def test_tiny_training_preserves_optimizer_contract_and_reproducibility(monkeypatch, optimizer):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(91)
    X = rng.standard_normal((8, 2))
    X = np.column_stack([X, np.ones(8)])
    y = np.maximum(X[:, 0], 0) - 0.3 * X[:, 1]
    calls = []
    cls_name = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}[optimizer]
    original = getattr(torch.optim, cls_name)

    def construct(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.optim, cls_name, construct)
    kwargs = dict(
        optimizer=optimizer, seed=11, width=5, time_budget=30,
        config=dict(max_epochs=1, batch_size=4, sync_every=1, fused=False),
    )
    first = fit_torch(X, y, 0.01, **kwargs)
    second = fit_torch(X, y, 0.01, **kwargs)
    np.testing.assert_array_equal(first["U"], second["U"])
    np.testing.assert_array_equal(first["beta"], second["beta"])
    info = first["fit_info"]
    assert info["fused"] is False
    assert calls[0]["fused"] is False
    assert info["steps"] == 2
    assert info["epochs"] == 1
    assert info["samples_seen"] == 8
    assert info["stopping_reason"] == "max_epochs"
    assert info["hidden_bias"] is info["output_bias"] is False
    assert info["finite_width"]
    assert info["fit_seconds"] >= first["trajectory"][-1]["seconds"]
    assert [item["step"] for item in first["trajectory"]] == [0, 2]
    selected = first["trajectory"][info["selected_checkpoint"]]
    selection_field = "train_loss" if optimizer == "adamw" else "train_objective"
    assert selected[selection_field] == min(item[selection_field] for item in first["trajectory"])
    predicted = np.maximum(X @ first["U"], 0) @ first["beta"]
    common = 0.5 * np.mean((predicted - y) ** 2) + 0.01 * np.abs(first["beta"]).sum()
    np.testing.assert_allclose(info["selected_train_objective"], common, rtol=2e-6, atol=1e-8)
    if optimizer == "adamw":
        assert calls[0]["weight_decay"] == 0.01
        assert info["regularization"] == "decoupled_weight_decay"
        assert info["training_l2_penalty"] is None
        assert info["selected_optimization_objective"] is None
    else:
        assert calls[0]["weight_decay"] == 0
        assert info["regularization"] == "explicit_two_layer_l2"
        assert info["training_l2_penalty"] > 0
        assert info["selected_optimization_objective"] >= info["selected_train_objective"] - 1e-6


def test_setup_and_export_overrun_are_reported():
    pytest.importorskip("torch")
    result = fit_torch(np.eye(2), np.ones(2), 0.1, width=3, time_budget=1e-12)
    info = result["fit_info"]
    assert info["fused"] is False
    assert info["steps"] == 0
    assert info["stopping_reason"] == "time_budget"
    assert info["budget_overrun_seconds"] > 0
    np.testing.assert_allclose(info["budget_overrun_seconds"], info["fit_seconds"] - 1e-12)


def test_training_dtype_does_not_depend_on_global_torch_default():
    torch = pytest.importorskip("torch")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        result = fit_torch(
            np.eye(3), np.array([1.0, -0.5, 0.2]), 0.01, width=4,
            config=dict(max_epochs=1, batch_size=3, fused=False),
        )
    finally:
        torch.set_default_dtype(previous)
    assert result["fit_info"]["dtype"] == "float32"
    assert result["fit_info"]["steps"] == 1
    assert np.isfinite(result["beta"]).all()


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
def test_cuda_fused_default_and_explicit_override_preserve_small_fit(optimizer):
    if os.environ.get("RELUOPT_TEST_ACCELERATOR") != "1":
        pytest.skip("accelerator checks require explicit opt-in")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    X = np.array([[1.0, 0.5, 1.0], [0.2, 1.0, 1.0], [-1.0, 0.1, 1.0], [0.5, -0.3, 1.0]])
    y = np.array([1.0, -1.0, 0.5, 0.2])
    options = dict(optimizer=optimizer, device="cuda", width=5, seed=13, time_budget=30)
    config = dict(max_epochs=1, batch_size=2, sync_every=1)
    fused = fit_torch(X, y, 0.01, config=config, **options)
    unfused = fit_torch(X, y, 0.01, config=dict(config, fused=False), **options)
    assert fused["fit_info"]["fused"] is True
    assert unfused["fit_info"]["fused"] is False
    assert fused["fit_info"]["steps"] == unfused["fit_info"]["steps"] == 2
    np.testing.assert_allclose(
        fused["fit_info"]["selected_train_objective"],
        unfused["fit_info"]["selected_train_objective"], rtol=2e-5, atol=1e-7,
    )
    first = np.maximum(X @ fused["U"], 0) @ fused["beta"]
    second = np.maximum(X @ unfused["U"], 0) @ unfused["beta"]
    np.testing.assert_allclose(first, second, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("options", [
    {"optimizer": "other"}, {"width": 0}, {"time_budget": 0}, {"seed": -1},
    {"config": {"batch_size": 257}}, {"config": {"max_epochs": 1_000_001}},
    {"config": {"lr": 0}}, {"config": {"unknown": 1}},
    {"optimizer": "adam", "config": {"weight_decay": 0.1}},
    {"optimizer": "adamw", "config": {"weight_decay": 0}},
    {"config": {"fused": "yes"}},
])
def test_invalid_options_fail_without_importing_torch(options):
    with pytest.raises(ValueError):
        fit_torch(np.eye(2), np.ones(2), 0.1, **options)
