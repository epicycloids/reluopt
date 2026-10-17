"""Pricing acceleration keeps search values and direction state aligned."""

import numpy as np
import pytest

from reluopt.adaptive_pricing import (
    _ascent_pm,
    _ascent_pm_torch,
    _sketch_spectral,
    search_signed_atoms,
)
from reluopt.solver import make_engine
import reluopt.adaptive_pricing as pricing


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("prune_at", [0, 3])
@pytest.mark.parametrize("mode", ["reuse", "reuse-active"])
def test_reused_projection_matches_both_trials(dtype, prune_at, mode):
    rng = np.random.default_rng(24)
    X = rng.standard_normal((173, 7)).astype(dtype)
    nu = rng.standard_normal(173).astype(dtype)
    U0 = rng.standard_normal((7, 62)).astype(dtype)
    signs = np.r_[np.ones(31), -np.ones(31)]
    kwargs = dict(n_iters=30, prune_at=prune_at, keep_min=9)
    reference = _ascent_pm(X, nu, U0, signs, trial_mode="both", **kwargs)
    reused = _ascent_pm(X, nu, U0, signs, trial_mode=mode, **kwargs)
    tolerance = 4e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(reused[0], reference[0], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(reused[1], reference[1], rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(reused[2], reference[2])
    replay = (np.maximum(X @ reused[1], 0).T @ nu) * reused[2]
    np.testing.assert_allclose(reused[0], replay, rtol=tolerance, atol=tolerance)


def test_reused_projection_torch_cpu():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    X = rng.standard_normal((173, 7), dtype=np.float32)
    nu = rng.standard_normal(173, dtype=np.float32)
    U0 = rng.standard_normal((7, 62), dtype=np.float32)
    signs = np.r_[np.ones(31), -np.ones(31)]
    kwargs = dict(n_iters=20, prune_at=3, keep_min=9, trial_mode="reuse")
    host = _ascent_pm(X, nu, U0, signs, **kwargs)
    device = _ascent_pm_torch(torch.as_tensor(X), torch.as_tensor(nu), U0, signs, **kwargs)
    for left, right in zip(host, device):
        np.testing.assert_allclose(left, right, rtol=4e-5, atol=4e-5)


def test_randomized_spectral_seeds_find_separated_leading_subspace():
    diagonal = np.r_[[40.0, 30.0, 20.0, 10.0], np.linspace(1, 0.1, 20)]
    X = np.diag(diagonal)
    directions, Q = _sketch_spectral(X, np.ones(24), method="randomized")
    np.testing.assert_allclose(Q.T @ Q, np.eye(4), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Q @ Q.T, np.diag(np.r_[np.ones(4), np.zeros(20)]), atol=1e-10)
    np.testing.assert_allclose(directions[:, ::2], -directions[:, 1::2])
    np.testing.assert_allclose(np.sort(np.sum((X @ Q) ** 2, axis=0)), diagonal[:4][::-1] ** 2)


@pytest.mark.parametrize("method", ["exact", "randomized"])
def test_spectral_zero_sketch_stays_finite(method):
    directions, Q = _sketch_spectral(np.zeros((10, 13)), np.ones(10), method=method)
    assert np.isfinite(directions).all()
    np.testing.assert_allclose(Q.T @ Q, np.eye(4))


def test_unknown_spectral_method_rejected():
    with pytest.raises(ValueError, match="spectral_method"):
        _sketch_spectral(np.zeros((2, 3)), np.ones(2), method="other")


def test_retained_dual_correlations_follow_ranked_features():
    rng = np.random.default_rng(17)
    X = rng.standard_normal((73, 5))
    y = rng.standard_normal(73)
    nu = rng.standard_normal(73)
    engine = make_engine(X, y)
    found = search_signed_atoms(
        engine, engine.scatter_vec(nu), nu, lam=0.3, k=3,
        rng=np.random.default_rng(1), n_random=3, iters_sketch=3,
        iters_full=1, candidate_order="decrease", retain_finalist_features=True,
    )
    cache = found["addition_cache"]
    assert cache is not None
    U = np.stack([addition[2] for addition in found["addition"]], axis=1)
    A = np.maximum(X @ U, 0)
    np.testing.assert_allclose(cache["dual_corr"], A.T @ nu, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.concatenate(cache["A_sh"]), A, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "rows,dimensions,expected",
    [(73, 5, "reuse"), (8192, 5, "reuse-active"), (8192, 17, "reuse")],
)
def test_auto_host_trial_resolution(monkeypatch, rows, dimensions, expected):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((rows, dimensions))
    nu = rng.standard_normal(rows)
    engine = make_engine(X, np.zeros(rows))
    original = pricing._ascent_pm
    selected = []

    def record(*args, **kwargs):
        selected.append(kwargs["trial_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing, "_ascent_pm", record)
    pricing.search_signed_atoms(
        engine, engine.scatter_vec(nu), nu, lam=0.1,
        n_random=0, n_subspace=0, iters_sketch=0, iters_full=0,
    )
    assert selected == [expected]


def test_auto_device_trial_resolution(monkeypatch):
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((31, 5))
    nu = rng.standard_normal(31)
    engine = make_engine(X, np.zeros(31), device="torch:cpu")
    original = pricing._ascent_pm_torch
    selected = []

    def record(*args, **kwargs):
        selected.append(kwargs["trial_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing, "_ascent_pm_torch", record)
    pricing.search_signed_atoms(
        engine, engine.scatter_vec(nu), nu, lam=0.1, search_backend="auto",
        n_random=0, n_subspace=0, iters_sketch=0, iters_full=0,
    )
    assert selected == ["reuse"]


def test_unknown_auto_trial_rejected():
    engine = make_engine(np.ones((3, 1)), np.zeros(3))
    with pytest.raises(ValueError, match="ascent_trial"):
        search_signed_atoms(engine, [np.ones(3)], np.ones(3), 0.1, ascent_trial="other")


def test_cuda_graph_replay_uses_new_inputs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from collections import OrderedDict

    from reluopt._cuda_pricing import ascent_cuda_graph

    rng = np.random.default_rng(61)
    signs = np.r_[np.ones(31), -np.ones(31)]
    cache = OrderedDict()
    # First call is eager; second captures; third replays the captured buffers.
    for iteration in range(3):
        # An index plan allocated before capture needs a live owner between
        # replays. Reuse small allocator blocks to expose dangling pointers.
        churn = (
            [torch.full((31,), 10_000, device="cuda", dtype=torch.int64) for _ in range(2)]
            if iteration else []
        )
        X = torch.as_tensor(rng.standard_normal((173, 7)), dtype=torch.float32, device="cuda")
        nu = torch.as_tensor(rng.standard_normal(173), dtype=torch.float32, device="cuda")
        U0 = rng.standard_normal((7, 62)).astype(np.float32)
        kwargs = dict(n_iters=8, prune_at=3, keep_min=9, trial_mode="reuse", device="cuda")
        eager = _ascent_pm_torch(X, nu, U0, signs, **kwargs)
        captured = ascent_cuda_graph(X, nu, U0, signs, graph_cache=cache, **kwargs)
        for left, right in zip(eager, captured):
            np.testing.assert_allclose(
                left, right, rtol=3e-6, atol=3e-6,
                err_msg=f"CUDA graph call {iteration + 1}",
            )
        del churn
    assert len(cache) == 1
