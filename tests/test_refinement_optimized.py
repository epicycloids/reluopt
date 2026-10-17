"""Independent small-problem checks for refinement and pricing bounds."""

import numpy as np
import pytest

from reluopt.consolidation import (
    _make_device_relu_evaluator,
    _refine_relu_lbfgs,
    _signed_relu_value_grad,
    gated_group_lasso,
    propose_consolidation,
)
from reluopt.data_engine import ShardedData
from reluopt.pricing_bounds import (
    _dual_ascent_host,
    _valid_bound_host,
    estimate_pricing_bounds,
)


def objective(X, y, M, Z, lam):
    residual = ((X @ Z) * M).sum(axis=1) - y
    return 0.5 * float(residual @ residual) + lam * np.linalg.norm(Z, axis=0).sum()


@pytest.mark.parametrize("method", ["coordinate", "accelerated"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_disjoint_orthonormal_groups_have_soft_threshold_solution(method, dtype):
    # Each masked design is an orthonormal block on disjoint observations.
    X = np.tile(np.eye(3), (2, 1)).astype(dtype)
    M = np.repeat(np.eye(2), 3, axis=0).astype(dtype)
    y = np.array([1., -2., 3., -1., 0.5, -2.], dtype=dtype)
    lam = 0.4
    target = y.reshape(2, 3).T
    expected = target * np.maximum(0, 1 - lam / np.linalg.norm(target, axis=0))
    Z, value = gated_group_lasso(X, y, M, lam, iters=200, tol=1e-12, method=method)
    np.testing.assert_allclose(Z, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(value, objective(X.astype(float), y, M, Z, lam), atol=2e-6)


@pytest.mark.parametrize("method", ["coordinate", "accelerated"])
def test_unregularized_masked_regression_matches_dense_least_squares(method):
    rng = np.random.default_rng(724)
    X = rng.standard_normal((30, 2))
    M = (rng.standard_normal((30, 3)) > 0).astype(float)
    y = rng.standard_normal(30)
    design = np.concatenate([X * M[:, j:j+1] for j in range(3)], axis=1)
    target, *_ = np.linalg.lstsq(design, y, rcond=None)
    initial = rng.standard_normal((2, 3))
    original = initial.copy()
    Z, value = gated_group_lasso(X, y, M, 0, Z0=initial, iters=1200,
                                tol=1e-13, method=method)
    np.testing.assert_allclose(Z.T.ravel(), target, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(value, 0.5 * np.sum((design @ target-y)**2), rtol=1e-11)
    np.testing.assert_array_equal(initial, original)


@pytest.mark.parametrize("method", ["coordinate", "accelerated"])
def test_regularized_solution_satisfies_group_kkt(method):
    rng = np.random.default_rng(38)
    X = rng.standard_normal((40, 2))
    M = (rng.standard_normal((40, 3)) > 0).astype(float)
    y = rng.standard_normal(40)
    lam = 2.0
    Z, value = gated_group_lasso(X, y, M, lam, iters=1600, tol=1e-13, method=method)
    residual = ((X @ Z) * M).sum(axis=1) - y
    gradient = X.T @ (M * residual[:, None])
    for j in range(3):
        norm = np.linalg.norm(Z[:, j])
        violation = (np.linalg.norm(gradient[:, j] + lam * Z[:, j] / norm)
                     if norm else max(np.linalg.norm(gradient[:, j]) - lam, 0))
        assert violation < 3e-5
    np.testing.assert_allclose(value, objective(X, y, M, Z, lam), rtol=1e-13)


@pytest.mark.parametrize("method", ["coordinate", "accelerated", "lbfgs"])
def test_refinement_preserves_negative_output_atom_orientation(method):
    if method == "lbfgs":
        pytest.importorskip("scipy")
    X = np.array([[1.], [-1.]])
    engine = ShardedData(X, np.array([-1., 0.]))
    U = propose_consolidation(engine, np.ones((1, 1)), np.array([-1.]),
                              lam=0, sketch_m=0, iters=30, method=method)
    np.testing.assert_allclose(U, np.ones((1, 1)), atol=1e-7)


def test_actual_relu_gradient_matches_finite_differences_for_both_signs():
    rng = np.random.default_rng(731)
    X = rng.standard_normal((15, 3))
    y = rng.standard_normal(15)
    V = rng.standard_normal((3, 2))
    signs = np.array([1., -1.])
    lam = 0.3
    value, gradient = _signed_relu_value_grad(V, X, y, signs, lam)
    expected = 0.5 * np.sum((np.maximum(X @ V, 0) @ signs-y)**2) + lam * np.linalg.norm(V, axis=0).sum()
    np.testing.assert_allclose(value, expected, rtol=1e-14)
    step = 1e-6
    for index in np.ndindex(V.shape):
        plus, minus = V.copy(), V.copy()
        plus[index] += step
        minus[index] -= step
        fp, _ = _signed_relu_value_grad(plus, X, y, signs, lam)
        fm, _ = _signed_relu_value_grad(minus, X, y, signs, lam)
        np.testing.assert_allclose(gradient[index], (fp-fm)/(2*step), rtol=2e-6, atol=2e-7)


def test_actual_relu_lbfgs_matches_signed_one_dimensional_optimum():
    pytest.importorskip("scipy")
    X = np.array([[1.], [-1.], [2.], [-2.]])
    y = np.array([1., -2., 2., -4.])
    U = np.array([[1., -1.]])
    beta = np.array([0.4, -0.5])
    original = U.copy(), beta.copy()
    V, value = _refine_relu_lbfgs(X, y, U, beta, lam=0.5, iters=100, tol=1e-12)
    # The two one-dimensional ReLU features are orthogonal and have norm²=5.
    np.testing.assert_allclose(V, [[0.9, -1.9]], atol=2e-6)
    np.testing.assert_allclose(value, 1.45, rtol=1e-11)
    np.testing.assert_array_equal(U, original[0])
    np.testing.assert_array_equal(beta, original[1])


def test_actual_relu_lbfgs_has_finite_monotone_proposal_and_respects_zero_budget():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(147)
    X = rng.standard_normal((25, 3))
    U = rng.standard_normal((3, 3))
    U /= np.linalg.norm(U, axis=0)
    beta = np.array([0.7, -0.3, 0.2])
    y = rng.standard_normal(25)
    V0 = U * np.abs(beta)
    initial, _ = _signed_relu_value_grad(V0, X, y, np.sign(beta), 0.1)
    V, value = _refine_relu_lbfgs(X, y, U, beta, lam=0.1, iters=15)
    assert np.isfinite(V).all()
    assert value <= initial
    actual, _ = _signed_relu_value_grad(V, X, y, np.sign(beta), 0.1)
    np.testing.assert_allclose(value, actual, rtol=1e-14)
    zero, zero_value = _refine_relu_lbfgs(X, y, U, beta, lam=0.1, iters=0)
    np.testing.assert_array_equal(zero, V0)
    assert zero_value == initial


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_actual_relu_callback_matches_cpu_gradient(device, dtype):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(471)
    X = rng.standard_normal((17, 3))
    y = rng.standard_normal(17)
    signs = np.array([1., -1.])
    V = rng.standard_normal((3, 2))
    lam = 0.7
    evaluate = _make_device_relu_evaluator(X, y, signs, lam, device, dtype=dtype)
    value, gradient = evaluate(V)
    cpu_value, cpu_gradient = _signed_relu_value_grad(V, X, y, signs, lam)
    tolerance = 3e-6 if dtype == "float32" else 2e-12
    np.testing.assert_allclose(value, cpu_value, rtol=tolerance)
    np.testing.assert_allclose(gradient, cpu_gradient, rtol=10*tolerance, atol=tolerance)
    # A step large enough to resolve float32 products, away from mask changes.
    direction = rng.standard_normal(V.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-3 if dtype == "float32" else 1e-6
    plus, _ = evaluate(V + step * direction)
    minus, _ = evaluate(V - step * direction)
    diff_tolerance = 2e-3 if dtype == "float32" else 2e-6
    np.testing.assert_allclose(np.sum(gradient * direction),
                               (plus-minus)/(2*step), rtol=diff_tolerance, atol=diff_tolerance)
    value_again, gradient_again = evaluate(V)
    assert value_again == value
    np.testing.assert_array_equal(gradient_again, gradient)


def test_device_callback_rejects_unsupported_dtype_before_torch_import():
    with pytest.raises(ValueError, match="dtype"):
        _make_device_relu_evaluator(np.ones((2, 1)), np.zeros(2),
                                    np.ones(1), 0.1, "cpu", dtype="float16")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_device_proposal_passes_engine_storage_dtype(monkeypatch, dtype):
    torch = pytest.importorskip("torch")
    from reluopt import consolidation

    class Engine:
        backend = "torch"
        devices = ["cuda:0"]
        _dt = torch.float32 if dtype == "float32" else torch.float64
        _Xhost = np.array([[1.], [-1.]])
        _yhost = np.array([-1., 0.])
        n, d = 2, 1

    seen = {}

    def refine(X, y, U, beta, lam, **kwargs):
        seen.update(kwargs)
        return U * np.abs(beta), 0.0

    monkeypatch.setattr(consolidation, "_refine_relu_lbfgs", refine)
    proposed = propose_consolidation(Engine(), np.ones((1, 1)), np.array([-1.]),
                                      lam=0, sketch_m=0, method="lbfgs-device")
    assert seen["device_dtype"] == dtype
    assert seen["device"] == "cuda:0"
    np.testing.assert_array_equal(proposed, np.ones((1, 1)))


def test_device_refinement_requires_torch_cuda_engine():
    engine = ShardedData(np.array([[1.], [-1.]]), np.array([-1., 0.]))
    with pytest.raises(ValueError, match="Torch CUDA engine"):
        propose_consolidation(engine, np.ones((1, 1)), np.array([-1.]),
                              lam=0, method="lbfgs-device")


def test_ascent_zero_row_compression_preserves_guard_and_bound():
    rng = np.random.default_rng(21)
    B = rng.standard_normal((7, 3))
    B[[0, 2, 4]] = 0
    initial = np.eye(3)
    before = _valid_bound_host(B, initial)
    value, Z = _dual_ascent_host(B, n_steps=15, Z0=initial)
    assert value <= before
    np.testing.assert_allclose(value, _valid_bound_host(B, Z), rtol=2e-14)
    np.testing.assert_array_equal(initial, np.eye(3))
    value, Z = _dual_ascent_host(np.zeros((5, 3)), n_steps=3)
    assert value == 0
    assert np.isfinite(Z).all()


def test_joint_and_split_bounds_dominate_enumerated_positive_part_oracle():
    rng = np.random.default_rng(82)
    X = rng.standard_normal((7, 3))
    nu = rng.standard_normal(7)
    engine = ShardedData(X, np.zeros(7))
    bound = estimate_pricing_bounds(engine, nu, n_steps=5, forms=("joint", "split"))
    patterns = (np.arange(128)[:, None] >> np.arange(7)) & 1
    for sign, name in ((1, "plus"), (-1, "minus")):
        weighted = patterns * np.maximum(sign * nu, 0)
        exact = np.linalg.norm(weighted @ X, axis=1).max()
        assert bound["ub_"+name] >= exact * (1-1e-12)
