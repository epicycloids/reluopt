"""Optional finite-cone ReLU baselines from SCNN and adelie.

Both adapters optimize sampled activation-cone programs with loss
``0.5 * mean((prediction-y)**2) + lam_mean * sum(group_norms)``. The supplied
constant feature is penalized along with every other coordinate. There is no
additional intercept or column normalization. Imports remain local to each
adapter; neither dependency is required by reluopt.

Source APIs inspected: pyscnn 0.0.10 (pilancilab/scnn), adelie 1.1.52
(JamesYang007/adelie). Adelie requires NumPy < 2. SCNN's public optimize_model
discards exit status and has no time callback, so its own AL procedure is
constructed through the pinned package's private interface. Its inequality
multiplier update is corrected to use signed gaps; native FISTA, augmented
objective, stopping criteria, and violation metrics are retained. Returned
U/beta always describe actual signed ReLU predictions,
including when an interrupted cone solve has not reached feasibility.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
from importlib.metadata import PackageNotFoundError, version
import logging
import signal
import threading
import time

import numpy as np


SCNN_DEFAULTS = dict(
    dtype="float64", max_primal_iters=10000, max_dual_iters=10000,
    tol=1e-6, constraint_tol=1e-6, metric_freq=25,
    use_delta_init=False, delta=1000.0,
)
ADELIE_DEFAULTS = dict(
    dtype="float64", solver_mode="pin", max_iters=100000, tol=1e-7, newton_tol=1e-12,
    newton_max_iters=1000, n_threads=4,
    constraint_tol=1e-9, constraint_max_iters=100,
)


def _config(defaults, updates):
    config = dict(defaults)
    if updates:
        unknown = set(updates) - set(defaults)
        if unknown:
            raise ValueError(f"unknown adapter options: {sorted(unknown)}")
        config.update(updates)
    if config["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    for key, value in config.items():
        if key == "solver_mode":
            if value not in {"grpnet", "pin"}:
                raise ValueError("solver_mode must be grpnet or pin")
            continue
        if key == "use_delta_init":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("use_delta_init must be boolean")
            continue
        if key != "dtype" and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be positive and finite")
        if (key.startswith("max_") or key.endswith("max_iters") or key in {"metric_freq", "n_threads"}):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be an integer")
    return config


def _inputs(X, y, lam_mean, width, time_budget, dtype):
    X = np.asarray(X)
    y = np.asarray(y)
    if (X.ndim != 2 or not all(X.shape) or y.shape != (X.shape[0],)
            or np.iscomplexobj(X) or np.iscomplexobj(y)):
        raise ValueError("expected real X(n,d) and y(n)")
    X, y = np.asarray(X, dtype=dtype), np.asarray(y, dtype=dtype)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    if not np.isfinite(lam_mean) or lam_mean < 0:
        raise ValueError("lam_mean must be nonnegative and finite")
    if not isinstance(width, (int, np.integer)) or width < 2:
        raise ValueError("width must be an integer of at least 2")
    if time_budget is not None and (not np.isfinite(time_budget) or time_budget <= 0):
        raise ValueError("time_budget must be positive or None")
    return X, y


def _patterns(X, width, seed):
    """Canonical host-float64 patterns, shared by both finite-cone solvers."""
    X = np.asarray(X, dtype=np.float64)
    gates = np.random.default_rng(seed).standard_normal((X.shape[1], width // 2))
    masks, indices = np.unique(X @ gates > 0, axis=1, return_index=True)
    nonzero = np.any(masks, axis=0)
    return gates[:, indices[nonzero]], np.asfortranarray(masks[:, nonzero])


def _relu_parameters(blocks):
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[0] != 2 or not np.isfinite(blocks).all():
        raise ValueError("expected finite positive/negative cone blocks (2,p,d)")
    vectors = blocks.reshape(-1, blocks.shape[-1])
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 0
    signs = np.r_[np.ones(blocks.shape[1]), -np.ones(blocks.shape[1])]
    return (vectors[keep] / norms[keep, None]).T, signs[keep] * norms[keep]


def _program_stats(X, y, blocks, masks, lam_mean):
    """Separate the finite cone objective from the actual returned ReLU fit."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    blocks = np.asarray(blocks, np.float64)
    U, beta = _relu_parameters(blocks)
    actual = np.maximum(X @ U, 0) @ beta
    finite = np.sum((X @ (blocks[0] - blocks[1]).T) * masks, axis=1)
    penalty = float(np.linalg.norm(blocks, axis=-1).sum())
    max_violation = 0.0
    violation_squared = 0.0
    for sign_blocks in blocks:
        gap = np.maximum(-(2 * masks.astype(np.float64) - 1) * (X @ sign_blocks.T), 0)
        max_violation = max(max_violation, float(gap.max(initial=0)))
        violation_squared += float(np.sum(gap**2))
    return U, beta, dict(
        objective_mean=0.5 * float(np.mean((actual - y) ** 2)) + lam_mean * penalty,
        finite_program_objective_mean=(
            0.5 * float(np.mean((finite - y) ** 2)) + lam_mean * penalty
        ),
        path_norm=penalty,
        cone_violation_max=max_violation,
        cone_violation_squared=violation_squared,
        prediction_mapping_max_error=float(np.max(np.abs(actual - finite), initial=0)),
    )


def _version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _result(X, y, blocks, masks, lam_mean, info):
    U, beta, stats = _program_stats(X, y, blocks, masks, lam_mean)
    info.update(stats)
    info.update(
        model_kind="relu_sampled_cones",
        objective_type="half_mean_squared_error_plus_relu_path_l1",
        lam_mean=float(lam_mean), bias=False, column_normalization=False,
        gate_count=int(masks.shape[1]),
        pattern_source="original_input_float64",
        pattern_sha256=hashlib.sha256(np.packbits(masks, axis=0).tobytes()).hexdigest(),
    )
    return dict(U=U, beta=beta, fit_info=info, trajectory=info.get("trajectory", []))


class _DeadlineCriterion:
    def __init__(self, criterion, deadline):
        self.criterion = criterion
        self.deadline = deadline
        self.expired = False

    def __call__(self, *args, **kwargs):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.expired = True
            return True
        return self.criterion(*args, **kwargs)


def _signed_gap_scnn_optimizer(native_optimizer):
    """Correct SCNN 0.0.10's clipped-gap inequality multiplier update.

    The native model reports max(g, 0) for feasibility metrics. Its outer
    optimizer also feeds that value into the dual update, preventing positive
    multipliers from decreasing on strictly feasible constraints. Keep its
    initialization, violation norms, tolerance schedule, and optimizer reset;
    replace only the resulting multiplier assignment with max(mu+delta*g, 0).
    No package classes or module globals are modified.
    """
    import lab
    from scnn.private.methods.core import update_multipliers

    class SignedGapAugmentedLagrangian(type(native_optimizer)):
        def step(self, model, inner_term_criterion, inner_optimizer, X, y,
                 batch_index=None):
            update_dual = self.delta is not None and not self.initializing_delta
            if update_dual:
                old_e, old_i = model.e_multipliers, model.i_multipliers
                signed_i_gap = model.i_constraint_gap(X)
            result = super().step(
                model, inner_term_criterion, inner_optimizer, X, y, batch_index,
            )
            if update_dual:
                model.e_multipliers, model.i_multipliers = update_multipliers(
                    old_e, old_i, lab.zeros_like(old_e), signed_i_gap, model.delta,
                )
            return result

    corrected = SignedGapAugmentedLagrangian()
    corrected.__dict__.update(native_optimizer.__dict__)
    return corrected


def _preserve_scnn_torch_state(function):
    """Contain LAB's global autograd/default-dtype changes on all exit paths."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch

        previous_dtype = torch.get_default_dtype()
        with torch.no_grad():
            try:
                return function(*args, **kwargs)
            finally:
                torch.set_default_dtype(previous_dtype)
    return wrapped


@_preserve_scnn_torch_state
def fit_scnn(X, y, lam_mean, *, seed=0, device="cpu", width=256,
             time_budget=30, config=None):
    """Fit SCNN ConvexReLU with its augmented-Lagrangian/FISTA solver.

    Time is checked before each primal step. Setup and individual steps can
    overrun the budget. Native stopping tolerances are squared gradient and
    total squared cone-violation tolerances, rather than coefficient KKT norms.
    The native AL penalty-window initialization is disabled by default: it
    keeps changing the penalty when the optimum lies on a cone boundary.
    ``use_delta_init=True`` selects the package's initialization heuristic.
    The package's clipped-gap dual update is corrected to use signed gaps.
    """
    started = time.perf_counter()
    config = _config(SCNN_DEFAULTS, config)
    reference_X, reference_y = X, y
    X, y = _inputs(X, y, lam_mean, width, time_budget, config["dtype"])
    reference_X, reference_y = np.asarray(reference_X, np.float64), np.asarray(reference_y, np.float64)
    if device not in {"cpu", "cuda"}:
        raise ValueError("SCNN device must be cpu or cuda")
    import lab
    from scnn.metrics import Metrics
    from scnn.models import ConvexReLU
    from scnn.private.interface import (
        build_metrics_tuple, build_optimizer, set_device, update_public_metrics,
    )
    from scnn.private.models import AL_MLP, GroupL1Regularizer
    from scnn.regularizers import NeuronGL1
    from scnn.solvers import AL

    gates, masks = _patterns(reference_X, width, seed)
    info = dict(
        solver="scnn_convex_relu_al", package="pyscnn", package_version=_version("pyscnn"),
        device=device, seed=int(seed), width_requested=int(width), options=config,
        time_budget_seconds=time_budget, budget_check="before_each_primal_step",
        regularization_path=[float(lam_mean)],
        upstream_corrections=["pyscnn_0.0.10_signed_inequality_multiplier_update"],
    )
    if not masks.shape[1]:
        info.update(status="empty_sampled_dictionary", iterations=0, elapsed_seconds=time.perf_counter() - started)
        return _result(reference_X, reference_y, np.zeros((2, 0, X.shape[1])), masks, lam_mean, info)
    set_device(device, config["dtype"], seed)
    public_model = ConvexReLU(gates, c=1, bias=False)
    solver = AL(
        public_model, max_primal_iters=int(config["max_primal_iters"]),
        max_dual_iters=int(config["max_dual_iters"]), tol=config["tol"],
        constraint_tol=config["constraint_tol"],
    )
    regularizer = NeuronGL1(lam=lam_mean)
    metrics = Metrics(
        metric_freq=int(config["metric_freq"]), model_loss=True, constraint_gaps=True,
    )
    # Construct backend-native tensors before any backend operation. In the
    # public process_data path, expand_dims is called on a NumPy target while
    # the CUDA backend expects torch.Tensor. Install canonical D directly in
    # the native constructor so masks/orthants/dual arrays share one dictionary
    # even when products subsequently use float32.
    train = (
        lab.tensor(X, dtype=lab.get_dtype()),
        lab.tensor(y[:, None], dtype=lab.get_dtype()),
    )
    test = train
    internal = AL_MLP(
        X.shape[1], lab.tensor(masks, dtype=lab.get_dtype()),
        lab.tensor(gates, dtype=lab.get_dtype()), kernel="einsum",
        delta=config["delta"], regularizer=GroupL1Regularizer(lam_mean), c=1,
    )
    procedure = build_optimizer(solver, regularizer, metrics)
    procedure.outer_optimizer = _signed_gap_scnn_optimizer(procedure.outer_optimizer)
    procedure.outer_optimizer.use_delta_init = config["use_delta_init"]
    procedure.metric_freq = int(config["metric_freq"])
    deadline = _DeadlineCriterion(
        procedure.outer_term_criterion,
        None if time_budget is None else started + time_budget,
    )
    procedure.outer_term_criterion = deadline
    optimization_offset = time.perf_counter() - started
    exit_status, internal, metric_log = procedure(
        logging.getLogger("benchmarks.scnn"), internal, lambda model: model,
        train, test, build_metrics_tuple(metrics),
    )
    public_metrics = update_public_metrics(metrics, metric_log)
    blocks = np.asarray(lab.to_np(internal.weights), dtype=np.float64)[:, 0]
    actual_masks = np.asarray(lab.to_np(internal.D), dtype=bool)
    if not np.array_equal(actual_masks, masks):
        raise RuntimeError("SCNN activation patterns differ from the shared sampled dictionary")
    info.update(
        status="time_budget" if deadline.expired else (
            "converged_native_criterion" if exit_status.get("success") else "iteration_limit"
        ),
        iterations=int(exit_status["total_inner_iterations"]),
        outer_iterations=int(exit_status["outer_iterations"]), native_exit_status=exit_status,
        termination_test="squared_augmented_gradient_and_total_squared_cone_violation",
        elapsed_seconds=time.perf_counter() - started,
        trajectory_objective="sampled_cone_program_before_exact_feasibility",
        trajectory=[
            dict(seconds=optimization_offset + float(seconds), objective_mean=float(objective),
                 cone_violation_squared=float(gap))
            for seconds, objective, gap in zip(
                public_metrics.time, public_metrics.model_loss, public_metrics.constraint_gaps
            )
        ],
    )
    return _result(reference_X, reference_y, blocks, masks, lam_mean, info)


class _AdelieBudgetStop(RuntimeError):
    pass


@contextmanager
def _native_deadline(deadline):
    """Use adelie's PyErr_CheckSignals hooks to retain a timed partial state."""
    if deadline is None:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("a timed adelie fit must run in a worker's main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.perf_counter()

    def expired(signum, frame):
        raise _AdelieBudgetStop("adelie benchmark time budget reached")

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(deadline - started, 1e-6))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL, max(previous_timer[0] - (time.perf_counter() - started), 1e-6),
                previous_timer[1],
            )


def _screen_coefficients(state, d, groups):
    coefficients = np.zeros(groups * d)
    for group, begin in zip(state.screen_set, state.screen_begins):
        coefficients[int(group) * d:(int(group) + 1) * d] = state.screen_beta[int(begin):int(begin) + d]
    return coefficients


def fit_adelie(X, y, lam_mean, *, seed=0, device="cpu", width=256,
               time_budget=30, config=None):
    """Fit adelie's constrained group lasso for the same sampled ReLU cones.

    One explicit lambda is solved, with unit group penalty factors and no
    intercept. A native interrupt deadline can stop coordinate descent at its
    signal checks; it is not a hard preemption deadline. Partial iterates and
    native errors are reported explicitly, and scored as actual ReLU models.

    ``solver_mode='grpnet'`` uses adelie's pathwise screening driver.
    ``'pin'`` uses its public Gaussian pin-state solver with every group in
    the screen set. This solves the same problem without estimating lambda-max
    or screening groups; it precomputes every group's Gram eigendecomposition.
    """
    started = time.perf_counter()
    config = _config(ADELIE_DEFAULTS, config)
    reference_X, reference_y = X, y
    X, y = _inputs(X, y, lam_mean, width, time_budget, config["dtype"])
    reference_X, reference_y = np.asarray(reference_X, np.float64), np.asarray(reference_y, np.float64)
    if device != "cpu":
        raise ValueError("adelie is a CPU solver; use device='cpu'")
    import adelie as ad

    gates, masks = _patterns(reference_X, width, seed)
    count, d = masks.shape[1], X.shape[1]
    info = dict(
        solver="adelie_constrained_relu_group_lasso", package="adelie",
        package_version=_version("adelie"), device=device, seed=int(seed),
        width_requested=int(width), options=config, time_budget_seconds=time_budget,
        budget_check="native_PyErr_CheckSignals",
        regularization_path=[float(lam_mean)],
        iterations=None, iteration_note="adelie does not expose its total coordinate-descent count",
        native_solver_mode=config["solver_mode"],
    )
    blocks = np.zeros((2, count, d))
    if not count:
        info.update(status="empty_sampled_dictionary", path_solutions=0, elapsed_seconds=time.perf_counter() - started)
        return _result(reference_X, reference_y, blocks, masks, lam_mean, info)
    X = np.asfortranarray(X)
    design = ad.matrix.convex_relu(X, masks, gated=False, n_threads=int(config["n_threads"]))
    # Native feature operations prefer columns; native constraint operations
    # prefer rows. Share one additional row-major copy across all constraints.
    shared = ad.matrix.dense(np.ascontiguousarray(X), method="constraint", copy=False)
    row_norms_squared = np.sum(X**2, axis=1)
    constraints = []
    for _sign in range(2):
        for mask in masks.T:
            constraints.append(ad.constraint.linear(
                shared, np.where(mask, 0.0, -np.inf).astype(X.dtype),
                np.where(mask, np.inf, 0.0).astype(X.dtype),
                vars=row_norms_squared, dtype=X.dtype.type,
                configs=dict(tol=config["constraint_tol"], max_iters=int(config["constraint_max_iters"]),
                             n_threads=int(config["n_threads"])),
            ))
    state = None
    setup_timeout = False
    deadline = None if time_budget is None else started + time_budget
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise _AdelieBudgetStop("adelie benchmark time budget reached")
        with _native_deadline(deadline):
            options = dict(
                X=design, constraints=constraints,
                groups=np.arange(0, 2 * count * d, d), alpha=1.0,
                penalty=np.ones(2 * count), lmda_path=np.array([lam_mean], dtype=X.dtype),
                intercept=False,
                max_iters=int(config["max_iters"]), tol=config["tol"],
                newton_tol=config["newton_tol"], newton_max_iters=int(config["newton_max_iters"]),
                n_threads=int(config["n_threads"]),
            )
            if config["solver_mode"] == "grpnet":
                state = ad.grpnet(
                    glm=ad.glm.gaussian(y), early_exit=False, progress_bar=False, **options,
                )
            else:
                # Every possible active group is included, satisfying the
                # pin solver's screen-set contract without a screening pass.
                # With intercept=False, its null MSE is mean(y**2), including
                # when callers supply a response that has not been centered.
                state = ad.state.gaussian_pin_naive(
                    y_mean=float(np.mean(y)), y_var=float(np.mean(y**2)),
                    weights=np.full(X.shape[0], 1 / X.shape[0], dtype=X.dtype),
                    screen_set=np.arange(2 * count),
                    screen_beta=np.zeros(2 * count * d, dtype=X.dtype),
                    screen_is_active=np.zeros(2 * count, dtype=bool),
                    active_set_size=0, active_set=np.zeros(2 * count, dtype=int),
                    resid=y, rsq=0.0, adev_tol=1.0, ddev_tol=0.0, **options,
                )
                state = state.solve()
    except _AdelieBudgetStop:
        # A signal during Python-side setup has no native iterate yet. Return
        # the solver's zero initial model, identified explicitly in metadata.
        setup_timeout = True
    error = "" if state is None else str(getattr(state, "error", ""))
    completed = [] if state is None else np.asarray(state.lmdas).tolist()
    target_completed = bool(completed and np.isclose(completed[-1], lam_mean, rtol=1e-12, atol=0))
    if target_completed:
        beta_path = state.betas[-1]
        coefficients = np.asarray(beta_path.toarray() if hasattr(beta_path, "toarray") else beta_path).ravel()
        blocks = coefficients.reshape(2, count, d)
    elif state is not None:
        blocks = _screen_coefficients(state, d, 2 * count).reshape(2, count, d)
    timed_out = setup_timeout or "adelie benchmark time budget reached" in error
    info.update(
        status="time_budget_setup" if setup_timeout else "time_budget" if timed_out else (
            "solver_error" if error else "converged_native_criterion" if target_completed else "target_not_completed"
        ),
        native_error=error, path_solutions=len(completed), completed_regularization_path=completed,
        parameter_source="completed_path" if target_completed else "zero_initial" if state is None else "partial_screen_state",
        termination_test="adelie_coordinate_descent_tolerance",
        elapsed_seconds=time.perf_counter() - started,
    )
    return _result(reference_X, reference_y, blocks, masks, lam_mean, info)
