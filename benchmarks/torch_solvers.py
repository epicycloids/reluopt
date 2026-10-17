"""Finite-width Torch prediction baselines for standardized tabular inputs.

The caller supplies the constant input feature and selects hyperparameters on
validation data. This module only uses training data. SGD and Adam optimize
half mean squared error plus the balanced two-layer L2 regularizer. AdamW uses
decoupled weight decay and selects checkpoints by training squared error; it
does not optimize that explicit penalized objective.
"""

from __future__ import annotations

import math
import time

import numpy as np


def _canonical_parameters(hidden_weights, output_weights):
    """Map ``relu(X @ W.T) @ a`` to unit directions and signed coefficients."""
    W = np.asarray(hidden_weights, dtype=np.float64)
    a = np.asarray(output_weights, dtype=np.float64)
    if W.ndim != 2 or a.shape != (W.shape[0],):
        raise ValueError("hidden/output parameter shapes do not agree")
    norms = np.linalg.norm(W, axis=1)
    keep = (norms > 0) & (a != 0)
    return (W[keep] / norms[keep, None]).T.copy(), (a[keep] * norms[keep]).copy()


def _penalized_loss(prediction, target, W, a, lam_mean):
    return 0.5 * (prediction - target).square().mean() + 0.5 * lam_mean * (
        W.square().sum() + a.square().sum()
    )


def fit_torch(
    X,
    y,
    lam_mean,
    *,
    optimizer="adam",
    seed=0,
    device="cpu",
    width=256,
    time_budget=30,
    config=None,
):
    """Fit a finite-width two-layer ReLU model within a measured time budget.

    ``X`` already includes a constant feature. Neither layer has a separate
    bias. The returned ``U`` and ``beta`` represent the selected checkpoint in
    float64; ``fit_info`` describes the actual float32 training objective.

    SGD/Adam select the best measured *canonical* training objective, whereas
    AdamW selects training squared error. The reported trajectory includes
    both canonical and actual optimization objectives. Checkpoint evaluation,
    input transfers, and parameter export count toward the elapsed time. A
    final evaluation/export may overrun the budget; the overrun is reported.
    CUDA defaults to Torch's fused optimizers; ``config['fused']`` can override
    this choice. CPU defaults to unfused. Torch is imported only on a call.
    """
    started = time.perf_counter()
    config = dict(config or {})
    allowed = {
        "lr", "batch_size", "momentum", "weight_decay", "max_epochs",
        "checkpoint_every", "checkpoint_interval", "sync_every", "eval_batch_size",
        "fused",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown Torch solver options: {', '.join(unknown)}")
    if "fused" in config and not isinstance(config["fused"], (bool, np.bool_)):
        raise ValueError("fused must be a boolean")
    if optimizer not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer must be 'sgd', 'adam', or 'adamw'")
    if not np.isfinite(lam_mean) or lam_mean < 0:
        raise ValueError("lam_mean must be finite and nonnegative")
    if not np.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("time_budget must be finite and positive")
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("width must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    lr = float(config.get("lr", 0.03 if optimizer == "sgd" else 0.003))
    momentum = float(config.get("momentum", 0.9))
    weight_decay = float(config.get("weight_decay", 0.01 if optimizer == "adamw" else 0.0))
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not np.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1)")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if optimizer != "adamw" and weight_decay != 0:
        raise ValueError("SGD/Adam use explicit L2 regularization; weight_decay must be zero")
    if optimizer == "adamw" and weight_decay == 0:
        raise ValueError("AdamW requires nonzero decoupled weight_decay")
    integer_defaults = {
        "batch_size": 256, "max_epochs": 1000, "checkpoint_every": 1,
        "sync_every": 10, "eval_batch_size": 4096,
    }
    integers = {}
    for name, default in integer_defaults.items():
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        integers[name] = int(value)
    if integers["batch_size"] > 256:
        raise ValueError("batch_size must be <= 256")
    if integers["max_epochs"] > 1_000_000:
        raise ValueError("max_epochs must be <= 1000000")
    interval = float(config.get("checkpoint_interval", 0.5))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("checkpoint_interval must be finite and positive")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite real matrix")
    if y.shape != (X.shape[0],) or y.dtype.kind not in "fiu" or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector matching X")

    import torch

    target_device = torch.device(device)
    # Explicitly select the fused float32 path supported by all three
    # optimizers in the pinned Torch 2.10 runtime; leave CPU unfused by default.
    fused = bool(config.get("fused", target_device.type == "cuda"))
    generator = torch.Generator(device=target_device).manual_seed(int(seed))
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=target_device)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device=target_device)
    n, d = X.shape
    batch_size = min(integers["batch_size"], n)
    # Start balanced without altering the represented random initial model.
    W = torch.randn(
        width, d, generator=generator, device=target_device, dtype=torch.float32
    ) / math.sqrt(d)
    a = torch.randn(
        width, generator=generator, device=target_device, dtype=torch.float32
    ) / math.sqrt(width)
    balancing = (a.abs() / torch.linalg.vector_norm(W, dim=1).clamp_min(1e-30)).sqrt()
    W *= balancing[:, None]
    a /= balancing.clamp_min(1e-30)
    W = torch.nn.Parameter(W)
    a = torch.nn.Parameter(a)
    if optimizer == "sgd":
        opt = torch.optim.SGD([W, a], lr=lr, momentum=momentum, weight_decay=0.0, fused=fused)
    elif optimizer == "adam":
        opt = torch.optim.Adam([W, a], lr=lr, weight_decay=0.0, fused=fused)
    else:
        opt = torch.optim.AdamW([W, a], lr=lr, weight_decay=weight_decay, fused=fused)

    def synchronize():
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    trajectory = []
    best_score = math.inf
    best_weights = None
    best_index = None
    steps = 0
    epochs = 0
    samples_seen = 0

    def checkpoint():
        nonlocal best_score, best_weights, best_index
        with torch.no_grad():
            rss = torch.zeros((), dtype=torch.float64, device=target_device)
            for start in range(0, n, integers["eval_batch_size"]):
                end = min(n, start + integers["eval_batch_size"])
                error = (Xt[start:end] @ W.T).relu() @ a - yt[start:end]
                rss += error.to(torch.float64).square().sum()
            half_mse = float((0.5 * rss / n).item())
            l2_term = float((0.5 * lam_mean * (W.square().sum() + a.square().sum())).item())
            path_term = float((lam_mean * (torch.linalg.vector_norm(W, dim=1) * a.abs()).sum()).item())
            common_objective = half_mse + path_term
            optimization_objective = None if optimizer == "adamw" else half_mse + l2_term
            score = half_mse if optimizer == "adamw" else common_objective
            finite = math.isfinite(score) and math.isfinite(l2_term)
            if finite and score < best_score:
                best_score = score
                best_weights = (W.detach().clone(), a.detach().clone())
                best_index = len(trajectory)
            synchronize()
            trajectory.append(dict(
                seconds=time.perf_counter() - started,
                step=steps,
                epoch=epochs,
                samples_seen=samples_seen,
                train_loss=half_mse if finite else None,
                train_objective=common_objective if finite else None,
                train_optimization_objective=optimization_objective if finite else None,
                training_l2_penalty=l2_term if finite and optimizer != "adamw" else None,
                finite=finite,
            ))
            return finite

    if not checkpoint():
        raise ValueError("initial float32 training objective is nonfinite")
    stopping_reason = "time_budget"
    stop = trajectory[-1]["seconds"] >= time_budget
    while not stop and epochs < integers["max_epochs"]:
        permutation = torch.randperm(n, generator=generator, device=target_device)
        completed_epoch = True
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            prediction = (Xt[indices] @ W.T).relu() @ a
            if optimizer == "adamw":
                loss = 0.5 * (prediction - yt[indices]).square().mean()
            else:
                loss = _penalized_loss(prediction, yt[indices], W, a, lam_mean)
            loss.backward()
            opt.step()
            steps += 1
            samples_seen += len(indices)
            if steps % integers["sync_every"] == 0:
                synchronize()
                finite = bool(torch.isfinite(loss).item())
                elapsed = time.perf_counter() - started
                if not finite or elapsed >= time_budget:
                    stopping_reason = "nonfinite" if not finite else "time_budget"
                    stop = True
                elif elapsed - trajectory[-1]["seconds"] >= interval:
                    if not checkpoint():
                        stopping_reason = "nonfinite"
                        stop = True
                    elif trajectory[-1]["seconds"] >= time_budget:
                        stop = True
                if stop:
                    completed_epoch = start + batch_size >= n
                    break
        if completed_epoch:
            epochs += 1
            if trajectory[-1]["step"] == steps:
                trajectory[-1]["epoch"] = epochs
        if stop:
            break
        if epochs % integers["checkpoint_every"] == 0 and trajectory[-1]["step"] != steps:
            if not checkpoint():
                stopping_reason = "nonfinite"
                stop = True
            elif trajectory[-1]["seconds"] >= time_budget:
                stop = True
    if not stop and epochs == integers["max_epochs"]:
        stopping_reason = "max_epochs"
    if trajectory[-1]["step"] != steps:
        if not checkpoint():
            stopping_reason = "nonfinite"
    selected = trajectory[best_index]
    Wbest, abest = best_weights
    W64 = Wbest.to("cpu", torch.float64).numpy()
    a64 = abest.to("cpu", torch.float64).numpy()
    U, beta = _canonical_parameters(W64, a64)
    synchronize()
    elapsed = time.perf_counter() - started
    explicit_l2 = None if optimizer == "adamw" else float(
        0.5 * lam_mean * (np.square(W64).sum() + np.square(a64).sum())
    )
    fit_info = dict(
        optimizer=optimizer,
        fused=bool(opt.defaults["fused"]),
        lr=lr,
        momentum=momentum if optimizer == "sgd" else None,
        weight_decay=weight_decay if optimizer == "adamw" else 0.0,
        regularization="decoupled_weight_decay" if optimizer == "adamw" else "explicit_two_layer_l2",
        loss="half_mean_squared_error",
        training_objective=(
            "MSE with decoupled optimizer weight decay; no explicit penalized objective"
            if optimizer == "adamw" else "0.5*mean_squared_error + lam_mean/2*(||W||^2+||a||^2)"
        ),
        checkpoint_selection="training_half_mse" if optimizer == "adamw" else "training_canonical_objective",
        trajectory_objective="0.5*mean_squared_error + lam_mean*canonical_path_norm",
        training_l2_penalty=explicit_l2,
        selected_train_loss=selected["train_loss"],
        selected_train_objective=selected["train_objective"],
        selected_optimization_objective=selected["train_optimization_objective"],
        selected_checkpoint=best_index,
        selected_step=selected["step"],
        selected_epoch=selected["epoch"],
        steps=steps,
        epochs=epochs,
        samples_seen=samples_seen,
        width=int(width),
        n_exported_atoms=len(beta),
        finite_width=True,
        hidden_bias=False,
        output_bias=False,
        balanced_initialization=True,
        dtype="float32",
        export_dtype="float64",
        seed=int(seed),
        device=str(target_device),
        torch_version=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
        batch_size=batch_size,
        max_epochs=integers["max_epochs"],
        checkpoint_every=integers["checkpoint_every"],
        checkpoint_interval=interval,
        sync_every=integers["sync_every"],
        fit_seconds=elapsed,
        time_budget=float(time_budget),
        budget_overrun_seconds=max(0.0, elapsed - time_budget),
        stopping_reason=stopping_reason,
        trajectory=trajectory,
    )
    return dict(U=U, beta=beta, fit_info=fit_info, trajectory=trajectory)
