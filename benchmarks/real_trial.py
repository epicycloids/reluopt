"""Isolated fit process used by the real-data benchmark."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_model(result):
    U = np.asarray(result["U"], dtype=np.float64)
    beta = np.asarray(result["beta"], dtype=np.float64)
    if U.ndim != 2 or beta.shape != (U.shape[1],):
        raise ValueError("invalid exported ReLU model shape")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(beta)):
        raise ValueError("nonfinite exported ReLU model")
    norms = np.linalg.norm(U, axis=0)
    keep = (norms > 0) & (beta != 0)
    return U[:, keep] / norms[keep], beta[keep] * norms[keep]


def fit(X, y, spec):
    solver = spec["solver"]
    device = "cuda" if solver.endswith("-cuda") else "cpu"
    torch = None
    if solver.startswith("scnn"):
        # SCNN's top-level import does not load its optimizer/interface modules.
        # Keep their initialization outside fit time, as for Torch optimizers.
        for module in ("scnn.metrics", "scnn.models", "scnn.private.interface",
                       "scnn.private.models", "scnn.private.methods.core",
                       "scnn.regularizers", "scnn.solvers"):
            import_module(module)
    elif solver == "adelie":
        import adelie  # noqa: F401
    if device == "cuda" or solver in {"sgd", "adam", "adamw"}:
        import torch

        torch.set_num_threads(spec["threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if solver in {"sgd", "adam", "adamw"}:
            device = spec.get("training_device", "cuda")
            # Optimizer module initialization is a library startup cost.
            parameter = torch.nn.Parameter(torch.zeros(1))
            getattr(torch.optim, {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}[solver])([parameter])
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            tiny = torch.ones((2, 2), device="cuda")
            _ = tiny @ tiny
            del tiny
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    if solver.startswith("relu-"):
        from reluopt import solve_rcg
        import scipy.linalg  # noqa: F401: optional CPU kernels used during fitting
        import scipy.optimize  # noqa: F401
    with threadpool_limits(limits=spec["threads"]):
        start = time.perf_counter()
        if solver.startswith("relu-"):
            result = solve_rcg(
                X, y, lam=len(y) * spec["lambda_mean"], device=device,
                max_iter=80, max_cols=spec["width"], sketch_m=8192,
                max_feature_bytes=768 * 1024**2, gated_every=2, gated_iters=60,
                certify_every=0, final_verify=True, final_diagnostics=False,
                eps_heur=1e-4, time_budget=spec["time_budget"], rng_seed=spec["seed"],
                oracle_kwargs=dict(n_random=16, n_subspace=8, iters_sketch=40,
                                   iters_full=8, search_backend="device" if device == "cuda" else "host"),
            )
            result["fit_info"] = {key: result[key] for key in (
                "stop_reason", "n_iter", "finite_master_kkt", "finite_master_converged",
                "verified_float64", "n_cols", "n_active", "timings",
            )}
            result["fit_info"].update(model_kind="relu", regularization="path_norm",
                                      final_diagnostics=False)
        elif solver.startswith("scnn-"):
            from benchmarks.convex_solvers import fit_scnn

            result = fit_scnn(X, y, spec["lambda_mean"], seed=spec["seed"], device=device,
                              width=spec["width"], time_budget=spec["time_budget"], config=spec["config"])
        elif solver == "adelie":
            from benchmarks.convex_solvers import fit_adelie

            result = fit_adelie(X, y, spec["lambda_mean"], seed=spec["seed"], device="cpu",
                                width=spec["width"], time_budget=spec["time_budget"], config=spec["config"])
        elif solver in {"sgd", "adam", "adamw"}:
            from benchmarks.torch_solvers import fit_torch

            result = fit_torch(X, y, spec["lambda_mean"], optimizer=solver, seed=spec["seed"],
                               device=device, width=spec["width"], time_budget=spec["time_budget"],
                               config=spec["config"])
        else:
            raise ValueError(f"unknown solver: {solver}")
        if torch is not None and device == "cuda":
            torch.cuda.synchronize()
        U, beta = normalize_model(result)
        seconds = time.perf_counter() - start
    return U, beta, dict(
        status="ok", seconds=seconds, fit_info=jsonable({
            key: value for key, value in result.get("fit_info", {}).items() if key != "trajectory"
        }),
        trajectory=jsonable(result.get("trajectory", [])),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if torch is not None and device == "cuda" else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    try:
        with np.load(args.input) as arrays:
            U, beta, record = fit(arrays["X_train"], arrays["y_train"], spec)
        np.savez(args.output.with_suffix(".npz"), U=U, beta=beta)
    except Exception as exc:
        record = dict(status="error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    args.output.write_text(json.dumps(jsonable(record), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
