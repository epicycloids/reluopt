"""One isolated AdamW or learned-direction ReLU fit using training data only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.real_trial import fit as fit_baseline, jsonable, normalize_model


def model_digest(U, beta):
    """Fingerprint the exact exported float64 model shared by paired fits."""
    digest = hashlib.sha256()
    for value in (U, beta):
        value = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit(X, y, spec, initial_U=None, initial_beta=None):
    operation = spec["operation"]
    if operation == "adamw":
        return fit_baseline(X, y, dict(spec, solver="adamw"))
    if operation not in {"coefficient-refit", "warm-rcg", "cold-rcg", "cold-rcg-long"}:
        raise ValueError(f"unknown hybrid operation: {operation}")
    warm = not operation.startswith("cold-rcg")
    if warm and (initial_U is None or initial_beta is None):
        raise ValueError("learned-direction fits require the exported initializer")
    if not warm and (initial_U is not None or initial_beta is not None):
        raise ValueError("cold fits must not receive an initializer")
    if warm:
        actual_hash = model_digest(initial_U, initial_beta)
        if actual_hash != spec["initializer_sha256"]:
            raise ValueError("initializer does not match its recorded fingerprint")

    from reluopt import solve_rcg
    import scipy.linalg  # noqa: F401: optional kernels loaded before fit timing
    import scipy.optimize  # noqa: F401

    device = spec.get("device", "cuda")
    torch = None
    if device == "cuda":
        import torch

        torch.set_num_threads(spec["threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        tiny = torch.ones((2, 2), device="cuda")
        _ = tiny @ tiny
        del tiny
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=spec["threads"]):
        start = time.perf_counter()
        result = solve_rcg(
            X, y, lam=len(y) * spec["lambda_mean"], device=device,
            initial_U=initial_U if warm else None,
            initial_beta=initial_beta if warm else None,
            max_iter=0 if operation == "coefficient-refit" else 80,
            max_cols=spec["max_cols"], sketch_m=8192,
            max_feature_bytes=768 * 1024**2, gated_every=2, gated_iters=60,
            certify_every=0, final_verify=True, final_diagnostics=False,
            eps_heur=1e-4, time_budget=spec["time_budget"], rng_seed=spec["seed"],
            oracle_kwargs=dict(n_random=16, n_subspace=8, iters_sketch=40,
                               iters_full=8, search_backend="device" if device == "cuda" else "host"),
        )
        if torch is not None:
            torch.cuda.synchronize()
        U, beta = normalize_model(result)
        seconds = time.perf_counter() - start
    info = {key: result[key] for key in (
        "stop_reason", "n_iter", "finite_master_kkt", "finite_master_converged",
        "verified_float64", "n_cols", "n_active", "timings",
    )}
    if "initialization" in result:
        info["initialization"] = result["initialization"]
    info.update(model_kind="relu", regularization="path_norm", final_diagnostics=False,
                operation=operation, time_budget=spec["time_budget"],
                budget_overrun_seconds=max(0., seconds - spec["time_budget"]))
    return U, beta, dict(
        status="ok", seconds=seconds, fit_info=jsonable(info),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if torch is not None else None,
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
            initial = {key: arrays[key] for key in ("initial_U", "initial_beta") if key in arrays}
            U, beta, record = fit(arrays["X_train"], arrays["y_train"], spec, **initial)
        np.savez(args.output.with_suffix(".npz"), U=U, beta=beta)
    except Exception as exc:
        record = dict(status="error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    args.output.write_text(json.dumps(jsonable(record), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
