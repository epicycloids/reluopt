"""Benchmark the installed solver on synthetic CPU/CUDA workloads."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import pstats
import resource
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import reluopt


CASES = {
    "smoke": (256, 4, 4, "gaussian", 0.01),
    "small": (2048, 8, 4, "gaussian", 0.01),
    "correlated": (16384, 16, 8, "correlated", 0.003),
    "wide": (4096, 128, 8, "gaussian", 0.003),
    "many": (32768, 32, 32, "gaussian", 0.003),
    "tall": (262144, 32, 8, "gaussian", 0.003),
    "highdim": (16384, 256, 16, "gaussian", 0.003),
    "heavy": (65536, 64, 16, "student", 0.003),
    "agnostic": (32768, 64, 0, "gaussian", 0.003),
}
SUITES = {
    "smoke": ["smoke"],
    "standard": ["small", "correlated", "wide", "many"],
    "scale": ["tall", "highdim", "heavy", "agnostic"],
}
BACKENDS = ("cpu", "cuda")
VARIANTS = (
    "default", "reuse", "active", "accelerated", "lbfgs", "randomized",
    "device", "device64", "graph",
)
CUDA_VARIANTS = {"device", "device64", "graph"}


def source_digest(root=None):
    """Hash the Python sources used by this run, independently of Git."""
    root = Path(root or Path(reluopt.__file__).parent)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def names_for_suite(suite):
    if suite in SUITES:
        return SUITES[suite]
    if suite in CASES:
        return [suite]
    raise ValueError(f"unknown suite or case: {suite!r}")


def parse_choices(value, allowed, label):
    choices = value.split(",") if isinstance(value, str) else list(value)
    if not choices or any(choice not in allowed for choice in choices):
        raise ValueError(f"{label} must be chosen from {', '.join(allowed)}")
    if len(set(choices)) != len(choices):
        raise ValueError(f"{label} must not contain duplicates")
    return choices


def make_problem(name, seed):
    n, d, units, distribution, mean_lam = CASES[name]
    rng = np.random.default_rng(seed)
    X = (
        rng.standard_t(3, size=(n, d)) / np.sqrt(3)
        if distribution == "student"
        else rng.standard_normal((n, d))
    )
    if distribution == "correlated":
        X = 0.95 * X[:, :1] + 0.05 * X
    if units:
        U = rng.standard_normal((d, units))
        U /= np.linalg.norm(U, axis=0)
        beta = rng.choice([-1.0, 1.0], units) / np.sqrt(units)
        y = np.maximum(X @ U, 0) @ beta
    else:
        y = np.sin(2 * X[:, 0]) + 0.3 * (X[:, 1] ** 2 - 1) + 0.2 * X[:, 2] * X[:, 3]
    y += 0.05 * rng.standard_normal(n)
    return X, y, n * mean_lam


def options(variant, backend):
    """Solver settings for one execution configuration."""
    if variant not in VARIANTS or backend not in BACKENDS:
        raise ValueError("unknown variant or backend")
    if variant in CUDA_VARIANTS and backend != "cuda":
        raise ValueError(f"variant {variant!r} requires the CUDA backend")
    params = dict(
        max_iter=40, add_per_round=8, max_cols=256,
        max_feature_bytes=768 * 1024**2, sketch_m=8192,
        certify_every=0, gated_every=2, gated_iters=60,
        final_verify=True, eps_heur=1e-4, time_budget=180, rng_seed=19,
        oracle_kwargs=dict(
            n_random=16, n_subspace=8, iters_sketch=40, iters_full=8,
            search_backend="host" if backend == "cpu" else "device",
            retain_finalist_features=True,
            retained_feature_max_bytes=192 * 1024**2,
        ),
    )
    if variant != "default":
        params["oracle_kwargs"]["ascent_trial"] = "reuse"
    if variant == "randomized":
        params["oracle_kwargs"]["spectral_method"] = "randomized"
    if variant == "active":
        params["oracle_kwargs"].update(ascent_trial="reuse-active", search_backend="host")
    if variant in {"accelerated", "lbfgs"}:
        params["gated_method"] = variant
    if variant in CUDA_VARIANTS:
        params["gated_method"] = "lbfgs-device"
    if variant == "graph":
        params["oracle_kwargs"]["search_backend"] = "device-graph"
    if variant == "device64":
        params["dtype_data"] = "float64"
    return params


def replay(X, y, lam, result):
    """Independently check the returned model using original float64 input."""
    U, beta = result["U"], result["beta"]
    residual = y.copy()
    corr = np.zeros(len(beta))
    for start in range(0, len(y), 4096):
        residual[start:start + 4096] -= np.maximum(X[start:start + 4096] @ U, 0) @ beta
    for start in range(0, len(y), 4096):
        corr += np.maximum(X[start:start + 4096] @ U, 0).T @ residual[start:start + 4096]
    objective = 0.5 * float(residual @ residual) + lam * float(np.abs(beta).sum())
    violation = np.maximum(np.abs(corr) - lam, 0)
    active = beta != 0
    violation[active] = np.abs(corr[active] - lam * np.sign(beta[active]))
    kkt = float(violation.max(initial=0))
    if not np.isclose(objective, result["obj"], rtol=1e-9, atol=1e-9):
        raise AssertionError("original-data objective replay differs")
    if not result["finite_master_converged"] or kkt > 2e-8 * max(1, lam):
        raise AssertionError(f"unconverged finite dictionary: {kkt}")
    return objective, kkt


def _profile_entries(profiler):
    entries = sorted(pstats.Stats(profiler).stats.items(), key=lambda item: -item[1][3])[:30]
    return [
        dict(
            function=f"{Path(key[0]).name}:{key[1]}:{key[2]}",
            calls=value[1], own_seconds=value[2], cumulative_seconds=value[3],
        )
        for key, value in entries
    ]


def run_suite(suite="smoke", backends=("cpu",), variants=("default",),
              repeats=3, seed=177, profile=False, threads=4):
    """Run current configurations with one warm-up and alternating order.

    Records use repeat -1 for the warm-up. With ``profile=True`` the first
    measured repeat is instrumented and marked in each record.
    """
    names = names_for_suite(suite)
    backends = parse_choices(backends, BACKENDS, "backends")
    variants = parse_choices(variants, VARIANTS, "variants")
    if repeats < 1 or threads < 1:
        raise ValueError("repeats and threads must be positive")
    runs = [
        (variant, backend)
        for backend in backends for variant in variants
        if variant not in CUDA_VARIANTS or backend == "cuda"
    ]
    if not runs:
        raise ValueError("device, device64, and graph variants require --backends cuda")
    torch = None
    if "cuda" in backends:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    records = []
    blas_info = []
    for name in names:
        X, y, lam = make_problem(name, seed)
        for repeat in range(-1, repeats):
            order = runs if repeat % 2 else list(reversed(runs))
            for variant, backend in order:
                params = options(variant, backend)
                profiler = cProfile.Profile() if profile and repeat == 0 else None
                if torch:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                with threadpool_limits(limits=threads):
                    start = time.perf_counter()
                    if profiler:
                        profiler.enable()
                    result = reluopt.solve_rcg(
                        X, y, lam, device="cpu" if backend == "cpu" else "cuda", **params
                    )
                    if torch:
                        torch.cuda.synchronize()
                    if profiler:
                        profiler.disable()
                    elapsed = time.perf_counter() - start
                    objective, kkt = replay(X, y, lam, result)
                    blas_info = threadpool_info()
                record = dict(
                    case=name, variant=variant, backend=backend, repeat=repeat,
                    profiled=profiler is not None, seconds=elapsed,
                    objective=objective, objective_per_row=objective / len(y),
                    finite_kkt=kkt, active=result["n_active"], rounds=result["n_iter"],
                    stop_reason=result["stop_reason"], timings=result["timings"],
                    parameters=params, lam=lam,
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    gpu_peak_bytes=(
                        int(torch.cuda.max_memory_allocated())
                        if torch is not None and backend == "cuda" else None
                    ),
                )
                if profiler:
                    record["profile"] = _profile_entries(profiler)
                records.append(record)
                print(
                    f"{name} {variant}/{backend} {repeat}: {elapsed:.3f}s "
                    f"obj/n={objective / len(y):.8g} kkt={kkt:.2g}", flush=True,
                )
    cpu_model = "unknown"
    if Path("/proc/cpuinfo").exists():
        cpu_model = next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.startswith("model name")), cpu_model,
        )
    try:
        scipy_version = version("scipy")
    except PackageNotFoundError:
        scipy_version = None
    return dict(
        schema_version=2, source_sha256=source_digest(),
        benchmark_sha256=source_digest(Path(__file__).parent),
        suite=suite, seed=seed, repeats=repeats, warmups=1, profiled=profile,
        threads=threads, backends=backends, variants=variants,
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy_version,
            cpu_model=cpu_model, blas=blas_info,
            torch=str(torch.__version__) if torch else None,
            gpu=torch.cuda.get_device_name() if torch else None,
        ),
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(set(SUITES) | set(CASES)), default="smoke")
    parser.add_argument("--backends", default="cpu", help="Comma-separated cpu,cuda")
    parser.add_argument("--variants", default="default", help="Comma-separated " + ",".join(VARIANTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile", action="store_true", help="Instrument the first measured repeat")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    args = parser.parse_args()
    result = run_suite(
        args.suite, args.backends, args.variants, args.repeats, args.seed, args.profile, args.threads
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
