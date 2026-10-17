"""Real regression data: validation-selected fits and fixed-penalty comparisons."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.datasets import load_dataset, prepare_split
from benchmarks.real_trial import jsonable
from benchmarks.run import source_digest


DATASETS = ("concrete", "wine-white", "california", "superconductivity")
SOLVERS = ("relu-cpu", "relu-cuda", "scnn-cuda", "adelie", "sgd", "adam", "adamw")
LAMBDAS = (0.0001, 0.001, 0.01)


def configurations(solver, width=256, time_budget=15, seed=0, threads=4, smoke=False):
    common = dict(solver=solver, width=width, time_budget=time_budget, seed=seed,
                  threads=threads, training_device="cuda")
    lambdas = (0.001,) if smoke else LAMBDAS
    if solver in {"sgd", "adam"}:
        rates = ((0.01,) if solver == "sgd" else (0.001,)) if smoke else (
            (0.01, 0.1) if solver == "sgd" else (0.001, 0.01)
        )
        return [dict(common, lambda_mean=lam, config=dict(lr=lr))
                for lam, lr in itertools.product(lambdas, rates)]
    if solver == "adamw":
        rates = (0.001,) if smoke else (0.001, 0.01)
        decays = (0.01,) if smoke else (0.0001, 0.01)
        return [dict(common, lambda_mean=0.001, config=dict(lr=lr, weight_decay=decay))
                for lr, decay in itertools.product(rates, decays)]
    config = {"dtype": "float32"} if solver == "scnn-cuda" else (
        {"n_threads": threads} if solver == "adelie" else {}
    )
    return [dict(common, lambda_mean=lam, config=dict(config)) for lam in lambdas]


def predict(X, U, beta):
    output = np.zeros(len(X))
    for start in range(0, len(X), 4096):
        output[start:start + 4096] = np.maximum(X[start:start + 4096] @ U, 0) @ beta
    return output


def scores(y, prediction, target_scale):
    prediction = np.asarray(prediction, np.float64)
    if prediction.shape != np.shape(y) or not np.all(np.isfinite(prediction)):
        raise ValueError("invalid or nonfinite predictions")
    residual = np.asarray(y, np.float64) - prediction
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(residual @ residual) / len(y)
    if not np.isfinite(mse):
        raise ValueError("nonfinite squared error")
    centered = np.asarray(y) - np.mean(y)
    total = float(centered @ centered)
    return dict(mse=mse, rmse_original_units=float(np.sqrt(mse) * target_scale),
                r2=1 - float(residual @ residual) / total if total > 0 else None)


def _trial(input_path, spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    spec_path, output = folder / "config.json", folder / "result.json"
    spec_path.write_text(json.dumps(spec, allow_nan=False))
    start = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, "-m", "benchmarks.real_trial", "--input", str(input_path),
             "--spec", str(spec_path), "--output", str(output)],
            capture_output=True, text=True, timeout=spec["time_budget"] + 90,
        )
        if process.returncode or not output.exists():
            return dict(status="error", error=process.stderr[-4000:], process_seconds=time.perf_counter() - start)
        result = json.loads(output.read_text())
    except subprocess.TimeoutExpired:
        result = dict(status="timeout", error="fit process exceeded budget plus startup/verification allowance")
    result["process_seconds"] = time.perf_counter() - start
    return result


def environment(threads=4):
    versions = {}
    for package in ("numpy", "scipy", "torch", "torchvision", "pyscnn", "adelie", "cvxpy", "linalg-backends"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    import torch

    cpu = platform.processor() or "unavailable"
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    with threadpool_limits(limits=threads):
        blas = threadpool_info()
    return dict(python=platform.python_version(), packages=versions,
                cpu=cpu, blas=blas,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def run_dataset(name, seeds=(0, 1, 2), solvers=SOLVERS, width=256, time_budget=15,
                threads=4, output_dir=None, smoke=False):
    """Select on validation loss; evaluate the selected model on test only once."""
    if name not in DATASETS or not solvers or any(s not in SOLVERS for s in solvers):
        raise ValueError("unknown dataset or solver")
    if width < 2 or not np.isfinite(time_budget) or time_budget <= 0 or threads < 1 or not seeds:
        raise ValueError("invalid width, budget, threads, or seeds")
    if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds) or len(set(solvers)) != len(solvers):
        raise ValueError("seeds and solvers must be distinct; seeds must be nonnegative")
    folder = Path(output_dir or tempfile.mkdtemp(prefix="relu-real-" )).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    X, y, dataset_metadata = load_dataset(name)
    records, selected, fixed_penalty, splits = [], [], [], []
    for seed in seeds:
        split = prepare_split(X, y, seed)
        splits.append(split["metadata"])
        split_dir = folder / f"split-{seed}"
        split_dir.mkdir(exist_ok=True)
        input_path = split_dir / "data.npz"
        np.savez(input_path, X_train=split["X_train"], y_train=split["y_train"])
        queues = {s: configurations(s, width, time_budget, seed, threads, smoke) for s in solvers}
        # Interleave methods and rotate the start order between splits.
        order = list(solvers)[seed % len(solvers):] + list(solvers)[:seed % len(solvers)]
        by_solver = {s: [] for s in solvers}
        for config_index in range(max(map(len, queues.values()))):
            for solver in order if config_index % 2 == 0 else list(reversed(order)):
                if config_index >= len(queues[solver]):
                    continue
                spec = queues[solver][config_index]
                trial_dir = split_dir / solver / str(config_index)
                result = _trial(input_path, spec, trial_dir)
                scoring_start = time.perf_counter()
                row = dict(dataset=name, split_seed=seed, solver=solver,
                           config_index=config_index, configuration=spec, **result)
                if result["status"] == "ok":
                    with np.load(trial_dir / "result.npz") as model:
                        U, beta = model["U"], model["beta"]
                    try:
                        with threadpool_limits(limits=threads):
                            train = scores(split["y_train"], predict(split["X_train"], U, beta),
                                           split["transforms"]["y_scale"])
                            validation = scores(split["y_val"], predict(split["X_val"], U, beta),
                                                split["transforms"]["y_scale"])
                        path_norm = float(np.abs(beta).sum())
                        objective = 0.5 * train["mse"] + spec["lambda_mean"] * path_norm
                        if not np.isfinite(objective):
                            raise ValueError("nonfinite training objective")
                    except ValueError as exc:
                        row.update(status="invalid", error=str(exc))
                        print(f"{name} split={seed} {solver} {config_index}: invalid {exc}", flush=True)
                    else:
                        row.update(train=train, validation=validation, n_units=len(beta),
                                   path_norm=path_norm, train_objective=objective)
                        by_solver[solver].append((row, trial_dir / "result.npz"))
                        print(f"{name} split={seed} {solver} {config_index}: {result['seconds']:.3f}s "
                              f"val_rmse={validation['rmse_original_units']:.6g}", flush=True)
                else:
                    print(f"{name} split={seed} {solver} {config_index}: {result['status']} {result.get('error', '')}", flush=True)
                row["scoring_seconds"] = time.perf_counter() - scoring_start
                row["trial_seconds"] = row["process_seconds"] + row["scoring_seconds"]
                records.append(row)
        for solver in solvers:
            candidates = by_solver[solver]
            all_rows = [r for r in records if r["split_seed"] == seed and r["solver"] == solver]
            if not candidates:
                selected.append(dict(dataset=name, split_seed=seed, solver=solver, status="failed",
                                     failed_trials=len(all_rows)))
                continue
            best, model_path = min(candidates, key=lambda pair: pair[0]["validation"]["mse"])
            with np.load(model_path) as model:
                with threadpool_limits(limits=threads):
                    test = scores(split["y_test"], predict(split["X_test"], model["U"], model["beta"]),
                                  split["transforms"]["y_scale"])
            selected.append(dict(dataset=name, split_seed=seed, solver=solver, status="ok", test=test,
                                 validation=best["validation"], train=best["train"],
                                 configuration=best["configuration"], seconds=best["seconds"],
                                 train_objective=best["train_objective"], n_units=best["n_units"],
                                 fit_info=best["fit_info"], process_peak_rss_bytes=best["process_peak_rss_bytes"],
                                 gpu_peak_bytes=best["gpu_peak_bytes"], trials=len(all_rows),
                                 failed_trials=sum(r["status"] != "ok" for r in all_rows),
                                 successful_fit_seconds=sum(r.get("seconds", 0) for r in all_rows),
                                 tuning_seconds=sum(r["trial_seconds"] for r in all_rows)))
            if solver != "adamw":
                fixed = [r for r, _ in candidates if r["configuration"]["lambda_mean"] == 0.001]
                if fixed:
                    winner = min(fixed, key=lambda r: r["train_objective"])
                    fixed_penalty.append({key: winner[key] for key in (
                        "dataset", "split_seed", "solver", "configuration", "train_objective", "seconds",
                        "train", "path_norm", "n_units", "fit_info",
                    )})
    return jsonable(dict(
        schema_version=1, dataset=name, dataset_metadata=dataset_metadata, splits=splits,
        source_sha256=source_digest(), benchmark_sha256=source_digest(Path(__file__).parent),
        environment=environment(threads), protocol=dict(split="approximately_60_20_20_by_raw_input_groups", seeds=list(seeds), width=width,
            time_budget=time_budget, threads=threads, solvers=list(solvers), lambdas=list((0.001,) if smoke else LAMBDAS),
            selection="minimum validation MSE; test scored after selection", smoke=smoke),
        selected=selected, fixed_penalty=fixed_penalty, records=records,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="concrete")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solvers", default=",".join(SOLVERS))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--time-budget", type=float, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/real-data.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_dataset(args.dataset, tuple(map(int, args.seeds.split(","))), args.solvers.split(","),
                         args.width, args.time_budget, args.threads, args.output.parent / (args.output.stem + "-trials"),
                         args.smoke)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
