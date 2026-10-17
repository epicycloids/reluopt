"""Paired AdamW initialization, coefficient refit, and ReLU column generation."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.datasets import load_dataset, prepare_split
from benchmarks.hybrid_trial import model_digest
from benchmarks.real import DATASETS, LAMBDAS, configurations, environment, predict, scores
from benchmarks.real_trial import jsonable
from benchmarks.run import source_digest


TRACKS = ("adamw", "coefficient-refit", "warm-rcg", "cold-rcg", "cold-rcg-long", "adamw-long")


def chamber_count(X, U):
    """Count distinct learned training activation patterns without deduplicating directions."""
    if U.shape[1] == 0:
        return 0
    patterns = [np.packbits(X @ U[:, start:start + 32] >= 0, axis=0)
                for start in range(0, U.shape[1], 32)]
    return np.unique(np.concatenate(patterns, axis=1), axis=1).shape[1]


def _input_digest(path):
    """Hash array contents independently of NPZ ZIP metadata and member order."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for name in sorted(arrays.files):
            array = np.asarray(arrays[name])
            for field in (name, str(array.shape), array.dtype.str):
                digest.update(field.encode() + b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _trial_source_digest():
    """Bind reusable fits to the solver and the fit implementation in this process."""
    digest = hashlib.sha256(source_digest().encode())
    for name in ("hybrid.py", "hybrid_trial.py", "real_trial.py", "torch_solvers.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _model_fingerprint(path):
    with np.load(path, allow_pickle=False) as arrays:
        U, beta = np.asarray(arrays["U"], dtype=np.float64), np.asarray(arrays["beta"], dtype=np.float64)
    if U.ndim != 2 or beta.shape != (U.shape[1],) or not np.isfinite(U).all() or not np.isfinite(beta).all():
        raise ValueError("invalid cached model")
    return model_digest(U, beta)


def _trial_environment():
    """Read installed runtime versions without importing Torch or initializing CUDA."""
    packages = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return dict(python=platform.python_version(), packages=packages)


def _trial(input_path, spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    spec_path, output = folder / "config.json", folder / "result.json"
    cache_path, model_path = folder / "trial-cache.json", output.with_suffix(".npz")
    identity = dict(schema_version=1,
                    configuration_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"),
                                                                  allow_nan=False).encode()).hexdigest(),
                    input_sha256=_input_digest(input_path), source_sha256=_trial_source_digest(),
                    environment=_trial_environment())
    try:
        cache = json.loads(cache_path.read_text())
        result = cache["result"]
        timing_ok = all(isinstance(result.get(key), (float, int)) and not isinstance(result.get(key), bool)
                        and np.isfinite(result[key]) and result[key] >= 0 for key in ("seconds", "process_seconds"))
        if (all(cache.get(key) == value for key, value in identity.items())
                and result.get("status") == "ok" and timing_ok
                and cache["model_sha256"] == _model_fingerprint(model_path)):
            return result
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        pass
    # A completed cache is the only reusable state. Remove unfinished/stale output
    # before rerunning, so a failed child cannot inherit a preceding result.
    for path in (cache_path, output, model_path):
        path.unlink(missing_ok=True)
    spec_path.write_text(json.dumps(spec, allow_nan=False))
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, "-m", "benchmarks.hybrid_trial", "--input", str(input_path),
             "--spec", str(spec_path), "--output", str(output)],
            capture_output=True, text=True, timeout=spec["time_budget"] + 90,
        )
        if process.returncode or not output.exists():
            result = dict(status="error", error=process.stderr[-4000:])
        else:
            result = json.loads(output.read_text())
    except subprocess.TimeoutExpired:
        result = dict(status="timeout", error="fit process exceeded budget plus startup/verification allowance")
    result["process_seconds"] = time.perf_counter() - started
    if result.get("status") == "ok":
        try:
            fingerprint = _model_fingerprint(model_path)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
        else:
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(identity, model_sha256=fingerprint, result=result),
                                            allow_nan=False) + "\n")
            temporary.replace(cache_path)
    return result


def run_dataset(name, seeds=(0, 1, 2), width=256, max_cols=512, time_budget=15,
                threads=4, output_dir=None, smoke=False, controls=True, device="cuda"):
    """Freeze one validation-selected initializer, then compare paired refinements.

    Child processes receive training arrays only. AdamW is selected using its
    validation MSE; its exact exported model is shared by every refit and warm
    fit. Each subsequent track selects its own penalty on validation and scores
    only that selected model on test. Costs include all attempted configurations
    and explicitly identify the shared initializer search.
    """
    if name not in DATASETS or device not in {"cpu", "cuda"}:
        raise ValueError("unknown dataset or device")
    if width < 2 or max_cols < width or not np.isfinite(time_budget) or time_budget <= 0 or threads < 1:
        raise ValueError("invalid width, column limit, time budget, or threads")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be nonempty, distinct, and nonnegative")
    folder = Path(output_dir or tempfile.mkdtemp(prefix="relu-hybrid-")).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    X, y, dataset_metadata = load_dataset(name)
    records, selected, paired, splits = [], [], [], []
    lambdas = (0.001,) if smoke else LAMBDAS
    tracks = TRACKS if controls else TRACKS[:3]

    for seed in seeds:
        split = prepare_split(X, y, seed)
        splits.append(split["metadata"])
        split_dir = folder / f"split-{seed}"
        split_dir.mkdir(exist_ok=True)
        training_path = split_dir / "data.npz"
        np.savez(training_path, X_train=split["X_train"], y_train=split["y_train"])
        candidates = {track: [] for track in tracks}

        def attempt(track, index, spec, input_path):
            trial_dir = split_dir / track / str(index)
            result = _trial(input_path, spec, trial_dir)
            row = dict(dataset=name, split_seed=seed, solver=track, config_index=index,
                       configuration=spec, **result)
            started = time.perf_counter()
            if row["status"] == "ok":
                try:
                    with np.load(trial_dir / "result.npz") as model:
                        U, beta = model["U"], model["beta"]
                    with threadpool_limits(limits=threads):
                        train = scores(split["y_train"], predict(split["X_train"], U, beta),
                                       split["transforms"]["y_scale"])
                        validation = scores(split["y_val"], predict(split["X_val"], U, beta),
                                            split["transforms"]["y_scale"])
                        n_chambers = chamber_count(split["X_train"], U)
                    path_norm = float(np.abs(beta).sum())
                    objective = 0.5 * train["mse"] + spec["lambda_mean"] * path_norm
                    if not np.isfinite(objective):
                        raise ValueError("nonfinite training objective")
                    row.update(train=train, validation=validation, path_norm=path_norm,
                               train_objective=objective, n_units=len(beta),
                               n_chambers=n_chambers, model_sha256=model_digest(U, beta))
                except (OSError, ValueError, KeyError) as exc:
                    row.update(status="invalid", error=str(exc))
                else:
                    candidates[track].append((row, trial_dir / "result.npz"))
                    print(f"{name} split={seed} {track} {index}: {row['seconds']:.3f}s "
                          f"val_rmse={validation['rmse_original_units']:.6g}", flush=True)
            if row["status"] != "ok":
                print(f"{name} split={seed} {track} {index}: {row['status']} {row.get('error', '')}", flush=True)
            row["scoring_seconds"] = time.perf_counter() - started
            row["trial_seconds"] = row["process_seconds"] + row["scoring_seconds"]
            records.append(row)

        def adamw_specs(budget):
            return [dict(spec, operation="adamw", training_device=device,
                         config=dict(spec["config"], max_epochs=1_000_000))
                    for spec in configurations("adamw", width, budget, seed, threads, smoke)]

        for index, spec in enumerate(adamw_specs(time_budget)):
            attempt("adamw", index, spec, training_path)
        initializer = min(candidates["adamw"], key=lambda item: item[0]["validation"]["mse"], default=None)
        init_row = initializer[0] if initializer else None
        shared_rows = [r for r in records if r["split_seed"] == seed and r["solver"] == "adamw"]
        shared_cost = sum(r["trial_seconds"] for r in shared_rows)
        warm_path = split_dir / "initialized-data.npz"
        if initializer:
            with np.load(initializer[1]) as model:
                np.savez(warm_path, X_train=split["X_train"], y_train=split["y_train"],
                         initial_U=model["U"], initial_beta=model["beta"])

        # Rotate warm/cold ordering while sharing the same frozen parent model.
        operations = [track for track in tracks if track not in {"adamw", "adamw-long"}]
        if seed % 2:
            operations.reverse()
        for index, lam in enumerate(lambdas):
            for track in operations if index % 2 == 0 else list(reversed(operations)):
                cold = track.startswith("cold-rcg")
                if not cold and not initializer:
                    continue
                spec = dict(operation=track, lambda_mean=lam, max_cols=max_cols,
                            width=width, seed=seed, threads=threads, device=device,
                            time_budget=time_budget * (2 if track == "cold-rcg-long" else 1))
                if not cold:
                    spec["initializer_sha256"] = init_row["model_sha256"]
                    spec["initializer_configuration"] = init_row["configuration"]
                attempt(track, index, spec, training_path if cold else warm_path)
        if controls:
            for index, spec in enumerate(adamw_specs(2 * time_budget)):
                attempt("adamw-long", index, spec, training_path)

        for track in tracks:
            all_rows = [r for r in records if r["split_seed"] == seed and r["solver"] == track]
            if not candidates[track]:
                selected.append(dict(dataset=name, split_seed=seed, solver=track, status="failed",
                                     failed_trials=len(all_rows),
                                     error="initializer unavailable" if track in {"warm-rcg", "coefficient-refit"} and not initializer else "no successful fits"))
                continue
            best, model_path = min(candidates[track], key=lambda item: item[0]["validation"]["mse"])
            with np.load(model_path) as model:
                with threadpool_limits(limits=threads):
                    test = scores(split["y_test"], predict(split["X_test"], model["U"], model["beta"]),
                                  split["transforms"]["y_scale"])
            is_child = track in {"warm-rcg", "coefficient-refit"}
            own_cost = sum(r["trial_seconds"] for r in all_rows)
            pipeline_rows = (best, init_row) if is_child else (best,)
            selected.append(dict(
                dataset=name, split_seed=seed, solver=track, status="ok", test=test,
                **{key: best[key] for key in ("train", "validation", "configuration", "seconds",
                    "train_objective", "path_norm", "n_units", "n_chambers", "fit_info", "model_sha256",
                    "process_peak_rss_bytes", "gpu_peak_bytes")},
                selected_fit_seconds=best["seconds"] + (init_row["seconds"] if is_child else 0),
                initializer_fit_seconds=init_row["seconds"] if is_child else 0,
                initializer_sha256=init_row["model_sha256"] if is_child else None,
                initializer_trials=len(shared_rows) if is_child else 0,
                failed_initializer_trials=sum(r["status"] != "ok" for r in shared_rows) if is_child else 0,
                trials=len(all_rows), failed_trials=sum(r["status"] != "ok" for r in all_rows),
                stage_successful_fit_seconds=sum(r.get("seconds", 0) for r in all_rows),
                shared_initializer_successful_fit_seconds=sum(r.get("seconds", 0) for r in shared_rows) if is_child else 0,
                stage_tuning_seconds=own_cost,
                shared_initializer_tuning_seconds=shared_cost if is_child else 0,
                tuning_seconds=own_cost + (shared_cost if is_child else 0),
                pipeline_peak_rss_bytes=max(r["process_peak_rss_bytes"] for r in pipeline_rows),
                pipeline_gpu_peak_bytes=max((r["gpu_peak_bytes"] for r in pipeline_rows
                                             if r["gpu_peak_bytes"] is not None), default=None),
            ))
        if initializer:
            for lam in lambdas:
                row = dict(dataset=name, split_seed=seed, lambda_mean=lam,
                           initializer_sha256=init_row["model_sha256"],
                           initial_train_objective=0.5 * init_row["train"]["mse"] + lam * init_row["path_norm"],
                           initial_train=init_row["train"], initial_path_norm=init_row["path_norm"],
                           initial_n_units=init_row["n_units"], initial_n_chambers=init_row["n_chambers"])
                for track in ("coefficient-refit", "warm-rcg", "cold-rcg", "cold-rcg-long"):
                    match = next((r for r, _ in candidates.get(track, [])
                                  if r["configuration"]["lambda_mean"] == lam), None)
                    row[track] = {key: match[key] for key in (
                        "train_objective", "path_norm", "n_units", "n_chambers", "seconds", "train", "fit_info",
                    )} if match else None
                paired.append(row)
    return jsonable(dict(
        schema_version=1, dataset=name, dataset_metadata=dataset_metadata, splits=splits,
        source_sha256=source_digest(), benchmark_sha256=source_digest(Path(__file__).parent),
        environment=environment(threads),
        protocol=dict(split="approximately_60_20_20_by_raw_input_groups", seeds=list(seeds),
            width=width, max_cols=max_cols, time_budget=time_budget,
            control_time_budget=2 * time_budget, threads=threads, device=device,
            tracks=list(tracks), lambdas=list(lambdas), controls=controls, smoke=smoke,
            initializer_selection="minimum validation MSE among AdamW configurations; frozen across penalties",
            selection="minimum validation MSE per track; test scored after selection",
            timing="shared initializer search counted once per child track; all attempted fit processes and validation scoring included"),
        selected=selected, paired=paired, records=records,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="concrete")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-cols", type=int, default=512)
    parser.add_argument("--time-budget", type=float, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/hybrid.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_dataset(args.dataset, tuple(map(int, args.seeds.split(","))), args.width,
                         args.max_cols, args.time_budget, args.threads,
                         args.output.parent / (args.output.stem + "-trials"), args.smoke,
                         not args.no_controls, args.device)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
