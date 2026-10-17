"""Paired AdamW initialization experiments on persistent Modal L4 workers."""
import hashlib
import io
import json
from pathlib import Path
import re
import uuid

import modal

from benchmarks.modal_real import SOURCE, image, results_volume, _read_saved, _save_local


app = modal.App("reluopt-adamw-initialization", include_source=False)


def _validate_result(result, manifest):
    for key in ("source_sha256", "benchmark_sha256"):
        if result[key] != manifest[key]:
            raise ValueError(f"saved result has a different {key}")
    expected, actual = manifest["protocol"], result["protocol"]
    for key in ("width", "max_cols", "time_budget", "smoke", "controls", "device"):
        if actual[key] != expected[key]:
            raise ValueError(f"saved result has a different {key}")
    if (result["dataset"] not in expected["datasets"] or len(actual["seeds"]) != 1
            or actual["seeds"][0] not in expected["seeds"]):
        raise ValueError("saved result is not a requested dataset/split")


def _read_unit(run_id, name, seed, manifest):
    result = _read_saved(run_id, f"{name}-{seed}.json")
    _validate_result(result, manifest)
    if result["dataset"] != name or result["protocol"]["seeds"] != [seed]:
        raise ValueError("saved result filename does not match its dataset/split")
    return result


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=600,
              max_containers=1, retries=0, scaledown_window=2)
def check():
    import os
    import subprocess
    import sys

    os.environ["RELUOPT_TEST_ACCELERATOR"] = "1"
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=True, timeout=400)
    subprocess.run([sys.executable, "-m", "tests.test_data_engine"], check=True, timeout=60)
    return "Tests passed"


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=5400,
              max_containers=6, retries=0, scaledown_window=2,
              volumes={"/results": results_volume})
def unit(dataset, seed, width, max_cols, budget, smoke, controls, manifest):
    import shutil
    import subprocess
    import sys
    import threading

    directory = Path("/results") / manifest["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset}-{seed}.json"
    if path.exists():
        result = json.loads(path.read_text())
        _validate_result(result, manifest)
        if result["dataset"] != dataset or result["protocol"]["seeds"] != [seed]:
            raise ValueError("saved result filename does not match its dataset/split")
        return dict(dataset=dataset, seed=seed, fits=len(result["records"]))
    work_output = directory / f"{dataset}-{seed}-work.json"

    if Path("/opt/benchmark-data").exists():
        shutil.copytree("/opt/benchmark-data", "/tmp/benchmark-data", dirs_exist_ok=True)
    command = [sys.executable, "-m", "benchmarks.hybrid", "--dataset", dataset,
               "--seeds", str(seed), "--width", str(width), "--max-cols", str(max_cols),
               "--time-budget", str(budget), "--output", str(work_output)]
    if smoke:
        command.append("--smoke")
    if not controls:
        command.append("--no-controls")
    # Each progress line follows a completed fit and its atomic checkpoint.
    # Commit those files before another preemption can discard the whole split.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    deadline = threading.Timer(5250, process.kill)
    deadline.start()
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            results_volume.commit()
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        deadline.cancel()
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    result = json.loads(work_output.read_text())
    _validate_result(result, manifest)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    results_volume.commit()
    return dict(dataset=dataset, seed=seed, fits=len(result["records"]))


@app.function(image=image, cpu=0.25, memory=512, timeout=14400,
              max_containers=1, retries=0, scaledown_window=2)
def coordinate(jobs):
    calls = [(job[0], job[1], unit.spawn(*job)) for job in jobs]
    acknowledgments = []
    for name, seed, call in calls:
        try:
            acknowledgment = call.get()
        except Exception as exc:
            print(f"Failed {name} split {seed}: {type(exc).__name__}: {exc}", flush=True)
            continue
        acknowledgments.append(acknowledgment)
        print(f"Saved {name} split {seed}: {acknowledgment['fits']} fits", flush=True)
    return acknowledgments


@app.local_entrypoint()
def main(dataset: str = "all", seeds: str = "0,1,2", width: int = 256,
         max_cols: int = 512, budget: float = 15, smoke: bool = False,
         controls: bool = True, output: str = "runs/adamw-initialization.json",
         checks: bool = True, run_id: str = "", collect_only: bool = False):
    if collect_only and not run_id:
        raise ValueError("--collect-only requires --run-id")
    run_id = run_id or uuid.uuid4().hex[:16]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("invalid run ID")
    destination = Path(output)
    try:
        saved_manifest = _read_saved(run_id, "manifest.json")
    except FileNotFoundError:
        saved_manifest = None
    if collect_only:
        if saved_manifest is None:
            raise ValueError(f"no saved run {run_id!r}")
        results = []
        for name in saved_manifest["protocol"]["datasets"]:
            for seed in saved_manifest["protocol"]["seeds"]:
                try:
                    results.append(_read_unit(run_id, name, seed, saved_manifest))
                except FileNotFoundError:
                    continue
        missing = _save_local(destination, saved_manifest, results)
        print(f"Collected {len(results)} dataset/splits; {len(missing)} missing: {destination}")
        return
    if saved_manifest is not None:
        raise ValueError("run already exists; collect it or choose a fresh run ID")

    from benchmarks.datasets import DATASETS
    from benchmarks.run import source_digest

    names = list(DATASETS) if dataset == "all" else dataset.split(",")
    seed_list = list(map(int, seeds.split(",")))
    if (not names or any(n not in DATASETS for n in names)
            or len(set(names)) != len(names) or not seed_list
            or len(set(seed_list)) != len(seed_list)):
        raise ValueError("datasets and seeds must be nonempty, valid, and distinct")
    if (width < 2 or max_cols < width or not 0 < budget <= 120
            or len(seed_list) > 5 or any(s < 0 for s in seed_list)):
        raise ValueError("require max_cols >= width >= 2, budget in (0,120], <=5 nonnegative seeds")
    manifest = dict(
        schema_version=1, run_id=run_id,
        source_sha256=source_digest(SOURCE / "reluopt"),
        benchmark_sha256=source_digest(SOURCE / "benchmarks"),
        requirements_sha256=hashlib.sha256((SOURCE / "benchmarks/requirements-real.txt").read_bytes()).hexdigest(),
        worker=dict(gpu="L4", cpu=4, memory_mib=32768, max_containers=6),
        protocol=dict(datasets=names, seeds=seed_list, width=width, max_cols=max_cols,
                      time_budget=budget, smoke=smoke, controls=controls, device="cuda"),
    )
    with results_volume.batch_upload() as batch:
        batch.put_file(io.BytesIO((json.dumps(manifest, indent=2) + "\n").encode()),
                       f"/{run_id}/manifest.json")
    print(f"Modal run ID: {run_id}; recover with --collect-only --run-id {run_id}", flush=True)
    if checks:
        print(check.remote())
    results = []
    _save_local(destination, manifest, results)
    jobs = [(name, seed, width, max_cols, budget, smoke, controls, manifest)
            for name in names for seed in seed_list]
    # One detached remote call owns every child, even if the local client closes.
    for acknowledgment in coordinate.remote(jobs):
        results.append(_read_unit(run_id, acknowledgment["dataset"], acknowledgment["seed"], manifest))
        _save_local(destination, manifest, results)
    missing = _save_local(destination, manifest, results)
    if missing:
        raise RuntimeError(f"incomplete run: {len(missing)} dataset/splits missing")
