"""Run real-data solver comparisons on a shared CPU/L4 worker configuration."""
from pathlib import Path
import hashlib
import io
import json
import re
import shutil
import tempfile
import uuid

import modal


ROOT = Path(__file__).resolve().parents[1]
# Freeze sources before upload so ongoing local edits cannot alter a run.
_snapshot = tempfile.TemporaryDirectory(prefix="relu-real-source-")
SOURCE = Path(_snapshot.name)
for folder in ("reluopt", "benchmarks", "tests"):
    shutil.copytree(ROOT / folder, SOURCE / folder,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
if (ROOT / "runs/data").exists():
    shutil.copytree(ROOT / "runs/data", SOURCE / "data")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgomp1")
         .pip_install(*(SOURCE / "benchmarks/requirements-real.txt").read_text().splitlines())
         .env({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
               "PYTHONHASHSEED": "0", "RELUOPT_DATA_CACHE": "/tmp/benchmark-data"}))
for folder in ("reluopt", "benchmarks", "tests"):
    image = image.add_local_dir(SOURCE / folder, "/root/" + folder,
                                ignore=["**/__pycache__/**", "**/*.pyc"])
if (SOURCE / "data").exists():
    image = image.add_local_dir(SOURCE / "data", "/opt/benchmark-data")
app = modal.App("reluopt-real-data", include_source=False)
results_volume = modal.Volume.from_name("reluopt-benchmarks", create_if_missing=True)


def _read_saved(run_id, filename):
    return json.loads(b"".join(results_volume.read_file(f"/{run_id}/{filename}")))


def _validate_result(result, manifest):
    """Reject results from a different source snapshot or requested protocol."""
    for key in ("source_sha256", "benchmark_sha256"):
        if result[key] != manifest[key]:
            raise ValueError(f"saved result has a different {key}")
    protocol = result["protocol"]
    expected = manifest["protocol"]
    for key in ("solvers", "width", "time_budget", "smoke"):
        if protocol[key] != expected[key]:
            raise ValueError(f"saved result has a different {key}")
    if (result["dataset"] not in expected["datasets"]
            or len(protocol["seeds"]) != 1 or protocol["seeds"][0] not in expected["seeds"]):
        raise ValueError("saved result is not a requested dataset/split")


def _read_unit_result(run_id, name, seed, manifest):
    result = _read_saved(run_id, f"{name}-{seed}.json")
    _validate_result(result, manifest)
    if result["dataset"] != name or result["protocol"]["seeds"] != [seed]:
        raise ValueError("saved result filename does not match its dataset/split")
    return result


def _save_local(destination, manifest, results):
    expected = [(name, seed) for name in manifest["protocol"]["datasets"]
                for seed in manifest["protocol"]["seeds"]]
    completed = {(result["dataset"], result["protocol"]["seeds"][0]) for result in results}
    missing = [dict(dataset=name, seed=seed) for name, seed in expected
               if (name, seed) not in completed]
    destination.parent.mkdir(parents=True, exist_ok=True)
    for result in results:
        seed = result["protocol"]["seeds"][0]
        part = destination.with_name(f"{destination.stem}-{result['dataset']}-{seed}.json")
        part.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    combined = dict(schema_version=1, run_id=manifest["run_id"], manifest=manifest,
                    complete=not missing, missing=missing, experiments=results)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return missing


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=600,
              max_containers=1, retries=0, scaledown_window=2)
def check():
    import os
    import subprocess
    import sys

    os.environ["RELUOPT_TEST_ACCELERATOR"] = "1"
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=True, timeout=300)
    subprocess.run([sys.executable, "-m", "tests.test_data_engine"], check=True, timeout=60)
    return "Tests passed"


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=3600,
              max_containers=6, retries=0, scaledown_window=2,
              volumes={"/results": results_volume})
def unit(dataset, seed, solvers, width, budget, smoke, manifest):
    import shutil
    import subprocess
    import sys

    if Path("/opt/benchmark-data").exists():
        shutil.copytree("/opt/benchmark-data", "/tmp/benchmark-data", dirs_exist_ok=True)
    command = [sys.executable, "-m", "benchmarks.real", "--dataset", dataset,
               "--seeds", str(seed), "--solvers", solvers, "--width", str(width),
               "--time-budget", str(budget), "--output", "/tmp/real-result.json"]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, timeout=3450)
    result = json.loads(Path("/tmp/real-result.json").read_text())
    _validate_result(result, manifest)
    directory = Path("/results") / manifest["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset}-{seed}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    results_volume.commit()
    return dict(dataset=dataset, seed=seed, fits=len(result["records"]))


@app.function(image=image, cpu=0.25, memory=512, timeout=10800,
              max_containers=1, retries=0, scaledown_window=2)
def coordinate(jobs):
    """Keep all child calls owned by one detached server-side invocation."""
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
def main(dataset: str = "all", seeds: str = "0,1,2", solvers: str = "all",
         width: int = 256, budget: float = 15, smoke: bool = False,
         output: str = "runs/real-comparison.json", checks: bool = True,
         run_id: str = "", collect_only: bool = False):
    if collect_only and not run_id:
        raise ValueError("--collect-only requires --run-id")
    run_id = run_id or uuid.uuid4().hex[:16]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run IDs must contain 1–80 letters, digits, underscores, or hyphens")
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
                    result = _read_unit_result(run_id, name, seed, saved_manifest)
                except FileNotFoundError:
                    continue
                results.append(result)
        missing = _save_local(destination, saved_manifest, results)
        print(f"Collected {len(results)} dataset/splits from {run_id}; {len(missing)} missing: {destination}")
        return
    if saved_manifest is not None:
        raise ValueError(f"run {run_id!r} already exists; use --collect-only or choose a fresh --run-id")

    from benchmarks.real import DATASETS, SOLVERS
    from benchmarks.run import source_digest

    names = list(DATASETS) if dataset == "all" else dataset.split(",")
    methods = list(SOLVERS) if solvers == "all" else solvers.split(",")
    seed_list = list(map(int, seeds.split(",")))
    if not names or any(n not in DATASETS for n in names) or not methods or any(s not in SOLVERS for s in methods):
        raise ValueError("unknown dataset or solver")
    if (len(set(names)) != len(names) or len(set(methods)) != len(methods)
            or len(set(seed_list)) != len(seed_list) or not seed_list):
        raise ValueError("datasets, solvers, and seeds must be nonempty and distinct")
    if width < 2 or not 0 < budget <= 120 or len(seed_list) > 5 or any(s < 0 for s in seed_list):
        raise ValueError("width >= 2, budget in (0,120], and at most five nonnegative seeds required")
    manifest = dict(
        schema_version=1, run_id=run_id,
        source_sha256=source_digest(SOURCE / "reluopt"),
        benchmark_sha256=source_digest(SOURCE / "benchmarks"),
        requirements_sha256=hashlib.sha256((SOURCE / "benchmarks/requirements-real.txt").read_bytes()).hexdigest(),
        worker=dict(gpu="L4", cpu=4, memory_mib=32768, max_containers=6),
        protocol=dict(datasets=names, seeds=seed_list, solvers=methods, width=width,
                      time_budget=budget, smoke=smoke),
    )
    with results_volume.batch_upload() as batch:
        batch.put_file(io.BytesIO((json.dumps(manifest, indent=2) + "\n").encode()),
                       f"/{run_id}/manifest.json")
    print(f"Modal run ID: {run_id} (saved in volume reluopt-benchmarks)", flush=True)
    print(f"Recover with --collect-only --run-id {run_id} --output {output}", flush=True)
    if checks:
        print(check.remote())
    results = []
    _save_local(destination, manifest, results)
    jobs = [(name, seed, ",".join(methods), width, budget, smoke, manifest)
            for name in names for seed in seed_list]
    # Detached local entrypoints retain only their last triggered call. A single
    # remote coordinator owns every unit, so a lost client cannot orphan them.
    for acknowledgment in coordinate.remote(jobs):
        result = _read_unit_result(run_id, acknowledgment["dataset"], acknowledgment["seed"], manifest)
        results.append(result)
        _save_local(destination, manifest, results)
        seed = result["protocol"]["seeds"][0]
        print(f"Saved {result['dataset']} split {seed}: {len(result['records'])} fits")
    missing = _save_local(destination, manifest, results)
    if missing:
        raise RuntimeError(f"run {run_id} is incomplete: {len(missing)} dataset/splits missing; saved partial results")
