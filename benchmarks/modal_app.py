"""Run the current synthetic benchmark on Modal CPU, L4, or A100 workers."""

from pathlib import Path
import json
import os
import subprocess

import modal


ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "threadpoolctl==3.7.0", "pytest==9.1.1")
    .env({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"})
)
gpu_image = image.pip_install("torch==2.10.0")
for folder in ("reluopt", "benchmarks", "tests"):
    image = image.add_local_dir(ROOT / folder, "/root/" + folder, ignore=["**/__pycache__/**", "**/*.pyc"])
    gpu_image = gpu_image.add_local_dir(ROOT / folder, "/root/" + folder, ignore=["**/__pycache__/**", "**/*.pyc"])
app = modal.App("reluopt-benchmarks")


def execute(suite, variants, backends, repeats, seed, profile, threads):
    import sys

    os.environ["RELUOPT_TEST_ACCELERATOR"] = "1"
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=True, timeout=180)
    # Keep the solver and profiler independent of Modal's event-loop hooks.
    command = [
        sys.executable, "-m", "benchmarks.run", "--suite", suite,
        "--variants", ",".join(variants), "--backends", ",".join(backends),
        "--repeats", str(repeats), "--seed", str(seed), "--threads", str(threads),
        "--output", "/tmp/benchmark.json",
    ]
    if profile:
        command.append("--profile")
    subprocess.run(command, check=True, timeout=2250)
    return Path("/tmp/benchmark.json").read_text()


@app.function(image=image, cpu=4, memory=16384, timeout=2400,
              max_containers=1, retries=0, scaledown_window=2)
def cpu(suite, variants, repeats, seed, profile, threads):
    return execute(suite, variants, ["cpu"], repeats, seed, profile, threads)


@app.function(image=gpu_image, gpu="L4", cpu=4, memory=16384, timeout=2400,
              max_containers=1, retries=0, scaledown_window=2)
def gpu(suite, variants, backends, repeats, seed, profile, threads):
    return execute(suite, variants, backends, repeats, seed, profile, threads)


@app.function(image=gpu_image, gpu="A100-40GB", cpu=4, memory=16384, timeout=2400,
              max_containers=1, retries=0, scaledown_window=2)
def a100(suite, variants, backends, repeats, seed, profile, threads):
    return execute(suite, variants, backends, repeats, seed, profile, threads)


@app.local_entrypoint()
def main(suite: str = "smoke", variants: str = "default", backends: str = "cpu",
         repeats: int = 3, seed: int = 177, profile_run: bool = False,
         output: str = "runs/modal.json", gpu_type: str = "L4", threads: int = 4):
    from benchmarks.run import BACKENDS, CUDA_VARIANTS, VARIANTS, names_for_suite, parse_choices

    names_for_suite(suite)
    variant_list = parse_choices(variants, VARIANTS, "variants")
    backend_list = parse_choices(backends, BACKENDS, "backends")
    if not 1 <= repeats <= 5 or threads < 1:
        raise ValueError("repeats must be 1..5; threads must be positive")
    if all(variant in CUDA_VARIANTS for variant in variant_list) and "cuda" not in backend_list:
        raise ValueError("device, device64, and graph variants require --backends cuda")
    if "cuda" in backend_list:
        if gpu_type not in {"L4", "A100"}:
            raise ValueError("gpu-type must be L4 or A100")
        function = gpu if gpu_type == "L4" else a100
        raw = function.remote(suite, variant_list, backend_list, repeats, seed, profile_run, threads)
    else:
        raw = cpu.remote(suite, variant_list, repeats, seed, profile_run, threads)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json.loads(raw), indent=2, allow_nan=False) + "\n")
    print(f"Saved {destination}")
