"""Print real-data comparison tables and optionally save the compact raw schema.

Example: python -m benchmarks.report_real runs/real-comparison.json \
    --compact-output benchmarks/real-results.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = ("concrete", "wine-white", "california", "superconductivity")
DEFAULT_SOLVERS = ("relu-cpu", "relu-cuda", "scnn-cuda", "adelie", "sgd", "adam", "adamw")
DEFAULT_SEEDS = (0, 1, 2)
FIXED_LAMBDA = 0.001


def compact_results(value):
    """Preserve the input schema while removing every stored trajectory list."""
    if isinstance(value, dict):
        return {key: compact_results(item) for key, item in value.items()
                if not (key == "trajectory" and isinstance(item, list))}
    if isinstance(value, list):
        return [compact_results(item) for item in value]
    return value


def _number(value, name, *, nullable=False, nonnegative=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _stats(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def _choices(values, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and distinct")
    return values


def aggregate_results(payload, *, datasets=DEFAULT_DATASETS, solvers=DEFAULT_SOLVERS,
                      seeds=DEFAULT_SEEDS):
    """Aggregate explicit coverage, retaining counts for every displayed cell.

    Successful selected models, failed selections, and absent selections stay
    distinct. Fixed-penalty coverage is checked independently of predictive
    selection, since all trials at the fixed lambda can fail even when another
    lambda supplies a selected model.
    """
    datasets, solvers, seeds = (_choices(values, name) for values, name in (
        (datasets, "datasets"), (solvers, "solvers"), (seeds, "seeds")))
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if payload.get("schema_version") != 1:
        raise ValueError("expected real-data schema_version 1")
    experiments = payload.get("experiments", [payload])
    if not isinstance(experiments, list):
        raise ValueError("experiments must be a list")
    expected = set(itertools.product(datasets, solvers, seeds))
    selected, fixed = {}, {}
    records = defaultdict(list)
    issues = []
    fingerprints = defaultdict(set)

    def key_for(row, dataset):
        if row.get("dataset") != dataset:
            raise ValueError("row dataset does not match its experiment")
        key = (dataset, row.get("solver"), row.get("split_seed"))
        if isinstance(key[2], bool) or not isinstance(key[2], int):
            raise ValueError("split_seed must be an integer")
        return key

    for experiment in experiments:
        if experiment.get("schema_version") != 1:
            raise ValueError("expected experiment schema_version 1")
        dataset = experiment.get("dataset")
        if dataset not in datasets:
            continue
        included = False
        for row in experiment.get("selected", []):
            key = key_for(row, dataset)
            if key not in expected:
                continue
            included = True
            if key in selected:
                raise ValueError(f"duplicate selected result: {key}")
            selected[key] = row
        for row in experiment.get("fixed_penalty", []):
            key = key_for(row, dataset)
            if key not in expected or key[1] == "adamw":
                continue
            included = True
            if key in fixed:
                raise ValueError(f"duplicate fixed-penalty result: {key}")
            lam = row.get("configuration", {}).get("lambda_mean")
            if lam is None or not math.isclose(float(lam), FIXED_LAMBDA, rel_tol=0, abs_tol=1e-15):
                raise ValueError(f"fixed-penalty row has lambda != {FIXED_LAMBDA}: {key}")
            fixed[key] = row
        for row in experiment.get("records", []):
            key = key_for(row, dataset)
            if key in expected:
                included = True
                records[key].append(row)
        if included:
            protocol = experiment.get("protocol", {})
            for name in ("split", "width", "time_budget", "threads", "lambdas", "smoke"):
                if name in protocol:
                    fingerprints[f"protocol.{name}"].add(json.dumps(protocol[name], sort_keys=True))
            for name in ("source_sha256", "benchmark_sha256"):
                if name in experiment:
                    fingerprints[name].add(experiment[name])

    for name, values in fingerprints.items():
        if len(values) > 1:
            issues.append(f"mixed {name} across experiments")
    missing = sorted(expected - set(selected))
    failed = sorted(key for key, row in selected.items() if row.get("status") != "ok")
    missing_fixed = sorted(key for key in expected if key[1] != "adamw" and key not in fixed)
    rows = []
    for dataset, solver in itertools.product(datasets, solvers):
        successful = []
        stops = Counter()
        trial_count = failed_trials = 0
        for seed in seeds:
            key = dataset, solver, seed
            row = selected.get(key)
            trials = records[key]
            trial_count += len(trials)
            failures = sum(trial.get("status") != "ok" for trial in trials)
            failed_trials += failures
            if row is None:
                stops["missing"] += 1
                continue
            for field, actual in (("trials", len(trials)), ("failed_trials", failures)):
                if field in row and row[field] != actual:
                    issues.append(f"{dataset}/{solver}/seed-{seed}: {field} disagrees with raw records")
            if row.get("status") != "ok":
                stops[f"selection_{row.get('status', 'unknown')}"] += 1
                continue
            test = row.get("test", {})
            prefix = f"{dataset}/{solver}/seed-{seed}"
            successful.append(dict(
                rmse=_number(test.get("rmse_original_units"), prefix + " RMSE", nonnegative=True),
                r2=_number(test.get("r2"), prefix + " R2", nullable=True),
                fit=_number(row.get("seconds"), prefix + " fit seconds", nonnegative=True),
                tuning=_number(row.get("tuning_seconds"), prefix + " tuning seconds", nonnegative=True),
                rss=_number(row.get("process_peak_rss_bytes"), prefix + " process RSS", nonnegative=True),
                gpu=_number(row.get("gpu_peak_bytes"), prefix + " GPU allocated bytes", nullable=True,
                            nonnegative=True),
            ))
            info = row.get("fit_info", {})
            stop = next((info[name] for name in ("stop_reason", "stopping_reason", "status")
                         if info.get(name) is not None), "unspecified")
            stops[str(stop)] += 1
        fixed_values = [_number(fixed[dataset, solver, seed].get("train_objective"),
                                 f"{dataset}/{solver}/seed-{seed} fixed objective", nonnegative=True)
                        for seed in seeds if (dataset, solver, seed) in fixed]
        rows.append(dict(
            dataset=dataset, solver=solver, expected=len(seeds), successful=len(successful),
            metrics={name: _stats(item[name] for item in successful)
                     for name in ("rmse", "r2", "fit", "tuning", "rss", "gpu")},
            fixed=_stats(fixed_values), stop_counts=dict(sorted(stops.items())),
            failed_trials=failed_trials, trials=trial_count,
        ))
    return dict(rows=rows, expected=len(expected), successful=len(selected) - len(failed),
                missing=missing, failed=failed, missing_fixed=missing_fixed, issues=issues,
                complete=not (missing or failed or missing_fixed or issues))


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def _cell(stat, expected, *, mean_sd=False, mean_only=False, scale=1):
    n = stat["n"]
    if not n:
        return f"unavailable [0/{expected}]"
    value = stat["mean"] if mean_sd or mean_only else stat["median"]
    text = f"{value / scale:.5g}"
    if mean_sd:
        sd = "n/a" if stat["sd"] is None else f"{stat['sd'] / scale:.3g}"
        text += f" ± {sd}"
    return text + f" [{n}/{expected}{'; PARTIAL' if n != expected else ''}]"


def format_report(summary):
    """Return Markdown tables with explicit per-cell and selection coverage."""
    lines = [
        f"Selected fits: {summary['successful']}/{summary['expected']} successful; "
        f"{len(summary['failed'])} failed; {len(summary['missing'])} missing.",
        "Every cell shows observed/expected seed counts. SD is the sample SD; it is unavailable for one seed.",
    ]
    for label, keys in (("Missing selections", summary["missing"]),
                        ("Failed selections", summary["failed"]),
                        ("Missing fixed-penalty fits", summary["missing_fixed"])):
        if keys:
            lines.append(label + ": " + "; ".join(f"{d}/{s}/seed-{seed}" for d, s, seed in keys) + ".")
    lines.extend("Issue: " + issue for issue in summary["issues"])
    rows = summary["rows"]
    lines.extend(["", "Test accuracy (original target units)", "", _table(
        ("Dataset", "Method", "RMSE mean ± SD", "R² mean ± SD"),
        [(row["dataset"], row["solver"],
          _cell(row["metrics"]["rmse"], row["expected"], mean_sd=True),
          _cell(row["metrics"]["r2"], row["expected"], mean_sd=True)) for row in rows]),
        "", "Runtime and memory (medians across selected fits)", "", _table(
            ("Dataset", "Method", "Fit s", "All-trial tuning s", "Process RSS MiB", "Torch GPU allocated MiB"),
            [(row["dataset"], row["solver"],
              *(_cell(row["metrics"][name], row["expected"], scale=scale)
                for name, scale in (("fit", 1), ("tuning", 1), ("rss", 1024**2), ("gpu", 1024**2))))
             for row in rows]),
        "Tuning time includes successful and failed processes plus training/validation scoring. "
        "GPU memory is Torch allocation, not physical VRAM; unavailable also covers CPU fits.",
        "", "Fixed λ = 0.001 (mean canonical training objective)", "", _table(
            ("Dataset", "Method", "Train objective mean"),
            [(row["dataset"], row["solver"], _cell(row["fixed"], row["expected"], mean_only=True))
             for row in rows if row["solver"] != "adamw"]),
        "AdamW uses decoupled decay and is excluded from the fixed-penalty comparison.",
        "", "Selected stop reasons and all-trial failures", "", _table(
            ("Dataset", "Method", "Selected stop counts", "Failed / attempted trials"),
            [(row["dataset"], row["solver"],
              "; ".join(f"{name}: {count}" for name, count in row["stop_counts"].items()),
              f"{row['failed_trials']} / {row['trials']}") for row in rows]),
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--compact-output", type=Path)
    args = parser.parse_args(argv)
    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    summary = aggregate_results(payload, datasets=args.datasets.split(","),
                                solvers=args.solvers.split(","),
                                seeds=tuple(map(int, args.seeds.split(","))))
    print(format_report(summary), end="")
    if args.compact_output is not None:
        if args.compact_output.resolve() == args.input.resolve():
            raise ValueError("compact output must differ from the raw input")
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.compact_output.suffix == ".gz" else open
        with opener(args.compact_output, "wt", encoding="utf-8") as stream:
            json.dump(compact_results(payload), stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
