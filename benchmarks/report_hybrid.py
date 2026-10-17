"""Report paired AdamW initialization results, coverage, and shared compute costs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
import math
from pathlib import Path

from benchmarks.hybrid import TRACKS
from benchmarks.report_real import (
    DEFAULT_DATASETS, DEFAULT_SEEDS, FIXED_LAMBDA, _cell, _choices, _number,
    _stats, _table, compact_results,
)


PAIRED_STAGES = ("initial", "coefficient-refit", "warm-rcg", "cold-rcg", "cold-rcg-long")


def aggregate_results(payload, *, datasets=DEFAULT_DATASETS, tracks=TRACKS, seeds=DEFAULT_SEEDS):
    """Expose incomplete selections and fixed-penalty pairs before aggregation."""
    datasets, tracks, seeds = (_choices(value, name) for value, name in (
        (datasets, "datasets"), (tracks, "tracks"), (seeds, "seeds")))
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if payload.get("schema_version") != 1:
        raise ValueError("expected hybrid schema_version 1")
    experiments = payload.get("experiments", [payload])
    if not isinstance(experiments, list):
        raise ValueError("experiments must be a list")
    expected = set(itertools.product(datasets, tracks, seeds))
    selected, paired = {}, {}
    records, fingerprints = defaultdict(list), defaultdict(set)
    issues = []
    for experiment in experiments:
        if experiment.get("schema_version") != 1:
            raise ValueError("expected experiment schema_version 1")
        dataset = experiment.get("dataset")
        if dataset not in datasets:
            continue
        for field in ("selected", "records"):
            for row in experiment.get(field, []):
                if row.get("dataset") != dataset:
                    raise ValueError("row dataset does not match its experiment")
                seed = row.get("split_seed")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError("split_seed must be an integer")
                key = dataset, row.get("solver"), seed
                if key not in expected:
                    continue
                if field == "selected":
                    if key in selected:
                        raise ValueError(f"duplicate selected result: {key}")
                    selected[key] = row
                else:
                    records[key].append(row)
        for row in experiment.get("paired", []):
            if row.get("dataset") != dataset:
                raise ValueError("paired row dataset does not match its experiment")
            if not math.isclose(float(row.get("lambda_mean", -1)), FIXED_LAMBDA, rel_tol=0, abs_tol=1e-15):
                continue
            key = dataset, row.get("split_seed")
            if key[1] not in seeds:
                continue
            if key in paired:
                raise ValueError(f"duplicate paired result: {key}")
            paired[key] = row
        for name in ("split", "width", "max_cols", "time_budget", "threads", "lambdas", "smoke", "controls", "device"):
            if name in experiment.get("protocol", {}):
                fingerprints[f"protocol.{name}"].add(json.dumps(experiment["protocol"][name], sort_keys=True))
        for name in ("source_sha256", "benchmark_sha256"):
            if name in experiment:
                fingerprints[name].add(experiment[name])
    issues.extend(f"mixed {name} across experiments" for name, values in fingerprints.items() if len(values) > 1)
    missing = sorted(expected - set(selected))
    failed = sorted(key for key, row in selected.items() if row.get("status") != "ok")
    missing_paired = sorted(set(itertools.product(datasets, seeds)) - set(paired))
    rows = []
    for dataset, track in itertools.product(datasets, tracks):
        successful, stops = [], Counter()
        attempted = failures = 0
        for seed in seeds:
            key = dataset, track, seed
            raw = records[key]
            attempted += len(raw)
            failures += sum(row.get("status") != "ok" for row in raw)
            row = selected.get(key)
            if row is None or row.get("status") != "ok":
                stops["missing" if row is None else "selection_failed"] += 1
                continue
            prefix = f"{dataset}/{track}/seed-{seed}"
            if row.get("trials") != len(raw):
                issues.append(prefix + ": trials disagree with raw records")
            fields = dict(
                rmse=row.get("test", {}).get("rmse_original_units"),
                r2=row.get("test", {}).get("r2"),
                fit=row.get("selected_fit_seconds"), stage_fit=row.get("seconds"),
                initializer_fit=row.get("initializer_fit_seconds"),
                tuning=row.get("tuning_seconds"), stage_tuning=row.get("stage_tuning_seconds"),
                shared_tuning=row.get("shared_initializer_tuning_seconds"),
                rss=row.get("pipeline_peak_rss_bytes"), gpu=row.get("pipeline_gpu_peak_bytes"),
            )
            values = {name: _number(value, prefix + " " + name, nullable=name in {"r2", "gpu"},
                                    nonnegative=name != "r2") for name, value in fields.items()}
            for total, left, right in (("fit", "stage_fit", "initializer_fit"),
                                       ("tuning", "stage_tuning", "shared_tuning")):
                if not math.isclose(values[total], values[left] + values[right], rel_tol=1e-10, abs_tol=1e-9):
                    issues.append(prefix + f": {total} does not include its two components")
            successful.append(values)
            info = row.get("fit_info", {})
            stops[str(info.get("stop_reason", info.get("stopping_reason", "unspecified")))] += 1
        rows.append(dict(dataset=dataset, solver=track, expected=len(seeds), successful=len(successful),
                         metrics={name: _stats(row[name] for row in successful) for name in (
                             "rmse", "r2", "fit", "stage_fit", "initializer_fit", "tuning",
                             "stage_tuning", "shared_tuning", "rss", "gpu")},
                         stop_counts=dict(sorted(stops.items())), trials=attempted, failed_trials=failures))
    fixed_rows, improvements = [], []
    for dataset in datasets:
        stage_values = defaultdict(list)
        comparisons = defaultdict(list)
        for seed in seeds:
            pair = paired.get((dataset, seed))
            if pair is None:
                continue
            values = {"initial": dict(train_objective=pair.get("initial_train_objective"),
                                       path_norm=pair.get("initial_path_norm"),
                                       n_chambers=pair.get("initial_n_chambers"))}
            values.update({stage: pair.get(stage) for stage in PAIRED_STAGES[1:]})
            prefix = f"{dataset}/seed-{seed} paired"
            for stage, value in values.items():
                if value is None:
                    issues.append(prefix + ": missing " + stage)
                    continue
                stage_values[stage].append({field: _number(value.get(field), prefix + " " + stage + " " + field,
                                                          nonnegative=True)
                                            for field in ("train_objective", "path_norm", "n_chambers")})
            warm = values["warm-rcg"]
            if warm is not None:
                warm_value = _number(warm.get("train_objective"), prefix + " warm objective", nonnegative=True)
                for stage in PAIRED_STAGES:
                    reference = values[stage]
                    if stage == "warm-rcg" or reference is None:
                        continue
                    ref = _number(reference.get("train_objective"), prefix + " reference objective", nonnegative=True)
                    comparisons[stage].append((warm_value, ref))
        for stage in PAIRED_STAGES:
            fixed_rows.append(dict(dataset=dataset, stage=stage, expected=len(seeds),
                                   metrics={name: _stats(value[name] for value in stage_values[stage])
                                            for name in ("train_objective", "path_norm", "n_chambers")}))
        for stage, values in comparisons.items():
            improvements.append(dict(dataset=dataset, reference=stage, n=len(values), expected=len(seeds),
                                     improved=sum(warm < ref - 1e-10 * max(1., abs(ref)) for warm, ref in values),
                                     worsened=sum(warm > ref + 1e-10 * max(1., abs(ref)) for warm, ref in values),
                                     objective_difference=_stats(warm - ref for warm, ref in values)))
    return dict(rows=rows, fixed_rows=fixed_rows, improvements=improvements,
                expected=len(expected), successful=len(selected) - len(failed),
                missing=missing, failed=failed, missing_paired=missing_paired, issues=issues,
                complete=not (missing or failed or missing_paired or issues))


def format_report(summary):
    lines = [
        f"Selected fits: {summary['successful']}/{summary['expected']} successful; "
        f"{len(summary['failed'])} failed; {len(summary['missing'])} missing.",
        "Every cell shows observed/expected split counts. SD is the sample SD.",
    ]
    for label, keys in (("Missing selections", summary["missing"]), ("Failed selections", summary["failed"]),
                        ("Missing fixed-penalty pairs", summary["missing_paired"])):
        if keys:
            lines.append(label + ": " + "; ".join("/".join(map(str, key)) for key in keys) + ".")
    lines.extend("Issue: " + issue for issue in summary["issues"])
    rows = summary["rows"]
    lines.extend(["", "Test accuracy (original target units)", "", _table(
        ("Dataset", "Method", "RMSE mean ± SD", "R² mean ± SD"),
        [(row["dataset"], row["solver"],
          _cell(row["metrics"]["rmse"], row["expected"], mean_sd=True),
          _cell(row["metrics"]["r2"], row["expected"], mean_sd=True)) for row in rows]),
        "", "Runtime (medians, seconds)", "", _table(
            ("Dataset", "Method", "Selected pipeline", "Stage fit", "Initializer fit", "All-trial tuning"),
            [(row["dataset"], row["solver"], *(_cell(row["metrics"][name], row["expected"])
              for name in ("fit", "stage_fit", "initializer_fit", "tuning"))) for row in rows]),
        "The selected pipeline includes its selected AdamW initializer where applicable. "
        "All-trial tuning includes the full shared initializer grid once per refinement track, "
        "all attempted child configurations, process startup, and host training/validation scoring.",
        "", "Selected pipeline memory and stop reasons", "", _table(
            ("Dataset", "Method", "Peak RSS MiB", "Peak Torch GPU allocated MiB", "Stops", "Failed / attempted trials"),
            [(row["dataset"], row["solver"], _cell(row["metrics"]["rss"], row["expected"], scale=1024**2),
              _cell(row["metrics"]["gpu"], row["expected"], scale=1024**2),
              "; ".join(f"{name}: {count}" for name, count in row["stop_counts"].items()),
              f"{row['failed_trials']} / {row['trials']}") for row in rows]),
        "Pipeline peaks are the maximum of its separate fit processes, including the selected initializer. "
        "Torch allocation is not physical GPU memory.",
        "", "Fixed λ = 0.001 (means)", "", _table(
            ("Dataset", "Stage", "Train objective", "Path norm", "Training chambers"),
            [(row["dataset"], row["stage"], *(_cell(row["metrics"][name], row["expected"], mean_only=True)
              for name in ("train_objective", "path_norm", "n_chambers"))) for row in summary["fixed_rows"]]),
        "The initial AdamW model is evaluated at this penalty; AdamW itself uses decoupled weight decay.",
        "", "Paired warm-RCG objective changes at λ = 0.001", "", _table(
            ("Dataset", "Reference", "Improved / paired", "Worsened / paired", "Mean warm − reference"),
            [(row["dataset"], row["reference"], f"{row['improved']} / {row['n']}",
              f"{row['worsened']} / {row['n']}",
              _cell(row["objective_difference"], row["expected"], mean_only=True)) for row in summary["improvements"]]),
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--compact-output", type=Path)
    args = parser.parse_args(argv)
    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    summary = aggregate_results(payload, datasets=args.datasets.split(","), tracks=args.tracks.split(","),
                                seeds=tuple(map(int, args.seeds.split(","))))
    print(format_report(summary), end="")
    if args.compact_output:
        if args.compact_output.resolve() == args.input.resolve():
            raise ValueError("compact output must differ from raw input")
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.compact_output.suffix == ".gz" else open
        with opener(args.compact_output, "wt", encoding="utf-8") as stream:
            json.dump(compact_results(payload), stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
