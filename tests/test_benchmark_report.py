"""Aggregation must expose incomplete coverage and preserve raw scalar results."""

import copy
import gzip
import json

import numpy as np
import pytest

from benchmarks.report_real import aggregate_results, compact_results, format_report, main


METHODS = ("relu-cpu", "sgd", "adamw")
OPTIONS = dict(datasets=("concrete",), solvers=METHODS, seeds=(0, 1, 2))


def _experiment(seed):
    selected, fixed, records = [], [], []
    for method in METHODS:
        failure = seed == 1 and method == "sgd"
        info = dict(trajectory=[dict(seconds=1.0, train_objective=0.5)])
        info["stop_reason" if method == "relu-cpu" else "stopping_reason"] = "time_budget"
        selected.append(dict(
            dataset="concrete", solver=method, split_seed=seed, status="ok",
            test=dict(rmse_original_units=1 + 2 * seed, r2=0.1 * (seed + 1)),
            seconds=1 + seed, tuning_seconds=20 + seed, fit_info=info,
            process_peak_rss_bytes=(10 + 10 * seed) * 1024**2,
            gpu_peak_bytes=None if method == "relu-cpu" else (30 + 30 * seed) * 1024**2,
            trials=2, failed_trials=int(failure),
        ))
        if method != "adamw":
            fixed.append(dict(
                dataset="concrete", solver=method, split_seed=seed,
                configuration=dict(lambda_mean=0.001),
                train_objective=(0.5, 0.3, 0.7)[seed], fit_info=info,
            ))
        for index in range(2):
            records.append(dict(
                dataset="concrete", solver=method, split_seed=seed, config_index=index,
                status="error" if failure and index == 1 else "ok",
                error="native failure" if failure and index == 1 else None,
                configuration=dict(lambda_mean=0.001),
                fit_info=info, trajectory=[dict(seconds=1.0, train_loss=0.4)],
            ))
    return dict(schema_version=1, dataset="concrete", selected=selected,
                fixed_penalty=fixed, records=records,
                protocol=dict(seeds=[seed], time_budget=15, split="grouped", width=256),
                source_sha256="source", benchmark_sha256="benchmark")


def _payload():
    return dict(schema_version=1, experiments=[_experiment(seed) for seed in range(3)])


def test_complete_aggregation_uses_sample_sd_and_medians():
    summary = aggregate_results(_payload(), **OPTIONS)
    assert summary["complete"]
    assert summary["successful"] == summary["expected"] == 9
    assert not summary["missing"] and not summary["failed"]
    rows = {row["solver"]: row for row in summary["rows"]}
    row = rows["sgd"]
    np.testing.assert_allclose(row["metrics"]["rmse"]["mean"], 3)
    np.testing.assert_allclose(row["metrics"]["rmse"]["sd"], 2)
    np.testing.assert_allclose(row["metrics"]["r2"]["sd"], 0.1)
    assert row["metrics"]["fit"]["median"] == 2
    assert row["metrics"]["tuning"]["median"] == 21
    assert row["metrics"]["rss"]["median"] == 20 * 1024**2
    assert row["metrics"]["gpu"]["median"] == 60 * 1024**2
    assert row["stop_counts"] == {"time_budget": 3}
    assert row["failed_trials"] == 1 and row["trials"] == 6
    np.testing.assert_allclose(row["fixed"]["mean"], 0.5)
    assert rows["adamw"]["fixed"]["n"] == 0
    assert rows["relu-cpu"]["metrics"]["gpu"]["n"] == 0
    report = format_report(summary)
    assert "9/9 successful; 0 failed; 0 missing" in report
    assert "3 ± 2 [3/3]" in report
    fixed_table = report.split("Fixed λ = 0.001", 1)[1].split("AdamW uses", 1)[0]
    assert "| adamw |" not in fixed_table


def test_missing_seeds_are_explicit_and_partial_means_never_look_complete():
    payload = _payload()
    payload["experiments"].pop()
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert len(summary["missing"]) == 3
    assert len(summary["missing_fixed"]) == 2
    assert summary["rows"][0]["metrics"]["rmse"]["n"] == 2
    report = format_report(summary)
    assert "6/9 successful; 0 failed; 3 missing" in report
    assert "[2/3; PARTIAL]" in report
    assert "concrete/sgd/seed-2" in report


def test_failed_selection_is_distinct_from_a_missing_selection():
    payload = _payload()
    experiment = payload["experiments"][1]
    experiment["selected"][1] = dict(dataset="concrete", solver="sgd", split_seed=1,
                                      status="failed", failed_trials=2)
    experiment["fixed_penalty"] = [row for row in experiment["fixed_penalty"] if row["solver"] != "sgd"]
    for row in experiment["records"]:
        if row["solver"] == "sgd":
            row["status"] = "error"
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert summary["missing"] == []
    assert summary["failed"] == [("concrete", "sgd", 1)]
    row = next(row for row in summary["rows"] if row["solver"] == "sgd")
    assert row["successful"] == 2
    assert row["stop_counts"] == {"selection_failed": 1, "time_budget": 2}
    assert row["failed_trials"] == 2


def test_fixed_penalty_coverage_is_checked_separately():
    payload = _payload()
    payload["experiments"][0]["fixed_penalty"].pop(0)
    summary = aggregate_results(payload, **OPTIONS)
    assert summary["successful"] == 9
    assert summary["missing_fixed"] == [("concrete", "relu-cpu", 0)]
    assert not summary["complete"]


def test_duplicate_rows_and_mixed_protocols_cannot_be_averaged_silently():
    payload = _payload()
    payload["experiments"][0]["selected"].append(copy.deepcopy(payload["experiments"][0]["selected"][0]))
    with pytest.raises(ValueError, match="duplicate selected"):
        aggregate_results(payload, **OPTIONS)
    payload = _payload()
    payload["experiments"][0]["protocol"]["time_budget"] = 30
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert "mixed protocol.time_budget across experiments" in summary["issues"]


def test_nonfinite_metrics_and_inconsistent_failure_counts_are_not_hidden():
    payload = _payload()
    payload["experiments"][0]["selected"][0]["test"]["rmse_original_units"] = float("nan")
    with pytest.raises(ValueError, match="RMSE must be a finite number"):
        aggregate_results(payload, **OPTIONS)
    payload = _payload()
    payload["experiments"][0]["selected"][0]["failed_trials"] = 1
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert any("failed_trials disagrees" in issue for issue in summary["issues"])


def test_one_seed_has_no_invented_sample_standard_deviation():
    summary = aggregate_results(_payload(), datasets=("concrete",), solvers=METHODS, seeds=(0,))
    assert summary["complete"]
    assert summary["rows"][0]["metrics"]["rmse"]["sd"] is None
    assert "1 ± n/a [1/1]" in format_report(summary)


def test_explicit_subset_ignores_provenance_of_unselected_experiments():
    payload = _payload()
    payload["experiments"][2]["protocol"]["time_budget"] = 30
    payload["experiments"][2]["source_sha256"] = "another-source"
    summary = aggregate_results(payload, datasets=("concrete",), solvers=METHODS, seeds=(0,))
    assert summary["complete"]
    assert summary["successful"] == 3
    assert not summary["issues"]
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert "mixed protocol.time_budget across experiments" in summary["issues"]
    assert "mixed source_sha256 across experiments" in summary["issues"]


def test_compaction_preserves_schema_metadata_metrics_and_errors_without_mutation():
    payload = _payload()
    before = copy.deepcopy(payload)
    result = compact_results(payload)
    assert payload == before
    assert result["schema_version"] == payload["schema_version"]
    assert result["experiments"][0]["protocol"] == payload["experiments"][0]["protocol"]
    for experiment in result["experiments"]:
        for row in experiment["records"] + experiment["selected"] + experiment["fixed_penalty"]:
            assert "trajectory" not in row
            assert "trajectory" not in row["fit_info"]
    raw_error = payload["experiments"][1]["records"][3]
    assert result["experiments"][1]["records"][3]["error"] == raw_error["error"] == "native failure"
    assert aggregate_results(result, **OPTIONS) == aggregate_results(payload, **OPTIONS)


def test_cli_reads_gzip_writes_same_compact_schema_and_returns_coverage_status(tmp_path, capsys):
    source, output = tmp_path / "raw.json.gz", tmp_path / "compact.json"
    with gzip.open(source, "wt") as stream:
        json.dump(_payload(), stream)
    options = [str(source), "--datasets", "concrete", "--solvers", ",".join(METHODS),
               "--seeds", "0,1,2", "--compact-output", str(output)]
    assert main(options) == 0
    assert "9/9 successful" in capsys.readouterr().out
    assert json.loads(output.read_text()) == compact_results(_payload())
    options[options.index("0,1,2")] = "0,1,2,3"
    assert main(options) == 1
    assert "3 missing" in capsys.readouterr().out
