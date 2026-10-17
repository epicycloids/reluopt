"""Reports expose missing pairs and count the shared initializer exactly once."""
import copy
import json

import numpy as np
import pytest

from benchmarks.hybrid import TRACKS
from benchmarks.report_hybrid import aggregate_results, format_report, main


OPTIONS = dict(datasets=("concrete",), seeds=(0, 1, 2))


def _payload():
    experiments = []
    for seed in range(3):
        selected, records = [], []
        for track in TRACKS:
            child = track in {"warm-rcg", "coefficient-refit"}
            selected.append(dict(dataset="concrete", solver=track, split_seed=seed, status="ok",
                                 test=dict(rmse_original_units=1 + 2 * seed, r2=.5),
                                 seconds=1 + seed, selected_fit_seconds=2 + seed if child else 1 + seed,
                                 initializer_fit_seconds=1 if child else 0,
                                 tuning_seconds=30 if child else 20, stage_tuning_seconds=20,
                                 shared_initializer_tuning_seconds=10 if child else 0,
                                 pipeline_peak_rss_bytes=1024**2, pipeline_gpu_peak_bytes=2 * 1024**2,
                                 trials=1, fit_info=dict(stop_reason="time_budget")))
            records.append(dict(dataset="concrete", solver=track, split_seed=seed, status="ok",
                                trajectory=[dict(seconds=1)]))
        stages = {name: dict(train_objective=value + .1 * seed, path_norm=3, n_chambers=5)
                  for name, value in (("coefficient-refit", .4), ("warm-rcg", .3),
                                      ("cold-rcg", .5), ("cold-rcg-long", .35))}
        paired = [dict(dataset="concrete", split_seed=seed, lambda_mean=.001,
                       initial_train_objective=.6 + .1 * seed, initial_path_norm=4,
                       initial_n_chambers=6, **stages)]
        experiments.append(dict(schema_version=1, dataset="concrete", selected=selected,
                                records=records, paired=paired, protocol=dict(width=256)))
    return dict(schema_version=1, experiments=experiments)


def test_paired_report_uses_pipeline_time_sample_sd_and_matching_seed_differences():
    summary = aggregate_results(_payload(), **OPTIONS)
    assert summary["complete"]
    assert summary["successful"] == summary["expected"] == 18
    warm = next(row for row in summary["rows"] if row["solver"] == "warm-rcg")
    assert warm["metrics"]["fit"]["median"] == 3
    assert warm["metrics"]["stage_fit"]["median"] == 2
    assert warm["metrics"]["tuning"]["median"] == 30
    assert warm["metrics"]["rmse"]["mean"] == 3
    assert warm["metrics"]["rmse"]["sd"] == 2
    initial = next(row for row in summary["improvements"] if row["reference"] == "initial")
    assert initial["improved"] == initial["n"] == 3
    assert initial["worsened"] == 0
    np.testing.assert_allclose(initial["objective_difference"]["mean"], -.3)
    report = format_report(summary)
    assert "18/18 successful; 0 failed; 0 missing" in report
    assert "3 ± 2 [3/3]" in report
    assert "| concrete | initial | 0.7 [3/3]" in report


def test_missing_pairs_and_selections_are_explicit_and_untotaled_pipeline_time_is_rejected():
    payload = _payload()
    payload["experiments"].pop()
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert len(summary["missing"]) == 6 and len(summary["missing_paired"]) == 1
    assert "PARTIAL" in format_report(summary)
    payload = _payload()
    payload["experiments"][0]["selected"][2]["selected_fit_seconds"] = 1
    summary = aggregate_results(payload, **OPTIONS)
    assert not summary["complete"]
    assert "fit does not include" in " ".join(summary["issues"])


def test_duplicate_selection_is_rejected():
    payload = _payload()
    payload["experiments"].append(copy.deepcopy(payload["experiments"][0]))
    with pytest.raises(ValueError, match="duplicate selected"):
        aggregate_results(payload, **OPTIONS)


def test_cli_compacts_trajectories_without_changing_schema(tmp_path, capsys):
    source, compact = tmp_path / "raw.json", tmp_path / "compact.json"
    source.write_text(json.dumps(_payload()))
    assert main([str(source), "--datasets", "concrete", "--compact-output", str(compact)]) == 0
    assert "18/18" in capsys.readouterr().out
    result = json.loads(compact.read_text())
    assert result["schema_version"] == 1
    assert "trajectory" not in result["experiments"][0]["records"][0]
    assert "trajectory" in json.loads(source.read_text())["experiments"][0]["records"][0]
