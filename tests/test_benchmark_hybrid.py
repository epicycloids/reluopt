"""Paired learned-direction initialization and held-out selection checks."""
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import hybrid, hybrid_trial


def test_chamber_count_retains_distinct_directions_in_one_chamber():
    X = np.array([[1., 1.], [2., 1.], [-1., -1.]])
    assert hybrid.chamber_count(X, np.eye(2)) == 1
    assert hybrid.chamber_count(X, np.array([[1., -1.], [0., 0.]])) == 2
    assert hybrid.chamber_count(X, np.empty((2, 0))) == 0
    # Match the solver's nonnegative side at an exact chamber boundary.
    assert hybrid.chamber_count(np.array([[0., 1.], [1., 1.]]), np.eye(2)) == 1


def test_model_fingerprint_tracks_exact_weights_and_shapes():
    U, beta = np.eye(2), np.array([1., -1.])
    assert hybrid_trial.model_digest(U, beta) == hybrid_trial.model_digest(U.astype(np.float32), beta)
    assert hybrid_trial.model_digest(U, beta) != hybrid_trial.model_digest(U, -beta)
    assert hybrid_trial.model_digest(U, beta) != hybrid_trial.model_digest(U.reshape(1, 4), beta)


def test_trial_rejects_changed_initializer_before_fitting():
    U, beta = np.eye(2), np.array([1., -1.])
    with pytest.raises(ValueError, match="fingerprint"):
        hybrid_trial.fit(U, beta, dict(operation="warm-rcg", initializer_sha256="wrong"), U, beta)
    with pytest.raises(ValueError, match="must not receive"):
        hybrid_trial.fit(U, beta, dict(operation="cold-rcg"), U, beta)
    with pytest.raises(ValueError, match="require"):
        hybrid_trial.fit(U, beta, dict(operation="coefficient-refit"))


@pytest.mark.parametrize("operation,expected_rounds", [("coefficient-refit", 0), ("warm-rcg", 80)])
def test_warm_trial_passes_exact_exported_model_to_core(monkeypatch, operation, expected_rounds):
    import reluopt

    X, y = np.ones((4, 2)), np.ones(4)
    U, beta = np.eye(2), np.array([1., -1.])
    observed = {}

    def solve(X, y, **kwargs):
        observed.update(kwargs)
        return dict(U=U, beta=beta, stop_reason="done", n_iter=expected_rounds,
                    finite_master_kkt=0., finite_master_converged=True,
                    verified_float64=True, n_cols=2, n_active=2, timings={},
                    initialization=dict(n_cols=2))

    monkeypatch.setattr(reluopt, "solve_rcg", solve)
    spec = dict(operation=operation, device="cpu", threads=1, max_cols=512,
                time_budget=15, lambda_mean=.001, seed=0,
                initializer_sha256=hybrid_trial.model_digest(U, beta))
    actual_U, actual_beta, info = hybrid_trial.fit(X, y, spec, U, beta)
    assert observed["initial_U"] is U and observed["initial_beta"] is beta
    assert observed["max_iter"] == expected_rounds
    assert observed["max_cols"] == 512
    assert observed["lam"] == .004
    assert observed["final_verify"] and not observed["final_diagnostics"]
    np.testing.assert_array_equal(actual_U, U)
    np.testing.assert_array_equal(actual_beta, beta)
    assert info["fit_info"]["initialization"]["n_cols"] == 2


def test_paired_protocol_freezes_validation_selected_initializer_and_accounts_for_shared_cost(monkeypatch, tmp_path):
    split = dict(
        X_train=np.ones((4, 1)), y_train=np.ones(4),
        X_val=np.ones((3, 1)), y_val=np.ones(3),
        X_test=np.ones((2, 1)), y_test=np.full(2, 2.),
        transforms={"y_scale": 1.}, metadata={"seed": 0},
    )
    monkeypatch.setattr(hybrid, "load_dataset", lambda name: (np.ones((9, 1)), np.ones(9), {}))
    monkeypatch.setattr(hybrid, "prepare_split", lambda *args: split)
    monkeypatch.setattr(hybrid, "environment", lambda *args: {})
    parents, observed_specs = [], []

    def trial(input_path, spec, folder):
        observed_specs.append(spec)
        with np.load(input_path) as arrays:
            if spec["operation"] in {"warm-rcg", "coefficient-refit"}:
                assert set(arrays.files) == {"X_train", "y_train", "initial_U", "initial_beta"}
                parents.append((arrays["initial_U"].copy(), arrays["initial_beta"].copy()))
                assert spec["initializer_sha256"] == hybrid_trial.model_digest(*parents[-1])
            else:
                assert set(arrays.files) == {"X_train", "y_train"}
        if spec["operation"] == "adamw":
            coefficient = 1. if spec["config"]["lr"] == .001 and spec["config"]["weight_decay"] == .0001 else 2.
            assert spec["config"]["max_epochs"] == 1_000_000
        else:
            coefficient = 1. if spec["lambda_mean"] == .001 else 2.
        folder.mkdir(parents=True)
        np.savez(folder / "result.npz", U=np.ones((1, 1)), beta=[coefficient])
        return dict(status="ok", seconds=.1, process_seconds=.2, fit_info={},
                    process_peak_rss_bytes=100, gpu_peak_bytes=10)

    monkeypatch.setattr(hybrid, "_trial", trial)
    original_predict, test_calls = hybrid.predict, []

    def predict(X, U, beta):
        if X is split["X_test"]:
            test_calls.append(beta.copy())
        return original_predict(X, U, beta)

    monkeypatch.setattr(hybrid, "predict", predict)
    result = hybrid.run_dataset("concrete", seeds=(0,), output_dir=tmp_path)
    assert len(result["records"]) == 20
    assert len(parents) == 6
    for U, beta in parents:
        np.testing.assert_array_equal(U, np.ones((1, 1)))
        np.testing.assert_array_equal(beta, np.ones(1))
    assert len(test_calls) == len(hybrid.TRACKS) == 6
    assert all(beta[0] == 1 for beta in test_calls)  # Test prefers2; validation prefers1.
    selected = {r["solver"]: r for r in result["selected"]}
    warm = selected["warm-rcg"]
    assert warm["selected_fit_seconds"] == .2
    assert warm["stage_tuning_seconds"] >= .6
    assert warm["shared_initializer_tuning_seconds"] >= .8
    assert warm["tuning_seconds"] >= 1.4
    assert selected["cold-rcg"]["configuration"]["time_budget"] == 15
    assert selected["cold-rcg-long"]["configuration"]["time_budget"] == 30
    assert selected["adamw-long"]["configuration"]["time_budget"] == 30
    assert selected["cold-rcg"]["shared_initializer_tuning_seconds"] == 0
    assert len(result["paired"]) == 3
    assert all(r["initial_n_chambers"] == 1 for r in result["paired"])
    assert all(r["initial_train_objective"] == r["lambda_mean"] for r in result["paired"])
    assert len({spec["initializer_sha256"] for spec in observed_specs if "initializer_sha256" in spec}) == 1


def test_failed_initializer_does_not_start_paired_children(monkeypatch, tmp_path):
    split = dict(X_train=np.ones((2, 1)), y_train=np.ones(2), metadata={})
    monkeypatch.setattr(hybrid, "load_dataset", lambda name: (None, None, {}))
    monkeypatch.setattr(hybrid, "prepare_split", lambda *args: split)
    monkeypatch.setattr(hybrid, "environment", lambda *args: {})
    operations = []

    def trial(input_path, spec, folder):
        operations.append(spec["operation"])
        return dict(status="error", error="fit failed", process_seconds=.2)

    monkeypatch.setattr(hybrid, "_trial", trial)
    result = hybrid.run_dataset("concrete", seeds=(0,), output_dir=tmp_path, controls=False, smoke=True)
    assert operations == ["adamw"]
    assert all(r["status"] == "failed" for r in result["selected"])
    assert result["paired"] == []


def _cached_trial_setup(monkeypatch, tmp_path, first_failure=None):
    path = tmp_path / "data.npz"
    np.savez(path, X_train=np.ones((2, 1)), y_train=np.ones(2))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = tmp_path / "fit" / "result.json"
        if len(calls) == 1 and first_failure == "timeout":
            raise subprocess.TimeoutExpired(command, 91)
        if len(calls) == 1 and first_failure == "error":
            output.write_text(json.dumps(dict(status="error", error="interrupted fit")))
        else:
            np.savez(output.with_suffix(".npz"), U=np.ones((1, 1)), beta=[float(len(calls))])
            output.write_text(json.dumps(dict(status="ok", seconds=1.25, fit_info={})))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(hybrid.subprocess, "run", run)
    return path, dict(operation="adamw", time_budget=1.), tmp_path / "fit", calls


def test_completed_fit_cache_preserves_timing_and_ignores_npz_member_order(monkeypatch, tmp_path):
    path, spec, folder, calls = _cached_trial_setup(monkeypatch, tmp_path)
    first = hybrid._trial(path, spec, folder)
    # Rewriting the archive changes ZIP layout without changing its arrays.
    np.savez(path, y_train=np.ones(2), X_train=np.ones((2, 1)))
    repeated = hybrid._trial(path, dict(reversed(list(spec.items()))), folder)
    assert repeated == first
    assert repeated["seconds"] == 1.25
    assert repeated["process_seconds"] == first["process_seconds"]
    assert len(calls) == 1
    cache = json.loads((folder / "trial-cache.json").read_text())
    assert cache["result"] == first
    assert cache["model_sha256"] == hybrid_trial.model_digest(np.ones((1, 1)), np.ones(1))


def test_fit_cache_invalidates_changed_config_input_source_and_model(monkeypatch, tmp_path):
    path, spec, folder, calls = _cached_trial_setup(monkeypatch, tmp_path)
    hybrid._trial(path, spec, folder)
    spec = dict(spec, time_budget=2.)
    hybrid._trial(path, spec, folder)
    assert len(calls) == 2
    np.savez(path, X_train=np.ones((2, 1)), y_train=np.full(2, 3.))
    hybrid._trial(path, spec, folder)
    assert len(calls) == 3
    monkeypatch.setattr(hybrid, "_trial_source_digest", lambda: "changed implementation")
    hybrid._trial(path, spec, folder)
    assert len(calls) == 4
    np.savez(folder / "result.npz", U=np.ones((1, 1)), beta=[-77.])
    hybrid._trial(path, spec, folder)
    assert len(calls) == 5
    (folder / "trial-cache.json").write_text('{"interrupted":')
    hybrid._trial(path, spec, folder)
    assert len(calls) == 6


def test_fit_cache_invalidates_changed_python_or_package_versions(monkeypatch, tmp_path):
    path, spec, folder, calls = _cached_trial_setup(monkeypatch, tmp_path)
    runtime = dict(python="3.12.0", packages=dict(numpy="1", scipy="2", torch="3"))
    monkeypatch.setattr(hybrid, "_trial_environment", lambda: runtime)
    hybrid._trial(path, spec, folder)
    runtime["python"] = "3.12.1"
    hybrid._trial(path, spec, folder)
    assert len(calls) == 2
    runtime["packages"]["torch"] = "4"
    hybrid._trial(path, spec, folder)
    assert len(calls) == 3
    assert hybrid._trial(path, spec, folder)["status"] == "ok"
    assert len(calls) == 3


@pytest.mark.parametrize("failure", ["timeout", "error"])
def test_incomplete_fits_are_retried_and_only_completed_results_are_cached(monkeypatch, tmp_path, failure):
    path, spec, folder, calls = _cached_trial_setup(monkeypatch, tmp_path, failure)
    assert hybrid._trial(path, spec, folder)["status"] == failure
    assert not (folder / "trial-cache.json").exists()
    (folder / "trial-cache.tmp").write_text("interrupted cache write")
    successful = hybrid._trial(path, spec, folder)
    assert successful["status"] == "ok"
    assert len(calls) == 2
    assert (folder / "trial-cache.json").exists()
    assert not (folder / "trial-cache.tmp").exists()
    assert hybrid._trial(path, spec, folder) == successful
    assert len(calls) == 2
