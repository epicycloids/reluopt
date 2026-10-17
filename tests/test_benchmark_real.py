"""Protocol checks for comparable objectives and held-out model selection."""
import numpy as np
import pytest

from benchmarks import real


def test_optimizer_grids_keep_adamw_decay_distinct():
    adam = real.configurations("adam")
    adamw = real.configurations("adamw")
    assert len(adam) == 6 and len(adamw) == 4
    assert {r["lambda_mean"] for r in adam} == set(real.LAMBDAS)
    assert all(r["config"]["weight_decay"] > 0 for r in adamw)
    assert len(real.configurations("relu-cpu")) == 3


def test_scores_restore_target_units_without_affecting_r_squared():
    y = np.array([1.0, 2.0, 3.0])
    score = real.scores(y, np.array([1.0, 1.0, 3.0]), 7)
    np.testing.assert_allclose(score["rmse_original_units"], 7 / np.sqrt(3))
    assert score["r2"] == 0.5
    assert real.scores(np.ones(2), np.zeros(2), 1)["r2"] is None


def test_selection_uses_validation_and_scores_only_selected_test_model(monkeypatch, tmp_path):
    split = dict(
        X_train=np.ones((4, 1)), y_train=np.ones(4),
        X_val=np.ones((3, 1)), y_val=np.ones(3),
        X_test=np.ones((2, 1)), y_test=np.full(2, 2.0),
        transforms={"y_scale": 1.0}, metadata={"seed": 0},
    )
    monkeypatch.setattr(real, "load_dataset", lambda name: (np.ones((9, 1)), np.ones(9), {}))
    monkeypatch.setattr(real, "prepare_split", lambda *args: split)
    monkeypatch.setattr(real, "environment", lambda *args: {})
    configs = [dict(solver="relu-cpu", lambda_mean=0.001, config={"coefficient": b}) for b in (1, 2)]
    monkeypatch.setattr(real, "configurations", lambda *args: configs)

    def trial(input_path, spec, folder):
        # The child receives training observations only.
        with np.load(input_path) as arrays:
            assert set(arrays.files) == {"X_train", "y_train"}
        folder.mkdir(parents=True)
        np.savez(folder / "result.npz", U=np.ones((1, 1)), beta=[spec["config"]["coefficient"]])
        return dict(status="ok", seconds=0.1, process_seconds=0.2, fit_info={},
                    process_peak_rss_bytes=100, gpu_peak_bytes=None)

    monkeypatch.setattr(real, "_trial", trial)
    original_predict = real.predict
    test_calls = []

    def predict(X, U, beta):
        if X is split["X_test"]:
            test_calls.append(beta.copy())
        return original_predict(X, U, beta)

    monkeypatch.setattr(real, "predict", predict)
    result = real.run_dataset("concrete", seeds=(0,), solvers=("relu-cpu",), output_dir=tmp_path)
    # Coefficient2 wins on test but loses on validation; it must not be selected.
    assert result["selected"][0]["configuration"]["config"]["coefficient"] == 1
    assert result["selected"][0]["test"]["mse"] == 1
    assert len(test_calls) == 1
    assert test_calls[0][0] == 1
    assert result["selected"][0]["successful_fit_seconds"] == 0.2
    assert result["selected"][0]["tuning_seconds"] >= 0.4
    assert result["fixed_penalty"][0]["configuration"]["config"]["coefficient"] == 1


@pytest.mark.parametrize("prediction", [np.array([np.nan, 0.0]), np.array([1e308, 0.0])])
def test_nonfinite_scores_cannot_enter_model_selection(prediction):
    with pytest.raises(ValueError, match="nonfinite"):
        real.scores(np.zeros(2), prediction, 1)
