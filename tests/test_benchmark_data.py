"""Small loader fixtures and train-only preprocessing checks; no downloads."""

import hashlib
import io
import json
import tarfile
import zipfile

import numpy as np
import pytest

from benchmarks import datasets


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("unit tests must not access the network")
    monkeypatch.setattr(datasets, "urlopen", blocked)


def fixture_source(monkeypatch, tmp_path, name, payload, shape):
    spec = dict(datasets.DATASETS[name], sha256=hashlib.sha256(payload).hexdigest(),
                shape=shape)
    monkeypatch.setitem(datasets.DATASETS, name, spec)
    (tmp_path / spec["filename"]).write_bytes(payload)
    return spec


def test_csv_cache_loader_preserves_units_and_provenance(monkeypatch, tmp_path):
    payload = b'"feature a";"feature b";"quality"\n1;2;3\n4;5;6\n7;8;9\n'
    spec = fixture_source(monkeypatch, tmp_path, "wine-white", payload, (3, 2))
    X, y, meta = datasets.load_dataset("wine-white", tmp_path)
    np.testing.assert_array_equal(X, [[1, 2], [4, 5], [7, 8]])
    np.testing.assert_array_equal(y, [3, 6, 9])
    assert X.dtype == y.dtype == np.float64
    assert meta["feature_names"] == ["feature a", "feature b"]
    assert meta["sha256"] == spec["sha256"]
    assert meta["doi"] == "10.24432/C56S3T"
    assert meta["license"] == "CC-BY-4.0"
    assert json.loads((tmp_path / "wine-white.metadata.json").read_text()) == meta


def test_corrupt_cache_fails_without_network_or_replacement(monkeypatch, tmp_path):
    payload = b"a,b,target\n1,2,3\n"
    spec = fixture_source(monkeypatch, tmp_path, "concrete", payload, (1, 2))
    path = tmp_path / spec["filename"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        datasets.load_dataset("concrete", tmp_path)
    assert path.read_bytes() == b"changed"


def test_download_requires_pinned_hash_and_installs_atomically(monkeypatch, tmp_path):
    payload = b"a,b,target\n1,2,3\n"
    spec = dict(datasets.DATASETS["concrete"], sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(datasets, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    path = datasets._fetch_file(spec, tmp_path)
    assert path.read_bytes() == payload
    assert not list(tmp_path.glob(".download-*"))
    path.unlink()
    bad = dict(spec, sha256="0"*64)
    with pytest.raises(ValueError, match="checksum"):
        datasets._fetch_file(bad, tmp_path)
    assert not path.exists()
    assert not list(tmp_path.glob(".download-*"))


def test_superconductivity_archive_reads_only_feature_table(monkeypatch, tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("train.csv", "f1,f2,critical_temp\n1,2,7\n3,4,8\n")
        archive.writestr("unique_m.csv", "unrelated,chemical,identifiers\n")
    fixture_source(monkeypatch, tmp_path, "superconductivity", stream.getvalue(), (2, 2))
    X, y, meta = datasets.load_dataset("superconductivity", tmp_path)
    np.testing.assert_array_equal(X, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(y, [7, 8])
    assert meta["target_units"] == "K"


def test_california_feature_ratios_and_raw_dollar_target(monkeypatch, tmp_path):
    # longitude, latitude, age, rooms, bedrooms, population, households,
    # income, median house value.
    data = b"-122,38,25,100,20,80,10,3,210000\n-121,37,30,90,18,60,6,4,300000\n"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("CaliforniaHousing/cal_housing.data")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    fixture_source(monkeypatch, tmp_path, "california", stream.getvalue(), (2, 8))
    X, y, meta = datasets.load_dataset("california", tmp_path)
    np.testing.assert_array_equal(X, [[3, 25, 10, 2, 80, 8, 38, -122],
                                      [4, 30, 15, 3, 60, 10, 37, -121]])
    np.testing.assert_array_equal(y, [210000, 300000])
    assert meta["target_units"] == "USD"


def test_cache_environment_and_explicit_override(monkeypatch, tmp_path):
    payload = b"a,b,target\n1,2,3\n"
    fixture_source(monkeypatch, tmp_path, "concrete", payload, (1, 2))
    monkeypatch.setenv("RELUOPT_DATA_CACHE", str(tmp_path))
    X, y, _ = datasets.load_dataset("concrete")
    assert X.shape == (1, 2)
    assert y[0] == 3
    other = tmp_path / "other"
    other.mkdir()
    fixture_source(monkeypatch, other, "concrete", payload, (1, 2))
    datasets.load_dataset("concrete", other)
    assert (other / "concrete.metadata.json").exists()


def test_split_is_reproducible_disjoint_complete_and_traceable():
    X = np.arange(33, dtype=float).reshape(11, 3)
    y = np.arange(11, dtype=float)
    result = datasets.prepare_split(X, y, seed=7)
    again = datasets.prepare_split(X, y, seed=7)
    idx = result["indices"]
    assert {key: len(value) for key, value in idx.items()} == {"train": 7, "val": 2, "test": 2}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))), np.arange(11))
    for key in idx:
        np.testing.assert_array_equal(idx[key], again["indices"][key])
        assert result["metadata"]["split_sha256"][key] == hashlib.sha256(
            np.asarray(idx[key], dtype="<i8").tobytes()).hexdigest()
        assert result[f"X_{key}"].shape == (len(idx[key]), 4)
        np.testing.assert_array_equal(result[f"X_{key}"][:, -1], 1)
    assert result["metadata"] == again["metadata"]
    json.dumps(result["metadata"], allow_nan=False)
    json.dumps(result["transforms"], allow_nan=False)


def test_imputation_and_scaling_use_train_only_and_preserve_inputs():
    X = np.column_stack((np.arange(15, dtype=float), np.full(15, 0.1), np.full(15, np.nan)))
    y = np.linspace(10, 20, 15)
    initial = datasets.prepare_split(X, y, seed=4)
    train = initial["indices"]["train"]
    heldout = np.r_[initial["indices"]["val"], initial["indices"]["test"]]
    X[train[0], 0] = np.nan
    X[heldout, 2] = 99
    before_X, before_y = X.copy(), y.copy()
    result = datasets.prepare_split(X, y, seed=4)
    changed_X, changed_y = X.copy(), y.copy()
    changed_X[heldout, 0] += 10000
    changed_y[heldout] += 10000
    changed = datasets.prepare_split(changed_X, changed_y, seed=4)
    assert result["transforms"] == changed["transforms"]
    np.testing.assert_array_equal(result["X_train"], changed["X_train"])
    np.testing.assert_array_equal(result["y_train"], changed["y_train"])
    np.testing.assert_array_equal(X, before_X)
    np.testing.assert_array_equal(y, before_y)
    expected_mean = np.nanmean(X[train, 0])
    assert result["transforms"]["x_impute"][0] == expected_mean
    np.testing.assert_allclose(result["X_train"][:, 0].mean(), 0, atol=1e-15)
    np.testing.assert_allclose(result["X_train"][:, 0].std(), 1, atol=1e-15)
    np.testing.assert_array_equal(result["X_train"][:, 1:3], 0)
    assert result["transforms"]["all_missing_features"] == [2]
    assert result["transforms"]["x_scale"][1:] == [1, 1]
    for key, idx in result["indices"].items():
        reconstructed = result[f"y_{key}"] * result["transforms"]["y_scale"] + result["transforms"]["y_mean"]
        np.testing.assert_allclose(reconstructed, y[idx], rtol=1e-15)


def test_exact_constant_target_and_features_are_not_amplified():
    X = np.column_stack((np.arange(10), np.full(10, 0.1)))
    result = datasets.prepare_split(X, np.full(10, 0.1), seed=0)
    assert result["transforms"]["y_scale"] == 1
    assert result["transforms"]["x_scale"][1] == 1
    for key in ("train", "val", "test"):
        np.testing.assert_array_equal(result[f"X_{key}"][:, 1], 0)
        np.testing.assert_array_equal(result[f"y_{key}"], 0)


def test_exact_duplicate_X_with_different_targets_never_crosses_partitions():
    X = np.repeat(np.arange(10, dtype=float)[:, None], [1, 2, 3, 4, 5, 1, 2, 3, 4, 5], axis=0)
    y = np.arange(len(X), dtype=float)  # every repeated feature has different targets
    result = datasets.prepare_split(X, y, seed=18)
    partition = np.empty(len(X), dtype=np.int64)
    for number, (name, idx) in enumerate(result["indices"].items()):
        partition[idx] = number
        assert len(idx) > 0
        assert len(result["group_indices"][name]) == result["metadata"]["split_group_counts"][name]
    for value in np.unique(X):
        assert len(np.unique(partition[X[:, 0] == value])) == 1
    assert result["metadata"]["group_count"] == 10
    assert result["metadata"]["duplicate_group_count"] == 8
    assert result["metadata"]["duplicate_rows_beyond_first"] == 20
    assert result["metadata"]["rows_in_duplicate_groups"] == 28
    assert not result["metadata"]["grouping_uses_targets"]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(result["indices"].values()))), np.arange(len(X)))


def test_target_permutation_cannot_change_group_assignment():
    X = np.repeat(np.arange(9, dtype=float)[:, None], 2, axis=0)
    y = np.arange(len(X), dtype=float)
    original = datasets.prepare_split(X, y, seed=9)
    permuted = datasets.prepare_split(X, y[::-1], seed=9)
    for name in original["indices"]:
        np.testing.assert_array_equal(original["indices"][name], permuted["indices"][name])
    assert original["metadata"]["split_group_sha256"] == permuted["metadata"]["split_group_sha256"]
    assert original["metadata"]["array_sha256"] != permuted["metadata"]["array_sha256"]


def test_nan_payloads_and_signed_zero_use_same_X_group():
    X = np.array([[np.nan, 0.], [np.nan, -0.], [1., 2.], [1., 2.], [3., 4.], [5., 6.]])
    # Another quiet NaN payload must still represent the same missing value.
    X.view(np.uint64)[1, 0] = 0x7ff8000000000001
    result = datasets.prepare_split(X, np.arange(6, dtype=float), seed=8)
    assert result["group_ids"][0] == result["group_ids"][1]
    assert result["group_ids"][2] == result["group_ids"][3]
    assert result["metadata"]["group_count"] == 4
    for name in ("train", "val", "test"):
        assert np.isfinite(result[f"X_{name}"]).all()


def test_nearest_group_boundaries_are_nonempty_and_use_row_counts():
    X = np.repeat(np.arange(5, dtype=float)[:, None], [1, 8, 2, 3, 1], axis=0)
    result = datasets.prepare_split(X, np.arange(len(X), dtype=float), seed=1)
    order = np.concatenate(list(result["group_indices"].values()))
    sizes = np.bincount(result["group_ids"])
    cumulative = np.cumsum(sizes[order])
    first = result["metadata"]["split_group_counts"]["train"]
    second = first + result["metadata"]["split_group_counts"]["val"]
    assert abs(cumulative[first-1]-.6*len(X)) == min(abs(cumulative[:len(order)-2]-.6*len(X)))
    assert abs(cumulative[second-1]-.8*len(X)) == min(abs(cumulative[first:len(order)-1]-.8*len(X)))
    assert min(result["metadata"]["split_counts"].values()) > 0


def test_fewer_than_three_distinct_feature_groups_cannot_be_split():
    with pytest.raises(ValueError, match="three distinct X groups"):
        datasets.prepare_split(np.zeros((10, 2)), np.arange(10), seed=0)


@pytest.mark.parametrize("change", [
    {"X": np.ones((4, 1)), "y": np.ones(4)},
    {"X": np.ones((10, 0))},
    {"X": np.ones((10, 2), dtype=complex)},
    {"X": np.full((10, 2), np.inf)},
    {"y": np.full(10, np.nan)},
    {"y": np.ones(10, dtype=complex)},
    {"y": np.ones((10, 1))},
    {"seed": -1}, {"seed": True}, {"seed": 1.5},
])
def test_invalid_preprocessing_inputs(change):
    options = dict(X=np.ones((10, 2)), y=np.arange(10), seed=0)
    options.update(change)
    with pytest.raises(ValueError):
        datasets.prepare_split(**options)


def test_unknown_dataset_is_an_error():
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.load_dataset("not-a-dataset")
