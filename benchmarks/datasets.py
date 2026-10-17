"""Pinned public regression data and train-only benchmark preprocessing.

Raw downloads are cached, never bundled with the package. All parsers use
NumPy and the standard library; Concrete uses UCI's official CSV export.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np


LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASETS = {
    "concrete": dict(
        title="Concrete Compressive Strength", shape=(1030, 8),
        url="https://archive.ics.uci.edu/static/public/165/data.csv",
        filename="concrete.csv", format="csv", delimiter=",",
        sha256="8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af",
        source_url="https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        export_metadata_url="https://archive.ics.uci.edu/api/dataset?id=165",
        doi="10.24432/C5PK67", target_name="Concrete compressive strength",
        target_units="MPa", citation="Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.",
    ),
    "wine-white": dict(
        title="Wine Quality (white wine)", shape=(4898, 11),
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        filename="wine-white.csv", format="csv", delimiter=";",
        sha256="76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
        source_url="https://archive.ics.uci.edu/dataset/186/wine+quality",
        doi="10.24432/C56S3T", target_name="quality", target_units="quality score (0–10)",
        citation="Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Wine Quality. UCI Machine Learning Repository.",
    ),
    "california": dict(
        title="California Housing", shape=(20640, 8),
        url="https://ndownloader.figshare.com/files/5976036",
        filename="california.tgz", format="california",
        sha256="aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
        source_url="https://figshare.com/articles/dataset/cal_housing_tgz/3829992",
        license_source_url="https://api.figshare.com/v2/articles/3829992",
        feature_definition_url="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_california_housing.py",
        doi="10.6084/m9.figshare.3829992.v2", target_name="median house value",
        target_units="USD", citation="Pace, R. K. and Barry, R. (1997). Sparse Spatial Autoregressions. Statistics and Probability Letters 33, 291–297.",
    ),
    "superconductivity": dict(
        title="Superconductivty Data", shape=(21263, 81),
        url="https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        filename="superconductivity.zip", format="superconductivity",
        sha256="87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e",
        source_url="https://archive.ics.uci.edu/dataset/464/superconductivty+data",
        doi="10.24432/C53P47", target_name="critical_temp", target_units="K",
        citation="Hamidieh, K. (2018). Superconductivty Data. UCI Machine Learning Repository.",
    ),
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_file(spec, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / spec["filename"]
    if path.exists():
        if _file_sha256(path) != spec["sha256"]:
            raise ValueError(f"cached dataset checksum mismatch: {path}; remove it to download again")
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".download-", delete=False) as target:
            temporary = Path(target.name)
            request = Request(spec["url"], headers={"User-Agent": "reluopt benchmark data"})
            with urlopen(request, timeout=60) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
        if _file_sha256(temporary) != spec["sha256"]:
            raise ValueError(f"downloaded dataset checksum mismatch: {spec['url']}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _read_csv(stream, delimiter=","):
    header_line = stream.readline()
    if isinstance(header_line, bytes):
        header_line = header_line.decode("utf-8-sig")
    header = next(csv.reader([header_line], delimiter=delimiter))
    values = np.loadtxt(stream, delimiter=delimiter, dtype=np.float64, ndmin=2)
    if values.shape[1] != len(header):
        raise ValueError("dataset header and numeric columns do not match")
    return values[:, :-1], values[:, -1], [name.strip() for name in header[:-1]]


def _read_california(path):
    with tarfile.open(path, "r:gz") as archive:
        with archive.extractfile("CaliforniaHousing/cal_housing.data") as stream:
            data = np.loadtxt(stream, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 9:
        raise ValueError("California Housing archive has the wrong data shape")
    # StatLib columns: longitude, latitude, age, total rooms, total bedrooms,
    # population, households, income, median house value in dollars.
    households = data[:, 6]
    if np.any(households <= 0):
        raise ValueError("California Housing contains nonpositive household counts")
    X = np.column_stack((data[:, 7], data[:, 2], data[:, 3] / households,
                         data[:, 4] / households, data[:, 5], data[:, 5] / households,
                         data[:, 1], data[:, 0]))
    return X, data[:, 8], ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                          "Population", "AveOccup", "Latitude", "Longitude"]


def _data_sha256(X, y):
    digest = hashlib.sha256()
    for array in (X, y):
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_dataset(name, cache_dir=None):
    """Return ``(X, y, metadata)`` from a verified official download.

    ``X`` and ``y`` retain raw feature/target units. California uses the same
    eight feature definitions as scikit-learn, with its target in dollars.
    Cache precedence: explicit directory, ``RELUOPT_DATA_CACHE``,
    ``runs/data``. A modified cached or upstream file fails its SHA256 check.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    spec = DATASETS[name]
    cache = Path(cache_dir if cache_dir is not None else
                 os.environ.get("RELUOPT_DATA_CACHE", "runs/data"))
    path = _fetch_file(spec, cache)
    if spec["format"] == "california":
        X, y, names = _read_california(path)
    elif spec["format"] == "superconductivity":
        with zipfile.ZipFile(path) as archive:
            with archive.open("train.csv") as stream:
                X, y, names = _read_csv(stream)
    else:
        with path.open("rb") as stream:
            X, y, names = _read_csv(stream, spec["delimiter"])
    X, y = np.ascontiguousarray(X), np.ascontiguousarray(y)
    if X.shape != spec["shape"] or y.shape != (spec["shape"][0],):
        raise ValueError(f"{name} has unexpected dimensions: {X.shape}, {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{name} contains nonfinite raw data")
    metadata = {key: value for key, value in spec.items() if key not in ("format", "delimiter")}
    metadata.update(name=name, shape=list(X.shape), feature_names=names,
                    license="CC-BY-4.0", license_url=LICENSE_URL,
                    array_sha256=_data_sha256(X, y),
                    verified_at_utc=datetime.now(timezone.utc).isoformat())
    # Keep source attribution and the verified raw-file fingerprint by its cache.
    with tempfile.NamedTemporaryFile(mode="w", dir=cache, prefix=".metadata-",
                                     encoding="utf-8", delete=False) as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(cache / f"{name}.metadata.json")
    return X, y, metadata


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _grouped_indices(X, seed):
    """Shuffle X-only equality groups, then choose nearest row-count cuts."""
    canonical = np.array(X, dtype="<f8", copy=True, order="C")
    canonical[canonical == 0] = 0.0  # +0 and -0 describe the same feature.
    canonical[np.isnan(canonical)] = np.nan  # one missing-value bit pattern
    lookup, groups, keys = {}, [], []
    group_ids = np.empty(len(X), dtype=np.int64)
    # First-occurrence ordering keeps a distinct held-out row's value changes
    # from reordering every group. Targets are not inspected or hashed here.
    for row, values in enumerate(canonical):
        key = values.tobytes()
        group = lookup.get(key)
        if group is None:
            group = len(groups)
            lookup[key] = group
            groups.append([])
            keys.append(key)
        groups[group].append(row)
        group_ids[row] = group
    count = len(groups)
    if count < 3:
        raise ValueError("a grouped train/validation/test split requires at least three distinct X groups")
    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(count)
    sizes = np.array([len(group) for group in groups], dtype=np.int64)
    cumulative = np.cumsum(sizes[order])
    # A boundary follows one or more whole groups. Reserve at least one
    # group for each later partition; ties choose the smaller cumulative count.
    first = 1 + int(np.argmin(np.abs(5 * cumulative[:count-2] - 3 * len(X))))
    second = first + 1 + int(np.argmin(np.abs(5 * cumulative[first:count-1] - 4 * len(X))))
    group_indices = dict(train=order[:first], val=order[first:second], test=order[second:])
    indices = {name: np.concatenate([np.asarray(groups[group], dtype=np.int64)
                                     for group in selected])
               for name, selected in group_indices.items()}

    def group_hash(selected):
        digest = hashlib.sha256()
        for group in selected:
            digest.update(keys[group])
        return digest.hexdigest()

    metadata = dict(
        split_strategy="exact_X_groups", group_count=count,
        group_order="first_occurrence_then_seeded_PCG64_shuffle",
        grouping_uses_targets=False, grouping_missing_values="equal_nan_mask",
        grouping_signed_zero="equal", group_size_max=int(sizes.max()),
        duplicate_group_count=int(np.count_nonzero(sizes > 1)),
        duplicate_rows_beyond_first=int(len(X)-count),
        rows_in_duplicate_groups=int(sizes[sizes > 1].sum()),
        split_group_counts={name: len(selected) for name, selected in group_indices.items()},
        split_group_sha256={name: group_hash(selected) for name, selected in group_indices.items()},
        split_boundary_rule="nearest_cumulative_row_count_to_60_and_80_percent_nonempty",
    )
    return indices, group_ids, group_indices, metadata


def prepare_split(X, y, seed):
    """Group identical raw X rows into approximate 60/20/20 partitions.

    Groups ignore targets and keep every row. Seeded group shuffling and
    nearest cumulative row-count boundaries keep each partition nonempty.
    All transforms are fitted on training rows only.
    Missing features are imputed with training means (zero for wholly missing
    training columns), then standardized using training population deviations.
    Targets must be finite and are centered/scaled on training rows. A constant
    final feature represents a regularized hidden bias. The fixed raw-output
    intercept is ``transforms['y_mean']``. Raw predictions are reconstructed as
    ``y_mean + y_scale * standardized_prediction``.
    """
    X, y = _real_array(X, "X"), _real_array(y, "y")
    if X.ndim != 2 or X.shape[0] < 3 or not X.shape[1]:
        raise ValueError("X must have at least three rows and one feature")
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector with one target per row")
    if np.isinf(X).any():
        raise ValueError("X may contain NaN for missing values, but not infinity")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    n = len(X)
    indices, group_ids, group_indices, grouping = _grouped_indices(X, seed)
    train = X[indices["train"]]
    observed = ~np.isnan(train)
    counts = observed.sum(axis=0)
    minimum = np.where(observed, train, np.inf).min(axis=0)
    maximum = np.where(observed, train, -np.inf).max(axis=0)
    constant = (counts == 0) | (minimum == maximum)
    impute = np.divide(np.where(observed, train, 0).sum(axis=0), counts,
                       out=np.zeros(X.shape[1]), where=counts > 0)
    impute[constant & (counts > 0)] = minimum[constant & (counts > 0)]
    mean = impute.copy()
    train_imputed = np.where(observed, train, impute)
    scale = np.sqrt(np.mean((train_imputed - mean)**2, axis=0))
    constant |= scale == 0
    scale[constant] = 1
    target_train = y[indices["train"]]
    target_constant = np.ptp(target_train) == 0
    y_mean = float(target_train[0] if target_constant else np.mean(target_train))
    y_scale = float(1 if target_constant else np.sqrt(np.mean((target_train-y_mean)**2)))
    if not (np.isfinite(impute).all() and np.isfinite(scale).all()
            and np.isfinite(y_mean) and np.isfinite(y_scale) and y_scale > 0):
        raise ValueError("training means or scales exceed float64 range")
    transforms = dict(x_impute=impute.tolist(), x_mean=mean.tolist(), x_scale=scale.tolist(),
                      constant_features=np.flatnonzero(constant).tolist(),
                      all_missing_features=np.flatnonzero(counts == 0).tolist(),
                      y_mean=y_mean, y_scale=y_scale, append_constant=True,
                      hidden_bias_regularized=True, standard_deviation_ddof=0)
    metadata = dict(seed=int(seed), rng="PCG64", protocol="grouped_X_60_20_20",
                    input_shape=list(X.shape), array_sha256=_data_sha256(X, y),
                    split_counts={key: len(idx) for key, idx in indices.items()},
                    split_fractions={key: len(idx) / n for key, idx in indices.items()},
                    split_sha256={key: hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()
                                  for key, idx in indices.items()})
    metadata.update(grouping)
    result = dict(indices=indices, group_ids=group_ids, group_indices=group_indices,
                  transforms=transforms, metadata=metadata)
    for key, idx in indices.items():
        raw = X[idx]
        values = (np.where(np.isnan(raw), impute, raw) - mean) / scale
        values = np.column_stack((values, np.ones(len(idx))))
        target = (y[idx] - y_mean) / y_scale
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("preprocessing produced nonfinite values")
        result[f"X_{key}"] = np.ascontiguousarray(values)
        result[f"y_{key}"] = np.ascontiguousarray(target)
    return result
