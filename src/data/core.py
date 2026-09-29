"""Data loading, preprocessing and windowing for the paper experiments."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_MONTHS = {m: f"{i:02d}" for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_TS_RE = re.compile(r"\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):\d{2}")


def aggregate_nasa_logs(gz_paths, out_csv: str | Path) -> pd.DataFrame:
    """Aggregate Apache NASA log files into per-minute request counts."""
    counter: Counter = Counter()
    for gz in gz_paths:
        with gzip.open(gz, "rt", encoding="latin-1") as fh:
            for line in fh:
                match = _TS_RE.search(line)
                if not match:
                    continue
                day, month, year, hour, minute = match.groups()
                counter[f"{year}-{_MONTHS[month]}-{day} {hour}:{minute}:00"] += 1
    df = pd.DataFrame(sorted(counter.items()), columns=["period", "count"])
    df.to_csv(out_csv, index=False)
    return df


def load_series(csv_path: str | Path) -> pd.Series:
    """Load request counts and return a continuous, zero-filled minute series.

    Accepts the repository's `period,count` and `ds,y` CSV schemas. Duplicate
    timestamps are summed before missing minutes are filled with zero requests.
    """
    df = pd.read_csv(csv_path)
    if {"period", "count"}.issubset(df.columns):
        time_col, value_col = "period", "count"
    elif {"ds", "y"}.issubset(df.columns):
        time_col, value_col = "ds", "y"
    else:
        raise ValueError(
            f"{csv_path} must contain either period,count or ds,y columns")

    df[time_col] = pd.to_datetime(df[time_col]).dt.floor("min")
    df[value_col] = pd.to_numeric(df[value_col], errors="raise")
    series = df.groupby(time_col)[value_col].sum().sort_index().astype("float64")
    minute_grid = pd.date_range(series.index.min(), series.index.max(), freq="min")
    return series.reindex(minute_grid, fill_value=0.0).rename("count")


def train_test_split_series(values: np.ndarray, train_frac: float = 0.70):
    """Chronological split (no shuffling)."""
    n_train = int(len(values) * train_frac)
    return values[:n_train], values[n_train:]


class MinMaxToRange:
    """Min-max scaler to an arbitrary [lo, hi] range, fit on train only."""

    def __init__(self, lo: float = -1.0, hi: float = 1.0):
        self.lo, self.hi = lo, hi
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray):
        self.min_ = float(np.min(x))
        self.max_ = float(np.max(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.min_) / (self.max_ - self.min_)
        return self.lo + z * (self.hi - self.lo)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        z = (y - self.lo) / (self.hi - self.lo)
        return z * (self.max_ - self.min_) + self.min_


def make_windows(series: np.ndarray, window: int = 10, horizon: int = 1):
    """Create sliding windows shaped (samples, window, 1) and targets."""
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i + window])
        y.append(series[i + window:i + window + horizon])
    X = np.asarray(X, dtype="float32").reshape(-1, window, 1)
    y = np.asarray(y, dtype="float32").reshape(-1, horizon)
    return X, y


def prepare_dataset(csv_path, dataset: str, window: int = 10, horizon: int = 1,
                    train_frac: float = 0.70):
    """Prepare chronological train/test windows and paper-space targets."""
    series = load_series(csv_path)
    values = series.values.astype("float64")
    train_raw, test_raw = train_test_split_series(values, train_frac)

    if dataset.lower().startswith(("wc", "fifa")):
        scaler = MinMaxToRange(-1.0, 1.0).fit(train_raw)
        train_scaled = scaler.transform(train_raw)
        test_scaled = scaler.transform(test_raw)
    else:
        scaler = None
        train_scaled, test_scaled = train_raw, test_raw

    X_train, y_train = make_windows(train_scaled, window, horizon)
    X_test, y_test = make_windows(test_scaled, window, horizon)
    n_train = len(train_raw)
    test_index = series.index[n_train + window:n_train + window + len(y_test)]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "test_raw": test_raw, "train_raw": train_raw,
        "scaler": scaler, "test_index": test_index,
        "series": series,
    }