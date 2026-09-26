"""
Data loading, preprocessing and windowing.

Reproduces Section 6.1 of Dang-Quang & Yoo (2021):
  - Raw web-server logs are aggregated into per-minute request counts.
  - Each dataset is split 70% / 30% (chronological) into train / test.
  - FIFA World Cup 98 is min-max normalised to [-1, 1]; NASA is left in
    raw request units (Table 4 metrics are in raw units, Table 5 in [-1,1]).
  - A sliding window of the last `w` minutes (default 10) is used to predict
    the next 1 minute (single-step) or the next 5 minutes (multi-step).
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Raw-log aggregation (only needed if you want to regenerate the CSVs from the
# original .gz logs; the repo already ships the per-minute CSVs in data/).
# ----------------------------------------------------------------------------
_MONTHS = {m: f"{i:02d}" for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_TS_RE = re.compile(r"\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):\d{2}")


def aggregate_nasa_logs(gz_paths, out_csv: str | Path) -> pd.DataFrame:
    """Aggregate one or more NASA (Apache common-log) .gz files to per-minute
    request counts and write a `period,count` CSV."""
    from collections import Counter
    counter: Counter = Counter()
    for gz in gz_paths:
        with gzip.open(gz, "rt", encoding="latin-1") as fh:
            for line in fh:
                m = _TS_RE.search(line)
                if not m:
                    continue
                d, mon, y, H, M = m.groups()
                counter[f"{y}-{_MONTHS[mon]}-{d} {H}:{M}:00"] += 1
    df = (pd.DataFrame(sorted(counter.items()), columns=["period", "count"]))
    df.to_csv(out_csv, index=False)
    return df


# ----------------------------------------------------------------------------
# Loading the shipped per-minute CSVs
# ----------------------------------------------------------------------------
def load_series(csv_path: str | Path) -> pd.Series:
    """Load a `period,count` CSV as a time-indexed float series."""
    df = pd.read_csv(csv_path, parse_dates=["period"])
    df = df.sort_values("period").reset_index(drop=True)
    s = pd.Series(df["count"].astype("float64").values, index=df["period"])
    return s


# ----------------------------------------------------------------------------
# Split + scaling + windowing
# ----------------------------------------------------------------------------
def train_test_split_series(values: np.ndarray, train_frac: float = 0.70):
    """Chronological split (no shuffling)."""
    n_train = int(len(values) * train_frac)
    return values[:n_train], values[n_train:]


class MinMaxToRange:
    """Min-max scaler to an arbitrary [lo, hi] range, fit on train only.

    z = (x - min) / (max - min)          -> [0, 1]
    scaled = lo + z * (hi - lo)          -> [lo, hi]
    """

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
    """Sliding windows.

    X[i] = series[i : i+window]                       (window,)
    y[i] = series[i+window : i+window+horizon]        (horizon,)
    Returns X shaped (n, window, 1) and y shaped (n, horizon).
    """
    X, y = [], []
    n = len(series)
    for i in range(n - window - horizon + 1):
        X.append(series[i:i + window])
        y.append(series[i + window:i + window + horizon])
    X = np.asarray(X, dtype="float32").reshape(-1, window, 1)
    y = np.asarray(y, dtype="float32").reshape(-1, horizon)
    return X, y


def prepare_dataset(csv_path, dataset: str, window: int = 10, horizon: int = 1,
                    train_frac: float = 0.70):
    """End-to-end prep for one dataset/horizon.

    Returns a dict with windowed train/test tensors, the raw test target
    series (for the autoscaler), and the fitted scaler (or None).

    Normalisation follows the paper: FIFA -> [-1, 1]; NASA -> none.
    The scaler is fit on the TRAIN portion only to avoid look-ahead leakage.
    """
    s = load_series(csv_path)
    values = s.values.astype("float64")
    train_raw, test_raw = train_test_split_series(values, train_frac)

    if dataset.lower().startswith("wc") or dataset.lower().startswith("fifa"):
        scaler = MinMaxToRange(-1.0, 1.0).fit(train_raw)
        train_s = scaler.transform(train_raw)
        test_s = scaler.transform(test_raw)
    else:  # NASA: raw units
        scaler = None
        train_s, test_s = train_raw, test_raw

    X_train, y_train = make_windows(train_s, window, horizon)
    X_test, y_test = make_windows(test_s, window, horizon)

    # Index for the first target of each test window (for plotting)
    n_train = len(train_raw)
    test_index = s.index[n_train + window: n_train + window + len(y_test)]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "test_raw": test_raw, "train_raw": train_raw,
        "scaler": scaler, "test_index": test_index,
        "series": s,
    }
