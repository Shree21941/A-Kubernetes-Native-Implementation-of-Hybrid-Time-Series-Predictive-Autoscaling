"""Phase 2 data layer - the split, the sliding windows, and the scaler.

Every modelling script in this phase gets its data from here, so that the two rules
that matter most are enforced in exactly one place and cannot drift apart:

  1. The train/test split is TIME-ORDERED and never shuffled. Training on minutes that
     occur after test minutes leaks the future backwards and produces excellent,
     meaningless scores.
  2. The scaler is fit on TRAINING data only. Fitting on the full series lets the test
     set's minimum and maximum leak into the training transform - a subtler leak that
     is easy to miss because nothing about the code looks wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "data" / "processed" / "nasa_requests_per_min.csv"

# The paper's look-back: use the previous hour of history to predict the next minute.
LOOK_BACK = 60

# 70/30, expressed as integer arithmetic rather than int(n * 0.7). Floating point makes
# that expression lie: 89280 * 0.7 == 62495.999999999993, so int() silently yields
# 62495 instead of 62496. A one-minute difference is harmless in itself, but a split
# boundary that depends on float rounding is not reproducible, and reproducibility is
# the entire point of pinning seeds elsewhere in this phase.
TRAIN_PCT = 70


def load_series(csv: Path = DEFAULT_CSV) -> pd.Series:
    """Load the Phase 1 per-minute series, with its invariants re-checked.

    These are the same checks src/data/explore.py runs. They are repeated here because
    this is the door every model walks through, and a silently regenerated or
    hand-edited CSV would otherwise reach the models unexamined.
    """
    df = pd.read_csv(csv, parse_dates=["ds"])
    s = df.set_index("ds")["y"].astype("float64")

    expected = pd.date_range(s.index[0], s.index[-1], freq="min")
    if not s.index.equals(expected):
        raise ValueError(
            f"{csv} is not a continuous 1-minute grid "
            f"({len(s):,} rows vs {len(expected):,} expected) - re-run "
            f"src/data/preprocess_nasa.py"
        )
    if s.index.tz is not None:
        raise ValueError("series index must be timezone-naive (see Phase 1 notes)")
    if s.isna().any():
        raise ValueError("series contains NaN - preprocess_nasa.py fills gaps with 0")
    return s


def split_at(n: int) -> int:
    """Index of the first TEST sample. Everything before it is training."""
    return n * TRAIN_PCT // 100


@dataclass
class Supervised:
    """Sliding-window arrays ready for Keras, plus everything needed to undo them."""

    x_train: np.ndarray      # (n_train, look_back, 1) scaled
    y_train: np.ndarray      # (n_train, 1) scaled
    x_test: np.ndarray       # (n_test, look_back, 1) scaled
    y_test: np.ndarray       # (n_test, 1) scaled
    t_test: pd.DatetimeIndex  # timestamp of each test TARGET
    y_test_raw: np.ndarray   # (n_test,) unscaled targets - what metrics compare against
    scaler: MinMaxScaler
    look_back: int
    horizon: int

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        """Map network output back to the series' own units (req/min or residual)."""
        return self.scaler.inverse_transform(
            np.asarray(scaled).reshape(-1, 1)
        ).ravel()


def make_supervised(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    look_back: int = LOOK_BACK,
    horizon: int = 1,
) -> Supervised:
    """Build look-back windows over `values`, split time-ordered, scaled on train only.

    Indexing convention. To predict the value at target index `t` with a forecast
    horizon of `h`, the most recent observation we are allowed to see is at `t - h`.
    The input window is therefore the `look_back` values ending there:

        window = values[t - h - look_back + 1 : t - h + 1]     target = values[t]

    For h=1, look_back=60 this is the familiar `values[t-60:t] -> values[t]`. For h=5
    the window stops five minutes short of the target, which is what makes it a genuine
    5-step-ahead forecast rather than a 1-step forecast wearing a different label.

    A window belonging to a TEST sample may legitimately draw some of its inputs from
    the tail of the training period - that is just "using recent history", exactly what
    the deployed controller will do at inference time. What must never happen is the
    reverse, and it cannot here: a sample is assigned to test purely by its target
    index, and every input it uses is strictly earlier in time than that target.
    """
    values = np.asarray(values, dtype="float64")
    n = len(values)
    cut = split_at(n)

    first_t = horizon + look_back - 1  # earliest target with a full window behind it
    targets = np.arange(first_t, n)

    # Vectorised window construction: row i is values[origins[i]-look_back+1 : +1].
    origins = targets - horizon
    offsets = np.arange(-look_back + 1, 1)
    windows = values[origins[:, None] + offsets[None, :]]
    y = values[targets]

    is_test = targets >= cut

    # Fit on the training PERIOD, not merely on the training windows. These are almost
    # the same set of numbers, but "everything strictly before the split boundary" is
    # the honest definition of what was observable at training time.
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(values[:cut].reshape(-1, 1))

    def scale(a: np.ndarray) -> np.ndarray:
        return scaler.transform(a.reshape(-1, 1)).reshape(a.shape)

    return Supervised(
        x_train=scale(windows[~is_test])[..., None].astype("float32"),
        y_train=scale(y[~is_test]).reshape(-1, 1).astype("float32"),
        x_test=scale(windows[is_test])[..., None].astype("float32"),
        y_test=scale(y[is_test]).reshape(-1, 1).astype("float32"),
        t_test=index[targets[is_test]],
        y_test_raw=y[is_test],
        scaler=scaler,
        look_back=look_back,
        horizon=horizon,
    )


def describe_split(s: pd.Series) -> str:
    cut = split_at(len(s))
    tr, te = s.iloc[:cut], s.iloc[cut:]
    return (
        f"70/30 time-ordered split at index {cut:,} ({s.index[cut]})\n"
        f"  TRAIN {len(tr):>7,} min  {tr.index[0]} -> {tr.index[-1]}  "
        f"mean {tr.mean():6.2f}  zeros {(tr == 0).mean():.2%}\n"
        f"  TEST  {len(te):>7,} min  {te.index[0]} -> {te.index[-1]}  "
        f"mean {te.mean():6.2f}  zeros {(te == 0).mean():.2%}"
    )


if __name__ == "__main__":
    s = load_series()
    print(describe_split(s))
    d = make_supervised(s.to_numpy(), s.index)
    print(f"\nwindows (look_back={d.look_back}, horizon={d.horizon}):")
    print(f"  x_train {d.x_train.shape}   y_train {d.y_train.shape}")
    print(f"  x_test  {d.x_test.shape}   y_test  {d.y_test.shape}")
    print(f"  scaler fit on train only: data_min={d.scaler.data_min_[0]:.1f} "
          f"data_max={d.scaler.data_max_[0]:.1f}")
    print(f"  first test target {d.t_test[0]}, last {d.t_test[-1]}")
