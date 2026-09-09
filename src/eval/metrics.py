"""Phase 2 metrics - the four accuracy figures plus TPT, defined once.

All four accuracy metrics are computed on the TEST portion only, in the series' own
units (requests per minute), never on scaled values. Scaled metrics are not comparable
to the paper's Table 4 and would flatter every model equally, hiding the differences we
are trying to measure.

On TPT (total prediction time). The paper's Equation 14 defines the hybrid's prediction
time as the sum of its parts - Prophet's inference time plus the network's - because
both must complete before a forecast exists. We measure it the same way: wall-clock
milliseconds to produce the entire test-set forecast, after a warm-up pass, excluding
all training. See DEVIATIONS.md D-005 for why our absolute values cannot be compared
with the paper's like-for-like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Scores:
    model: str
    mse: float
    rmse: float
    mae: float
    r2: float
    tpt_ms: float          # total ms to forecast the whole test set
    tpt_us_per_pred: float  # per-prediction cost, which is the scale-free comparison
    n_test: int
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tpt_ms: float,
    notes: str = "",
) -> Scores:
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not np.isfinite(y_pred).all():
        raise ValueError(f"{model}: prediction contains NaN/inf")

    err = y_true - y_pred
    mse = float((err ** 2).mean())

    # R^2 computed explicitly rather than via sklearn, so it is obvious that the
    # baseline being beaten is "always predict the test mean". A model can post a fine
    # RMSE and still have R^2 near zero if the series barely varies; on this series it
    # varies a lot, which is what makes R^2 ~= 0.90 a meaningful claim.
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())

    return Scores(
        model=model,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(np.abs(err).mean()),
        r2=float(1.0 - ss_res / ss_tot),
        tpt_ms=float(tpt_ms),
        tpt_us_per_pred=float(tpt_ms * 1000.0 / len(y_true)),
        n_test=int(len(y_true)),
        notes=notes,
    )


# Paper Table 4 (CLAUDE.md 5), for side-by-side reporting. Never used in any
# computation - only printed next to our numbers so the comparison is visual.
PAPER_TABLE4 = {
    "Prophet+LSTM (hybrid)":   dict(mse=63.836, rmse=7.990, mae=6.164, r2=0.900, tpt_ms=3492.0),
    "Prophet+BiLSTM (hybrid)": dict(mse=67.995, rmse=8.246, mae=6.312, r2=0.894, tpt_ms=4647.0),
    "Prophet+GRU (hybrid)":    dict(mse=75.740, rmse=8.703, mae=6.640, r2=0.882, tpt_ms=4124.0),
    "ARIMA 1-step":            dict(mse=196.288, rmse=14.010, mae=10.572, r2=0.692, tpt_ms=2300.0),
    "BiLSTM 1-step":           dict(mse=183.642, rmse=13.551, mae=10.280, r2=0.712, tpt_ms=4.3),
    "ARIMA 5-step":            dict(mse=237.604, rmse=15.414, mae=11.628, r2=0.628, tpt_ms=2488.0),
    "BiLSTM 5-step":           dict(mse=207.313, rmse=14.390, mae=10.592, r2=0.675, tpt_ms=45.1),
}

# The display order of Table 4, so our table can be read against it line by line.
MODEL_ORDER = list(PAPER_TABLE4)
