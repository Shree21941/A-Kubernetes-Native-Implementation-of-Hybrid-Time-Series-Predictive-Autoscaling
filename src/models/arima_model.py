"""Phase 2 - the ARIMA baselines, at 1-step and 5-step horizons.

    ./.venv/bin/python src/models/arima_model.py

The paper reports ARIMA 1-step and 5-step rows in Table 4 but never states the order,
so we select it by AIC over a small grid and record the choice (DEVIATIONS.md D-007).

How the rolling forecast avoids leaking. The model's PARAMETERS are estimated once, on
the training period only, and are then frozen - `apply(..., refit=False)` re-runs the
Kalman filter over new data using the existing coefficients rather than re-estimating
them. The loop then feeds actual observations in one at a time and asks for a forecast
`h` steps ahead. So the model conditions on the observed past, which is exactly what the
LSTM does through its look-back window, and never sees the value it is predicting.

Why ARIMA is expected to lose. It has no seasonal terms, so it cannot represent the
daily cycle that dominates this workload - it can only extrapolate the last few minutes.
That is the point of including it: it is the classical baseline the hybrid must beat.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from dataset import load_series, split_at

warnings.filterwarnings("ignore")  # statsmodels is loud about convergence on long series

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
RESULTS = REPO / "results"

CANDIDATE_ORDERS = [
    (1, 0, 0), (2, 0, 0), (5, 0, 0), (1, 0, 1),
    (2, 0, 2), (1, 1, 1), (2, 1, 2), (5, 1, 0),
]
# Order selection runs on a trailing slice of TRAIN, not the whole of it: fitting eight
# candidates on 62 k points would cost minutes for a decision that a recent,
# representative window settles just as well. The chosen order is then re-fit on the
# full training period.
SELECT_WINDOW = 20_000

HORIZONS = {1: "ARIMA 1-step", 5: "ARIMA 5-step"}


def select_order(train: np.ndarray) -> tuple[tuple[int, int, int], list[dict]]:
    sub = train[-SELECT_WINDOW:]
    table = []
    for order in CANDIDATE_ORDERS:
        try:
            r = ARIMA(sub, order=order, trend="n").fit()
            table.append(dict(order=list(order), aic=float(r.aic), bic=float(r.bic)))
        except Exception as e:  # noqa: BLE001
            table.append(dict(order=list(order), aic=None, error=str(e)))
    ok = [t for t in table if t["aic"] is not None]
    best = min(ok, key=lambda t: t["aic"])
    return tuple(best["order"]), table


def rolling_forecast(res, y: np.ndarray, cut: int, horizon: int) -> tuple[np.ndarray, float]:
    """h-step-ahead forecast for every test target. Returns (predictions, elapsed ms).

    At loop position `p` the filter has consumed y[0..p], so `forecast(h)[h-1]` is the
    prediction for y[p+h] made using information available at time p - a genuine
    h-step-ahead forecast, not a 1-step forecast relabelled.
    """
    n = len(y)
    preds = np.empty(n - cut, dtype="float64")

    # Seed the filter with everything up to the first forecast origin.
    roll = res.apply(y[: cut - horizon + 1], refit=False)

    t0 = time.perf_counter()
    for p in range(cut - horizon, n - horizon):
        preds[p + horizon - cut] = roll.forecast(horizon)[horizon - 1]
        roll = roll.extend(y[p + 1 : p + 2])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return preds, elapsed_ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", type=int, nargs=3, default=None,
                    help="override AIC selection, e.g. --order 2 1 2")
    args = ap.parse_args()

    s = load_series()
    y = s.to_numpy()
    cut = split_at(len(y))
    train = y[:cut]

    if args.order:
        order, table = tuple(args.order), []
        print(f"using supplied order ARIMA{order}")
    else:
        print(f"selecting order by AIC on the last {SELECT_WINDOW:,} training "
              f"minutes ...", flush=True)
        order, table = select_order(train)
        for t in sorted((x for x in table if x["aic"] is not None),
                        key=lambda x: x["aic"]):
            print(f"  ARIMA{tuple(t['order'])}  AIC={t['aic']:12.1f}  "
                  f"BIC={t['bic']:12.1f}")
        print(f"  selected ARIMA{order}")

    print(f"\nfitting ARIMA{order} on {len(train):,} training minutes ...", flush=True)
    t0 = time.perf_counter()
    res = ARIMA(train, order=order, trend="n").fit()
    fit_s = time.perf_counter() - t0
    print(f"  fit in {fit_s:.1f}s   AIC={res.aic:.1f}", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.save(MODELS / "arima.pkl")

    metas = []
    for h, label in HORIZONS.items():
        print(f"\nrolling {h}-step forecast over {len(y) - cut:,} test minutes ...",
              flush=True)
        preds, ms = rolling_forecast(res, y, cut, h)
        y_true = y[cut:]
        rmse = float(np.sqrt(((y_true - preds) ** 2).mean()))
        print(f"  done in {ms / 1000:.1f}s   test RMSE {rmse:.3f}", flush=True)

        job = f"arima_{h}step"
        np.savez_compressed(
            RESULTS / f"phase2_preds_{job}.npz",
            y_true=y_true, y_pred=preds,
            ds=s.index[cut:].values.astype("datetime64[ns]"),
            resid_pred=np.array([]), resid_true=np.array([]),
        )
        metas.append(dict(
            job=job, label=label, order=list(order), horizon=h,
            fit_seconds=round(fit_s, 1), aic=float(res.aic),
            tpt_ms=round(ms, 1), n_test=int(len(y_true)),
            selection_window=SELECT_WINDOW, selection_table=table,
        ))
        (MODELS / f"{job}_meta.json").write_text(json.dumps(metas[-1], indent=2))

    print("\nARIMA baselines complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
