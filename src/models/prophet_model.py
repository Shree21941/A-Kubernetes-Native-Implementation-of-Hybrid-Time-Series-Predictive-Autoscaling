"""Phase 2 - Prophet, configured per the paper's Table 2.

    ./.venv/bin/python src/models/prophet_model.py

Prophet's job in the hybrid is the *predictable* part of the signal: trend plus the
daily and weekly cycles. It is fit on the training period only, then asked to predict
across the whole series, so that `residual = actual - yhat` is defined everywhere. The
recurrent network is then trained on those residuals - i.e. on precisely what Prophet
could not explain.

This script fits three configurations and saves all of them:

  reading_a   Table 2 read as daily Fourier order 20, weekly 50   (DEVIATIONS D-002)
  reading_b   Table 2 read as weekly 50, daily left at Prophet's default
  sens_trunc  reading_a, but with the July-truncation minutes withheld (D-E03)

Fitting both readings is not indecision - Table 2 lists "Weekly seasonality" twice and
never lists daily, so the paper genuinely does not say which was meant. We report both
rather than picking one silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

from dataset import DEFAULT_CSV, load_series, split_at

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# Paper Table 2. Only the seasonality Fourier orders are ambiguous; everything else is
# stated unambiguously and is used verbatim.
TABLE2 = dict(
    growth="linear",
    changepoint_prior_scale=5.1,
    seasonality_prior_scale=30.0,
    yearly_seasonality=False,   # the traces span 62 days - a yearly term is unfittable
)

# The July trace ends here; these minutes are zero-filled absence of measurement, not
# observed zero traffic. See DEVIATIONS.md D-E03.
TRUNC_FROM = pd.Timestamp("1995-07-28 13:33")
TRUNC_TO = pd.Timestamp("1995-07-31 23:59")

CONFIGS = {
    # Reading A: the duplicated row was meant to be daily. Makes the model consistent
    # with the paper's stated behaviour (its figures show Prophet tracking the daily
    # cycle, which an order-4 daily term captures only crudely).
    "reading_a": dict(daily_fourier=20, weekly_fourier=50, drop_truncation=False),
    # Reading B: both rows really are weekly, daily left at Prophet's default order 4.
    "reading_b": dict(daily_fourier=None, weekly_fourier=50, drop_truncation=False),
    # D-E03 sensitivity: reading A, with the unmeasured July minutes withheld from the
    # fit. Prophet tolerates an irregular `ds`, so they can be dropped outright rather
    # than imputed - which is the whole reason this check is cheap.
    "sens_trunc": dict(daily_fourier=20, weekly_fourier=50, drop_truncation=True),
}


def build(daily_fourier: int | None, weekly_fourier: int) -> Prophet:
    m = Prophet(
        **TABLE2,
        daily_seasonality=(True if daily_fourier is None else False),
        weekly_seasonality=False,  # added explicitly below to control the Fourier order
    )
    m.add_seasonality(name="weekly", period=7, fourier_order=weekly_fourier)
    if daily_fourier is not None:
        m.add_seasonality(name="daily", period=1, fourier_order=daily_fourier)
    return m


def fit_one(name: str, s: pd.Series, cut: int, outdir: Path) -> dict:
    cfg = CONFIGS[name]
    full = pd.DataFrame({"ds": s.index, "y": s.to_numpy()})
    train = full.iloc[:cut]

    if cfg["drop_truncation"]:
        keep = ~train["ds"].between(TRUNC_FROM, TRUNC_TO)
        train = train[keep]

    m = build(cfg["daily_fourier"], cfg["weekly_fourier"])

    t0 = time.perf_counter()
    m.fit(train)
    fit_ms = (time.perf_counter() - t0) * 1000

    # Warm-up so the timed call below measures steady-state prediction, not one-off
    # setup inside Prophet. TPT is a headline number in the paper's Table 4, so it has
    # to be measured the same way for every model.
    m.predict(full.iloc[:100][["ds"]])

    test_ds = full.iloc[cut:][["ds"]]
    t0 = time.perf_counter()
    fc_test = m.predict(test_ds)
    predict_test_ms = (time.perf_counter() - t0) * 1000

    # Full-series yhat is what the residual stage consumes.
    yhat = m.predict(full[["ds"]])["yhat"].to_numpy()
    resid = s.to_numpy() - yhat

    np.savez_compressed(
        outdir / f"prophet_{name}.npz",
        yhat=yhat, resid=resid,
        ds=s.index.values.astype("datetime64[ns]"),
    )
    (outdir / f"prophet_{name}.json").write_text(model_to_json(m))

    meta = dict(
        name=name,
        daily_fourier=cfg["daily_fourier"],
        weekly_fourier=cfg["weekly_fourier"],
        drop_truncation=cfg["drop_truncation"],
        n_train_rows=int(len(train)),
        fit_ms=round(fit_ms, 1),
        predict_test_ms=round(predict_test_ms, 1),
        n_test=int(len(test_ds)),
        # Prophet standing alone on the test set - the hybrid must improve on this,
        # otherwise the LSTM stage is adding cost for nothing.
        test_mae=float(np.abs(resid[cut:]).mean()),
        test_rmse=float(np.sqrt((resid[cut:] ** 2).mean())),
        train_rmse=float(np.sqrt((resid[:cut] ** 2).mean())),
    )
    assert len(fc_test) == len(test_ds)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--only", nargs="*", default=list(CONFIGS))
    ap.add_argument("--outdir", type=Path, default=MODELS)
    args = ap.parse_args()

    s = load_series(args.csv)
    cut = split_at(len(s))
    args.outdir.mkdir(parents=True, exist_ok=True)

    metas = []
    for name in args.only:
        print(f"fitting prophet/{name} "
              f"(daily={CONFIGS[name]['daily_fourier']}, "
              f"weekly={CONFIGS[name]['weekly_fourier']}, "
              f"drop_truncation={CONFIGS[name]['drop_truncation']}) ...", flush=True)
        meta = fit_one(name, s, cut, args.outdir)
        metas.append(meta)
        print(f"  fit {meta['fit_ms'] / 1000:.1f}s   "
              f"predict(test) {meta['predict_test_ms']:.0f}ms   "
              f"test RMSE {meta['test_rmse']:.3f}   test MAE {meta['test_mae']:.3f}",
              flush=True)

    (args.outdir / "prophet_meta.json").write_text(json.dumps(metas, indent=2))

    print()
    print("Prophet alone on the test set (lower is better)")
    print("=" * 78)
    print(f"{'config':<12} {'daily':>6} {'weekly':>7} {'train rows':>11} "
          f"{'RMSE':>8} {'MAE':>8} {'fit s':>7}")
    for m in metas:
        print(f"{m['name']:<12} {str(m['daily_fourier']):>6} {m['weekly_fourier']:>7} "
              f"{m['n_train_rows']:>11,} {m['test_rmse']:>8.3f} {m['test_mae']:>8.3f} "
              f"{m['fit_ms'] / 1000:>7.1f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
