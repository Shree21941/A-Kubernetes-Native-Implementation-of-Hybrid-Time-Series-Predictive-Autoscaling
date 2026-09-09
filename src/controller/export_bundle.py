"""Phase 5 - package the Phase 2 artifacts into something a container can load.

    ./.venv/bin/python src/controller/export_bundle.py

Writes models/controller_bundle/ containing everything the forecaster needs and nothing
it does not. This exists for three reasons:

1. **Size.** models/ is 246 MB, dominated by a 206 MB ARIMA pickle the controller never
   uses. The bundle is ~2 MB and can go into a container image.
2. **The scaler was never saved.** Phase 2 refits MinMaxScaler on the training residuals
   every run (src/models/dataset.py). That is fine offline, where the CSV is present, but
   the controller must not carry a 370 MB dataset just to recover two floats. They are
   extracted here, and re-derived from the same code path so they cannot drift.
3. **The Prophet lookup table.** See below - this is the interesting one.

--------------------------------------------------------------------------------
Why Prophet's output can be precomputed, and why that is not cheating
--------------------------------------------------------------------------------
Prophet's forecast is a function of the TIMESTAMP alone: trend + daily + weekly
seasonality evaluated at ds. It does not consume recent observations. The LSTM is the
only part of the hybrid that reads live data.

So for any timestamp we can name in advance - and the Phase 4 segments are frozen, so we
can name all of them - Prophet's yhat is a constant we may look up rather than compute.
The predicted values are bit-identical either way; only the cost moves.

How much this saves, measured rather than extrapolated. Phase 2 recorded Prophet's stage
at ~2.9 s, but that is 26,784 predictions in one batch (D-005). A SINGLE live Prophet
call costs **102 ms** - mostly fixed DataFrame and dispatch overhead, not per-row work -
against **0.04 ms** for a table lookup. So precomputing saves ~102 ms of a ~108 ms
forecast cycle: real, and worth taking, but nothing like the 2.9 s that a naive reading
of the Phase 2 number would suggest. An earlier version of this comment made exactly
that naive reading and claimed ~48% of the horizon; the measurement is in
results/phase5_forecaster_verify.json.

Both paths are shipped (`--prophet-mode live|table`) precisely so the claim is testable:
the controller asserts the two agree to 1e-9 at startup when both are available. If they
ever disagree, the table is stale and the run must not proceed.

This is a deployment property the paper does not discuss, and it is worth reporting: the
hybrid's headline weakness - a ~3.5 s TPT against single models' ~5 ms - is largely an
artifact of recomputing a deterministic function of the clock on every cycle.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "models")]

from dataset import load_series, split_at  # noqa: E402

MODELS = REPO / "models"
OUT = MODELS / "controller_bundle"

# The Phase 2 job whose weights the controller uses. The paper's headline model is
# Prophet+LSTM, so that is the default for reproduction fidelity - even though D-009
# found its accuracy edge over the alternatives is inside run-to-run noise. Phase 6's
# ablation arms will swap this.
DEFAULT_JOB = "hybrid_lstm_shuffled"
PROPHET_CONFIG = "reading_a"


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JOB

    meta_path = MODELS / f"{job}_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found - run Phase 2 first")
    meta = json.loads(meta_path.read_text())
    if meta.get("source") != "residual":
        raise SystemExit(
            f"job '{job}' has source={meta.get('source')!r}; the controller's hybrid "
            "path requires a residual-trained network"
        )

    keras_src = MODELS / f"{job}.keras"
    prophet_json = MODELS / f"prophet_{meta['prophet_config']}.json"
    prophet_npz = MODELS / f"prophet_{meta['prophet_config']}.npz"
    for p in (keras_src, prophet_json, prophet_npz):
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}")

    s = load_series()
    cut = split_at(len(s))
    npz = np.load(prophet_npz)
    resid = npz["resid"]
    yhat = npz["yhat"]

    if len(resid) != len(s):
        raise SystemExit(f"length mismatch: series {len(s)} vs prophet {len(resid)}")

    # Re-derive the scaler exactly as Phase 2 does: MinMaxScaler(0,1) fit on the TRAIN
    # portion of the residuals only. Two floats, but the two floats that decide whether
    # the network sees the input distribution it was trained on.
    train_resid = resid[:cut]
    resid_min = float(train_resid.min())
    resid_max = float(train_resid.max())
    if not np.isfinite([resid_min, resid_max]).all() or resid_max <= resid_min:
        raise SystemExit(f"degenerate scaler range [{resid_min}, {resid_max}]")

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(keras_src, OUT / "net.keras")
    shutil.copy2(prophet_json, OUT / "prophet.json")

    # Prophet lookup table. Only the TEST portion is exported: the controller replays
    # test-split segments (load/segments.py enforces this), so training timestamps would
    # be dead weight, and shipping them would make it possible to run the controller on
    # data the network memorised without noticing.
    ds_test = np.asarray(npz["ds"][cut:], dtype="datetime64[m]")
    np.savez_compressed(
        OUT / "prophet_table.npz",
        ds=ds_test.astype("int64"),      # minutes since epoch - compact and exact
        yhat=yhat[cut:].astype("float64"),
    )

    bundle_meta = dict(
        job=job,
        label=meta.get("label"),
        cell=meta.get("cell"),
        look_back=meta["look_back"],
        horizon=meta["horizon"],
        prophet_config=meta["prophet_config"],
        resid_scaler=dict(data_min=resid_min, data_max=resid_max,
                          note="MinMaxScaler(0,1) fit on TRAIN residuals only"),
        train_test_cut_index=int(cut),
        test_starts=str(s.index[cut]),
        table_covers=[str(s.index[cut]), str(s.index[-1])],
        seed=meta.get("seed"),
        phase2_tpt_ms=meta.get("tpt_ms"),
        phase2_prophet_predict_ms=meta.get("prophet_predict_ms"),
        source_artifacts=dict(
            net=str(keras_src.relative_to(REPO)),
            prophet=str(prophet_json.relative_to(REPO)),
        ),
    )
    (OUT / "bundle.json").write_text(json.dumps(bundle_meta, indent=2))

    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {OUT.relative_to(REPO)}/  ({size / 1e6:.2f} MB)")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<20} {p.stat().st_size / 1e6:8.3f} MB")
    print(f"\n  job              : {job}  ({meta.get('label')})")
    print(f"  look_back        : {meta['look_back']} trace-minutes")
    print(f"  residual scaler  : [{resid_min:.4f}, {resid_max:.4f}] (train only)")
    print(f"  prophet table    : {len(ds_test):,} minutes, "
          f"{bundle_meta['table_covers'][0]} -> {bundle_meta['table_covers'][1]}")
    print(f"\n  models/ total is {sum(p.stat().st_size for p in MODELS.glob('*')) / 1e6:.0f} MB; "
          f"the controller needs {size / 1e6:.2f} MB of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
