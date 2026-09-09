"""Phase 2 - the hybrid pipeline, and the trainer for every neural job.

    ./.venv/bin/python src/models/hybrid.py --job hybrid_lstm

The hybrid, in three lines:

    residual        = actual - prophet_yhat          (over the whole series)
    residual_hat    = recurrent_net(last 60 residuals)
    final_forecast  = prophet_yhat + residual_hat

Prophet supplies the part of the signal that repeats - trend, daily cycle, weekly
cycle. Subtracting it leaves the bursts and non-linear noise that a seasonality model
structurally cannot express, and that is what the network is trained on. Adding the two
back gives each model responsibility only for the component it is good at, which is the
paper's contribution.

This file also trains the two single-model baselines (Bi-LSTM 1-step and 5-step). They
run through exactly the same code path, differing only in being handed raw `y` instead
of residuals - so any gap between a hybrid and a baseline is a property of the method,
not of one of them having been implemented more carefully.

One job per process, on purpose: five trainings on a CPU take hours sequentially, and
separate processes let them run concurrently with bounded thread pools.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
RESULTS = REPO / "results"

# Must be set before TensorFlow is imported, hence the parsing dance below.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--job", required=True)
_ap.add_argument("--prophet", default="reading_a",
                 help="which Prophet config supplies the residuals")
_ap.add_argument("--threads", type=int, default=0,
                 help="cap intra/inter-op threads so jobs can run side by side")
_ap.add_argument("--epochs", type=int, default=None)
_ap.add_argument("--shuffle", action="store_true",
                 help="shuffle training WINDOWS between epochs (not the split)")
_ap.add_argument("--patience", type=int, default=None)
_ap.add_argument("--tag", default=None, help="override output name, for experiments")
_args = _ap.parse_args()

if _args.threads:
    os.environ["OMP_NUM_THREADS"] = str(_args.threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(_args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(_args.threads)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

import lstm_model as L  # noqa: E402
from dataset import load_series, make_supervised, split_at  # noqa: E402

from hybrid_jobs import JOBS  # noqa: E402


def main() -> int:
    job = _args.job
    if job not in JOBS:
        raise SystemExit(f"--job must be one of {list(JOBS)}")
    cfg = JOBS[job]

    if _args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(_args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(_args.threads)

    s = load_series()
    cut = split_at(len(s))

    prophet_yhat = None
    prophet_predict_ms = 0.0
    if cfg["source"] == "residual":
        npz = np.load(MODELS / f"prophet_{_args.prophet}.npz")
        prophet_yhat = npz["yhat"]
        target_series = npz["resid"]
        meta_all = json.loads((MODELS / "prophet_meta.json").read_text())
        prophet_predict_ms = next(
            m["predict_test_ms"] for m in meta_all if m["name"] == _args.prophet
        )
    else:
        target_series = s.to_numpy()

    data = make_supervised(target_series, s.index, horizon=cfg["horizon"])

    print(f"job={job} cell={cfg['cell']} source={cfg['source']} "
          f"horizon={cfg['horizon']}", flush=True)
    print(f"  train windows {data.x_train.shape}  test windows {data.x_test.shape}",
          flush=True)

    tag = _args.tag or job
    model, meta = L.train(
        cfg["cell"], data, tag=tag,
        epochs=_args.epochs or L.MAX_EPOCHS,
        shuffle=_args.shuffle,
        patience=_args.patience or L.PATIENCE,
    )

    pred_units, net_ms = L.predict_timed(model, data)

    if cfg["source"] == "residual":
        # Recombine. The residual windows for horizon=1 have targets covering exactly
        # the test period, so this slice lines up element-for-element; the assert below
        # is what stops a future change to the windowing silently misaligning them.
        yhat_test = prophet_yhat[cut:]
        assert len(yhat_test) == len(pred_units), (
            f"alignment: prophet {len(yhat_test)} vs net {len(pred_units)}"
        )
        final = yhat_test + pred_units
        # Equation 14: both stages must finish before a forecast exists, so their costs
        # add.
        tpt_ms = prophet_predict_ms + net_ms
        y_true = s.to_numpy()[cut:]
    else:
        final = pred_units
        tpt_ms = net_ms
        y_true = data.y_test_raw

    assert len(final) == len(y_true)

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / f"phase2_preds_{tag}.npz",
        y_true=y_true, y_pred=final,
        ds=data.t_test.values.astype("datetime64[ns]"),
        resid_pred=pred_units if cfg["source"] == "residual" else np.array([]),
        resid_true=(target_series[cut:] if cfg["source"] == "residual"
                    else np.array([])),
    )

    meta.update(
        job=job, label=cfg["label"], source=cfg["source"],
        prophet_config=_args.prophet if cfg["source"] == "residual" else None,
        prophet_predict_ms=prophet_predict_ms,
        net_predict_ms=round(net_ms, 1),
        tpt_ms=round(tpt_ms, 1),
        threads=_args.threads or "default",
    )
    (MODELS / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))

    rmse = float(np.sqrt(((y_true - final) ** 2).mean()))
    print(f"  epochs run {meta['epochs_run']}  train {meta['train_seconds']}s  "
          f"test RMSE {rmse:.3f}  TPT {tpt_ms:.0f}ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
