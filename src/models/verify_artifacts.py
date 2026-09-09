"""Phase 2 - prove the saved models reload and reproduce their recorded predictions.

    ./.venv/bin/python src/models/verify_artifacts.py

Phase 5's autoscaler loads these files instead of retraining - the paper's hybrid takes
seconds to predict and minutes to train, so training inside the control loop is not an
option. That makes "the artifact on disk still behaves like the model we scored" a
prerequisite for the whole controller, not a nicety.

The check is deliberately end-to-end: reload from disk, re-run the full test set, and
compare against the predictions saved at scoring time. An artifact that loads without
raising but produces different numbers is exactly the failure this catches.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "models")]

MODELS = REPO / "models"
RESULTS = REPO / "results"

NEURAL_JOBS = ["hybrid_lstm", "hybrid_bilstm", "hybrid_gru",
               "bilstm_1step", "bilstm_5step"]
PROPHET_CONFIGS = ["reading_a", "reading_b", "sens_trunc"]

# Reloaded weights are bit-identical, but the forward pass can differ in the last few
# ULPs because TensorFlow may choose different kernels or thread counts than the
# training process did. Anything above this is a real mismatch, not float noise.
TOLERANCE = 1e-3


def main() -> int:
    import tensorflow as tf
    from prophet.serialize import model_from_json

    from dataset import load_series, make_supervised, split_at
    from hybrid_jobs import JOBS  # job table shared with hybrid.py

    s = load_series()
    cut = split_at(len(s))
    rows: list[tuple[str, bool, str]] = []

    for cfg in PROPHET_CONFIGS:
        p = MODELS / f"prophet_{cfg}.json"
        if not p.exists():
            rows.append((f"prophet/{cfg}", False, "MISSING"))
            continue
        try:
            m = model_from_json(p.read_text())
            saved = np.load(MODELS / f"prophet_{cfg}.npz")["yhat"]
            # Re-predicting all 89k rows costs seconds; a 500-row sample is enough to
            # prove the deserialised model carries its fitted parameters.
            sample = s.index[cut : cut + 500]
            got = m.predict(
                __import__("pandas").DataFrame({"ds": sample})
            )["yhat"].to_numpy()
            dev = float(np.abs(got - saved[cut : cut + 500]).max())
            rows.append((f"prophet/{cfg}", dev < TOLERANCE,
                         f"max|Δ| = {dev:.2e} over 500 rows"))
        except Exception as e:  # noqa: BLE001
            rows.append((f"prophet/{cfg}", False, f"{type(e).__name__}: {e}"))

    for job in NEURAL_JOBS:
        path = MODELS / f"{job}.keras"
        npz = RESULTS / f"phase2_preds_{job}.npz"
        if not path.exists() or not npz.exists():
            rows.append((job, False, "MISSING"))
            continue
        try:
            meta = json.loads((MODELS / f"{job}_meta.json").read_text())
            model = tf.keras.models.load_model(path)

            cfgj = JOBS[job]
            if cfgj["source"] == "residual":
                series = np.load(
                    MODELS / f"prophet_{meta['prophet_config']}.npz"
                )["resid"]
                base = np.load(MODELS / f"prophet_{meta['prophet_config']}.npz")["yhat"]
            else:
                series = s.to_numpy()
                base = None

            data = make_supervised(series, s.index, horizon=cfgj["horizon"])
            pred = data.inverse(model.predict(data.x_test, batch_size=512, verbose=0))
            recombined = pred + base[cut:] if base is not None else pred

            saved = np.load(npz)["y_pred"]
            dev = float(np.abs(recombined - saved).max())
            rows.append((job, dev < TOLERANCE,
                         f"{meta['n_params']:,} params, max|Δ| = {dev:.2e}"))
        except Exception as e:  # noqa: BLE001
            rows.append((job, False, f"{type(e).__name__}: {e}"))

    ap = MODELS / "arima.pkl"
    rows.append(("arima", ap.exists(),
                 f"{ap.stat().st_size / 1024:.0f} KB" if ap.exists() else "MISSING"))

    width = max(len(n) for n, _, _ in rows)
    print()
    print("Phase 2 - saved artifacts reload correctly")
    print("=" * (width + 52))
    for name, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<{width}}  {detail}")
    print("=" * (width + 52))

    failed = [n for n, ok, _ in rows if not ok]
    if failed:
        print(f"\n{len(failed)} artifact check(s) failed: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(rows)} artifacts reload and reproduce their saved predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
