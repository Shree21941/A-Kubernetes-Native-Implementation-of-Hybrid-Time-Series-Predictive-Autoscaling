"""Phase 2 - re-measure TPT serially, with nothing else running.

    ./.venv/bin/python src/eval/measure_tpt.py

Why this exists. The training runs were executed concurrently (three to four jobs at a
time on 12 logical CPUs) so that five CPU trainings would finish in hours rather than
most of a day. Accuracy metrics are unaffected by that - the weights do not care what
else is on the machine - but the prediction timings recorded alongside them were taken
under contention and are not comparable between models.

TPT is one of the paper's three headline claims, so it gets its own measurement pass:
every model reloaded from disk and timed one at a time, warm-up first, nothing else
running. Results go to results/phase2_tpt.json, which compare.py prefers over the
timings captured during training.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "models")]

MODELS = REPO / "models"
RESULTS = REPO / "results"

VARIANTS = ["", "_shuffled"]
NEURAL = ["hybrid_lstm", "hybrid_bilstm", "hybrid_gru", "bilstm_1step", "bilstm_5step"]
REPEATS = 3  # take the median, so one scheduler hiccup cannot set the headline number


def main() -> int:
    import tensorflow as tf

    from dataset import load_series, make_supervised, split_at
    from hybrid_jobs import JOBS

    # Single-threaded: a per-forecast latency figure that silently depends on how many
    # cores happened to be free is not a property of the model.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    s = load_series()
    cut = split_at(len(s))
    out: dict[str, float] = {}

    # --- Prophet ----------------------------------------------------------
    from prophet.serialize import model_from_json
    import pandas as pd

    test_ds = pd.DataFrame({"ds": s.index[cut:]})
    for cfg in ["reading_a"]:
        m = model_from_json((MODELS / f"prophet_{cfg}.json").read_text())
        m.predict(test_ds.iloc[:100])  # warm-up
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            m.predict(test_ds)
            times.append((time.perf_counter() - t0) * 1000)
        out[f"prophet_{cfg}"] = float(np.median(times))
        print(f"  prophet/{cfg:<12} {out[f'prophet_{cfg}']:9.1f} ms", flush=True)

    # --- Neural nets ------------------------------------------------------
    for variant in VARIANTS:
        for job in NEURAL:
            path = MODELS / f"{job}{variant}.keras"
            if not path.exists():
                continue
            cfg = JOBS[job]
            series = (np.load(MODELS / "prophet_reading_a.npz")["resid"]
                      if cfg["source"] == "residual" else s.to_numpy())
            data = make_supervised(series, s.index, horizon=cfg["horizon"])
            model = tf.keras.models.load_model(path)

            model.predict(data.x_test[:512], batch_size=512, verbose=0)  # warm-up
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                model.predict(data.x_test, batch_size=512, verbose=0)
                times.append((time.perf_counter() - t0) * 1000)
            net_ms = float(np.median(times))

            # Equation 14: for a hybrid both stages must complete before a forecast
            # exists, so Prophet's inference cost is part of every prediction.
            total = net_ms + (out["prophet_reading_a"]
                              if cfg["source"] == "residual" else 0.0)
            out[f"{job}{variant}"] = total
            print(f"  {job + variant:<24} net {net_ms:8.1f} ms   total {total:9.1f} ms",
                  flush=True)
            del model
            tf.keras.backend.clear_session()

    # --- ARIMA ------------------------------------------------------------
    import warnings

    warnings.filterwarnings("ignore")
    from statsmodels.tsa.arima.model import ARIMAResults

    from arima_model import rolling_forecast

    res = ARIMAResults.load(MODELS / "arima.pkl")
    y = s.to_numpy()
    for h in (1, 5):
        _, ms = rolling_forecast(res, y, cut, h)
        out[f"arima_{h}step"] = float(ms)
        print(f"  arima_{h}step{'':<16} {ms:9.1f} ms", flush=True)

    (RESULTS / "phase2_tpt.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/phase2_tpt.json ({len(out)} entries), "
          f"single-threaded, median of {REPEATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
