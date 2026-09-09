"""Phase 5 - prove the live forecaster reproduces Phase 2, and measure live TPT.

    ./.venv/bin/python src/controller/verify_forecaster.py

Two things are checked, and both must pass before the controller is worth deploying.

1. **Agreement.** The controller rebuilds the hybrid from a bundle, a different code
   path from the one that trained it: it scales with two floats from bundle.json rather
   than a live MinMaxScaler, looks Prophet up in a table rather than calling it, and
   feeds one window at a time rather than a batch of 26,784. Any of those could be
   subtly wrong in a way that still produces plausible numbers - a scaler off by a few
   percent yields forecasts that look fine and are quietly biased. So we replay real
   test windows through the controller's path and compare against the predictions Phase
   2 recorded in results/phase2_preds_*.npz.

   This is the same discipline as src/models/verify_artifacts.py, applied to the
   deployment path rather than the training path.

2. **Live TPT.** Phase 2's TPT is whole-test-set wall clock over 26,784 predictions
   (D-005). The controller cares about ONE forecast inside a 6-second cycle. That number
   has never been measured, and if it exceeds the cycle the architecture does not work
   at C=10. Measured here for both Prophet modes, because the difference between them is
   the entire argument for precomputing the Prophet term.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "models"), str(REPO / "src" / "controller")]

from config import Config  # noqa: E402
from dataset import load_series, split_at  # noqa: E402
from forecaster import Forecaster  # noqa: E402

RESULTS = REPO / "results"
N_CHECK = 200          # windows compared against Phase 2
N_TIMED = 30           # cycles timed per mode
TOLERANCE = 1e-3       # requests/minute


def main() -> int:
    cfg = Config()
    s = load_series()
    cut = split_at(len(s))
    values = s.to_numpy().astype("float64")
    index = s.index.to_numpy()

    meta = json.loads((cfg.bundle / "bundle.json").read_text())
    job = meta["job"]
    ref_path = RESULTS / f"phase2_preds_{job.replace('hybrid_', 'hybrid_')}.npz"
    if not ref_path.exists():
        raise SystemExit(f"no Phase 2 predictions at {ref_path}")
    ref = np.load(ref_path)
    ref_pred = ref["y_pred"]
    ref_ds = ref["ds"]

    print(f"bundle : {cfg.bundle.relative_to(REPO)}  (job={job}, "
          f"look_back={meta['look_back']})")
    print(f"ref    : {ref_path.relative_to(REPO)}  ({len(ref_pred):,} predictions)\n")

    fc = Forecaster(cfg.bundle, prophet_mode="table")
    lb, hz = fc.look_back, fc.horizon

    # Phase 2's prediction i corresponds to the target at series index cut + i, with the
    # input window being the lb minutes ending at cut + i - hz. Reconstructing that
    # mapping here is the point: if the controller gets the alignment wrong by one
    # minute it will still produce sensible-looking forecasts, and be wrong forever.
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ref_pred), size=min(N_CHECK, len(ref_pred)), replace=False)

    diffs, worst = [], (0.0, None)
    for i in picks:
        target_idx = cut + int(i)
        # The npz records the target timestamp for every prediction, so the alignment
        # reconstructed above can be checked against it rather than trusted.
        assert index[target_idx] == ref_ds[i].astype(index.dtype), (
            f"alignment: prediction {i} targets {ref_ds[i]} but cut+i is "
            f"{index[target_idx]}"
        )
        end = target_idx - hz + 1               # exclusive end of the input window
        if end - lb < 0:
            continue
        out = fc.predict(values[end - lb:end], index[end - lb:end], index[target_idx])
        d = abs(out.value - float(ref_pred[i]))
        diffs.append(d)
        if d > worst[0]:
            worst = (d, (i, out.value, float(ref_pred[i]), str(index[target_idx])))

    diffs = np.array(diffs)
    ok = bool(diffs.max() <= TOLERANCE)
    print(f"AGREEMENT with Phase 2 over {len(diffs)} random test windows")
    print(f"  max |diff|   : {diffs.max():.3e} req/min   (tolerance {TOLERANCE:g})")
    print(f"  mean |diff|  : {diffs.mean():.3e}")
    if worst[1]:
        i, got, want, ts = worst[1]
        print(f"  worst window : {ts}  controller {got:.6f} vs phase2 {want:.6f}")
    print(f"  verdict      : {'PASS' if ok else 'FAIL'}\n")

    # -- live TPT ------------------------------------------------------------------
    print(f"LIVE TPT - one forecast, single-threaded, median of {N_TIMED}")
    print(f"  {'mode':<10} {'prophet ms':>11} {'net ms':>9} {'total ms':>10} "
          f"{'% of 6s cycle':>14}")
    print("  " + "-" * 58)

    timings = {}
    for mode in ("table", "live"):
        try:
            f = Forecaster(cfg.bundle, prophet_mode=mode)
        except Exception as e:  # noqa: BLE001
            print(f"  {mode:<10} unavailable: {type(e).__name__}: {e}")
            continue
        end = len(values) - 1
        rows = []
        for _ in range(N_TIMED):
            o = f.predict(values[end - lb:end], index[end - lb:end], index[end])
            rows.append((o.prophet_ms, o.net_ms, o.total_ms))
        p = statistics.median(r[0] for r in rows)
        n = statistics.median(r[1] for r in rows)
        t = statistics.median(r[2] for r in rows)
        timings[mode] = dict(prophet_ms=round(p, 3), net_ms=round(n, 3),
                             total_ms=round(t, 3))
        print(f"  {mode:<10} {p:>11.2f} {n:>9.2f} {t:>10.2f} "
              f"{t / (cfg.control_period_wall_s * 1000) * 100:>13.1f}%")

    # Cross-check the table against the live model on real timestamps.
    agree = None
    if "live" in timings:
        f2 = Forecaster(cfg.bundle, prophet_mode="both")
        end = len(values) - 1
        try:
            f2.predict(values[end - lb:end], index[end - lb:end], index[end])
            agree = True
            print("\n  Prophet table == live model to 1e-9 on real timestamps: PASS")
        except ValueError as e:
            agree = False
            print(f"\n  Prophet table vs live: FAIL - {e}")

    out = dict(job=job, checked_windows=int(len(diffs)),
               max_abs_diff=float(diffs.max()), mean_abs_diff=float(diffs.mean()),
               tolerance=TOLERANCE, agreement_pass=ok,
               live_tpt=timings, prophet_table_matches_live=agree,
               control_period_wall_s=cfg.control_period_wall_s,
               phase2_whole_testset_tpt_ms=meta.get("phase2_tpt_ms"))
    (RESULTS / "phase5_forecaster_verify.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/phase5_forecaster_verify.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
