"""Phase 5 - the MAPE loop.

    ./.venv/bin/python src/controller/main.py --dry-run
    ./.venv/bin/python src/controller/main.py --trace-start "1995-08-24 06:52"

Monitor -> Analyze -> Plan -> Execute, once per trace-minute, forever.

--------------------------------------------------------------------------------
The decision log is the experiment's primary evidence
--------------------------------------------------------------------------------
Every cycle appends one JSON object to decisions.jsonl. It is designed as DATA: one flat
record per decision, stable keys, no free-text-only fields, so Phase 7 can load the whole
run with pandas and Phase 8 can plot replica count against request rate without parsing
prose. Anything a later phase might want to ask - "how often did the dead-band fire",
"what was inference latency at the moment of the burst", "how far behind was readiness
when we scaled" - has to be a column here, because the run will not be repeatable once
the cluster is torn down.

--------------------------------------------------------------------------------
Trace time vs wall time in this loop
--------------------------------------------------------------------------------
The controller must label its observations with TRACE timestamps, because that is the
clock the model was trained on - Prophet's seasonality term is a function of the trace
timestamp, and feeding it wall-clock 2026 dates would ask it to extrapolate 31 years
beyond its training data. So the loop keeps a trace clock that advances one trace-minute
per cycle, anchored at --trace-start, and a wall clock used only for cooldowns and
scheduling. See D-010.

--------------------------------------------------------------------------------
Warm-up
--------------------------------------------------------------------------------
For the first `look_back` cycles the history window is not full. The controller does not
forecast; it applies a reactive rule to the observed rate instead, which is honest about
what it knows. Padding the window would hand the model 60 numbers of which most are
fiction, and it has no way to tell which.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "controller")]

from config import Config  # noqa: E402
from monitor import Monitor  # noqa: E402
from planner import Action, Planner  # noqa: E402

_stop = False


def _handle_signal(signum, _frame) -> None:
    global _stop
    _stop = True
    print(f"\nsignal {signum} - finishing this cycle then exiting", flush=True)


class DecisionLog:
    """One JSON object per line. Flushed every write - a crash mid-experiment must not
    take the evidence with it."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fh = path.open("a", buffering=1)

    def write(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log, but never patch the Deployment")
    ap.add_argument("--trace-start", default=None,
                    help="trace timestamp the run begins at, e.g. '1995-08-24 06:52'. "
                         "Must lie inside the test split the bundle covers.")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--prophet-mode", default="table", choices=["table", "live", "both"])
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = Config()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print("Phase 5 - MAPE predictive autoscaler\n")
    print(cfg.describe())
    for w in cfg.warnings():
        print(f"\n  WARNING: {w}")
    print()

    # Imported late: TensorFlow takes seconds to load and we want the config warnings on
    # screen first, so a misconfigured run can be killed before paying that cost.
    from forecaster import Forecaster

    fc = Forecaster(cfg.bundle, prophet_mode=args.prophet_mode,
                    model=cfg.forecaster)
    mon = Monitor(cfg)
    planner = Planner(cfg)

    ok, detail = mon.healthy()
    print(f"  prometheus   : {'ready' if ok else 'NOT READY'}  ({detail})")
    if not ok:
        print("  refusing to start: without Prometheus the loop has no Monitor stage.")
        return 1

    from executor import Executor
    ex = Executor(cfg, dry_run=args.dry_run)
    scale = ex.read()
    print(f"  kubernetes   : {ex.context}, {cfg.namespace}/{cfg.deployment} "
          f"spec={scale.spec_replicas} ready={scale.ready_replicas}")

    trace_start = np.datetime64(
        args.trace_start or fc.meta["table_covers"][0].replace(" ", "T"), "m")
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    log = DecisionLog(cfg.decisions_path)
    log.write(dict(kind="run_start", run_id=run_id, ts=time.time(),
                   trace_start=str(trace_start), dry_run=args.dry_run,
                   arm=f"predictive-{cfg.forecaster}",
                   prophet_mode=args.prophet_mode, config=cfg.to_dict(),
                   bundle=fc.meta, warnings=cfg.warnings()))

    print(f"  run_id       : {run_id}")
    print(f"  trace start  : {trace_start}")
    print(f"  decisions    : {cfg.decisions_path}")
    print(f"  mode         : {'DRY RUN - no patches' if args.dry_run else 'LIVE'}\n")

    cycle = 0
    t_next = time.perf_counter()
    while not _stop and (args.max_cycles == 0 or cycle < args.max_cycles):
        cycle_start = time.perf_counter()
        wall_now = time.time()
        trace_now = trace_start + np.timedelta64(cycle, "m")
        rec: dict = dict(kind="decision", run_id=run_id, cycle=cycle,
                         wall_ts=wall_now, trace_ts=str(trace_now))

        # -- Monitor ---------------------------------------------------------------
        try:
            win = mon.fetch(trace_now, now_wall=wall_now)
            rec.update(observed_rate=round(float(win.values[-1]), 3),
                       window_gaps=win.gaps, window_coverage=round(win.coverage, 4),
                       window_stale=win.stale, monitor_ms=round(win.query_ms, 2))
        except Exception as e:  # noqa: BLE001
            # A Monitor failure must not kill the loop, but it must not be treated as
            # "no traffic" either. Hold position and say so.
            rec.update(error=f"monitor: {type(e).__name__}: {e}",
                       action=Action.HOLD_WARMUP.value, target=None)
            log.write(rec)
            print(f"  [{cycle:>4}] {trace_now}  MONITOR FAILED: {e}", flush=True)
            cycle += 1
            t_next += cfg.control_period_wall_s
            _sleep_until(t_next)
            continue

        scale = ex.read()
        rec.update(spec_replicas=scale.spec_replicas,
                   ready_replicas=scale.ready_replicas)

        # -- Analyze ---------------------------------------------------------------
        warming = cycle < cfg.look_back_trace_min
        predicted = None
        if warming:
            # Reactive fallback: use the most recent observed rate as the "forecast".
            # Stated in the log as such, so no warm-up cycle is ever mistaken for a
            # prediction when the results are analysed.
            predicted = float(win.values[-1])
            rec.update(mode="warmup_reactive", predicted_rate=round(predicted, 3))
        else:
            try:
                f = fc.predict(win.values, win.timestamps, trace_now
                               + np.timedelta64(cfg.horizon_trace_min, "m"))
                predicted = f.value
                rec.update(mode="forecast", arm=f"predictive-{cfg.forecaster}",
                           predicted_rate=round(f.value, 3),
                           prophet_term=round(f.prophet_term, 3),
                           residual_term=round(f.residual_term, 3),
                           forecast_clamped_negative=f.clamped,
                           net_ms=round(f.net_ms, 3), prophet_ms=round(f.prophet_ms, 3),
                           inference_ms=round(f.total_ms, 3))
            except Exception as e:  # noqa: BLE001
                rec.update(error=f"forecast: {type(e).__name__}: {e}", mode="failed")
                log.write(rec)
                print(f"  [{cycle:>4}] {trace_now}  FORECAST FAILED: {e}", flush=True)
                cycle += 1
                t_next += cfg.control_period_wall_s
                _sleep_until(t_next)
                continue

        # -- Plan ------------------------------------------------------------------
        d = planner.plan(predicted, current=scale.spec_replicas,
                         now=time.perf_counter(),
                         warming_up=warming and cfg.warmup_policy == "hold")
        rec.update(d.to_dict())

        # -- Execute ---------------------------------------------------------------
        applied, detail = (False, "no change")
        if d.action in (Action.SCALE_UP, Action.SCALE_DOWN):
            applied, detail = ex.scale_to(d.target)
        rec.update(applied=applied, execute_detail=detail,
                   cycle_ms=round((time.perf_counter() - cycle_start) * 1000, 2))
        log.write(rec)

        flag = "*" if d.action in (Action.SCALE_UP, Action.SCALE_DOWN) else " "
        print(f"  [{cycle:>4}]{flag} {trace_now}  obs {rec['observed_rate']:>7.1f}  "
              f"pred {predicted:>7.1f}  want {d.desired_clamped:>2}  "
              f"have {scale.spec_replicas:>2}/{scale.ready_replicas:<2}  "
              f"{d.action.value:<16} {detail}", flush=True)

        cycle += 1
        t_next += cfg.control_period_wall_s
        _sleep_until(t_next)

    log.write(dict(kind="run_end", run_id=run_id, ts=time.time(), cycles=cycle))
    log.close()
    print(f"\nstopped after {cycle} cycles; decisions in {cfg.decisions_path}")
    return 0


def _sleep_until(deadline: float) -> None:
    # Absolute deadline, as in the load generator: cumulative sleeps drift, and a
    # control loop that drifts stops being aligned with the trace it is following.
    d = deadline - time.perf_counter()
    if d > 0:
        time.sleep(d)


if __name__ == "__main__":
    raise SystemExit(main())
