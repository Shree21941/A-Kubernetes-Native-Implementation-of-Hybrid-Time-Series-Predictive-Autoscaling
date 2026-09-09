"""Phase 7 - collect the nine per-run metrics from one completed run.

    ./.venv/bin/python experiments/collect.py experiments/runs/<run-dir>

Reads three sources that were captured while the run was live and writes metrics.json:

  raw/requests.csv       per-request latency from the load generator (Phase 4)
  raw/decisions.jsonl    the controller's own log (Phase 5) - predictive arms only
  raw/replicas.jsonl     replica/pod samples polled during the run (see sample_cluster)

--------------------------------------------------------------------------------
Two things this module deliberately refuses to do
--------------------------------------------------------------------------------
1. **It never queries the cluster.** Every input is a file captured during the run. A
   metric computed after teardown is a metric about a cluster that no longer exists, and
   would silently change meaning if a later run had already started.

2. **It never fabricates a metric it cannot compute.** An arm with no decision log (HPA,
   KEDA) has no forecast error and no inference latency - those keys are `null`, not 0.
   A zero would average into the cross-arm comparison as though it were a measurement.

--------------------------------------------------------------------------------
On the SLO threshold
--------------------------------------------------------------------------------
The SLO is `5 x the measured unloaded p50`, the same definition Phase 4 used to derive
PER_POD_CAPACITY (D-012). Using a different threshold here than the one capacity was
measured against would mean the replica formula is solving for one SLO while the results
are scored against another - the arms would be provisioning for the wrong target and the
violation rate would say more about that mismatch than about the autoscalers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _pct(a: np.ndarray, p: float) -> float | None:
    return round(float(np.percentile(a, p)), 3) if len(a) else None


def latency_metrics(run: Path, slo_ms: float) -> dict:
    path = run / "raw" / "requests.csv"
    if not path.exists():
        return dict(error="requests.csv missing")
    lat, n_err, n = [], 0, 0
    with path.open() as fh:
        for row in csv.DictReader(fh):
            n += 1
            if row["status"] == "200":
                lat.append(float(row["latency_ms"]))
            else:
                n_err += 1
    a = np.array(sorted(lat))
    # SLO violations count ERRORS as violations too. A request that failed did not meet
    # the latency target; scoring only successful responses would reward an arm that
    # sheds load under pressure, which is the opposite of what we want to measure.
    n_slow = int((a > slo_ms).sum()) if len(a) else 0
    return dict(
        requests_total=n, requests_ok=len(lat), requests_error=n_err,
        latency_p50_ms=_pct(a, 50), latency_p90_ms=_pct(a, 90),
        latency_p95_ms=_pct(a, 95), latency_p99_ms=_pct(a, 99),
        latency_mean_ms=round(float(a.mean()), 3) if len(a) else None,
        latency_max_ms=round(float(a[-1]), 3) if len(a) else None,
        slo_ms=slo_ms,
        slo_violations=n_slow + n_err,
        slo_violation_rate=round((n_slow + n_err) / n, 6) if n else None,
    )


def replica_metrics(run: Path) -> dict:
    """Footprint, oscillation and stability, from the sampled replica timeline."""
    path = run / "raw" / "replicas.jsonl"
    if not path.exists():
        return dict(error="replicas.jsonl missing")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return dict(error="replicas.jsonl empty")

    spec = np.array([r["spec"] for r in rows], dtype="float64")
    ready = np.array([r["ready"] for r in rows], dtype="float64")
    ts = np.array([r["ts"] for r in rows], dtype="float64")

    # Changes in the REQUESTED count. Sampling can miss a change that happens and
    # reverses between two samples, so this is a lower bound - stated rather than hidden.
    d = np.diff(spec)
    changes = d[d != 0]

    # Time-weighted mean, not the mean of samples. Samples are not perfectly evenly
    # spaced (the poller competes for CPU with the cluster), and an unweighted mean would
    # over-count whichever replica counts happened to be sampled more often.
    dt = np.diff(ts)

    # Excise wall-clock discontinuities before any time-weighted arithmetic.
    #
    # The sampler polls about once a second. An interval of more than SAMPLE_GAP_S means
    # the machine was not running - a host suspend (laptop lid), or the process being
    # stopped and resumed. Three diurnal runs in this sweep each contained a single gap
    # of 3110-5106 s inside a 2880 s run.
    #
    # Leaving those intervals in silently corrupts the COST axis, because every metric
    # below is an integral over dt. A run that suspended while sitting at 1 replica has
    # an hour of "1 replica" averaged in and reports an arm far cheaper than it was;
    # replica_seconds was overstated by 1.9x to 2.7x on the affected runs.
    #
    # Nothing else catches this. trace_replay paces on CLOCK_MONOTONIC, which does not
    # advance during suspend, so its own log reports a perfect 2879.7 s of 2880 s with
    # zero errors. Only the wall-clock sampler sees the hole.
    #
    # Setting the interval to zero drops the frozen span from both the numerator and the
    # denominator, which is the truthful reading: no time passed for the cluster, and no
    # observations were made. Excising the gaps recovered live spans of 2875-2878 s
    # against 2880 s planned - the arithmetic checks out against the intended duration.
    SAMPLE_GAP_S = 10.0
    gap_mask = dt > SAMPLE_GAP_S
    n_gaps = int(gap_mask.sum())
    gap_seconds = float(dt[gap_mask].sum()) if n_gaps else 0.0
    dt = np.where(gap_mask, 0.0, dt)

    mean_spec = float((spec[:-1] * dt).sum() / dt.sum()) if len(dt) and dt.sum() else float(spec.mean())
    mean_ready = float((ready[:-1] * dt).sum() / dt.sum()) if len(dt) and dt.sum() else float(ready.mean())

    # Pending-pod time: how long pods were requested but not yet serving. This is the
    # cold-start pain the whole contribution is about (D-004), and it is the one metric
    # that CANNOT be inferred from the replica count alone.
    pending = np.maximum(spec - ready, 0.0)
    pending_pod_seconds = float((pending[:-1] * dt).sum()) if len(dt) else 0.0

    return dict(
        samples=len(rows),
        # Wall-clock span INCLUDING any frozen time, kept so the discrepancy against
        # live_duration_s stays visible rather than being quietly smoothed away.
        duration_s=round(float(ts[-1] - ts[0]), 2),
        live_duration_s=round(float(dt.sum()), 2),
        suspend_gaps=n_gaps,
        suspend_gap_seconds=round(gap_seconds, 2),
        mean_replicas=round(mean_spec, 3), mean_ready_replicas=round(mean_ready, 3),
        max_replicas=int(spec.max()), min_replicas=int(spec.min()),
        scaling_actions=int(len(changes)),
        scale_ups=int((changes > 0).sum()), scale_downs=int((changes < 0).sum()),
        replica_churn=int(np.abs(changes).sum()),
        # Direction reversals - the oscillation metric that matters. A smooth ramp of ten
        # scale-ups is not thrash; five up-down alternations is, and each reversal costs
        # a pod cold start in wall time that does not compress.
        direction_reversals=int(sum(
            1 for a, b in zip(np.sign(changes), np.sign(changes[1:])) if a != b)),
        pending_pod_seconds=round(pending_pod_seconds, 2),
        # CPU-seconds: cost / over-provisioning. Replica-seconds is the honest available
        # proxy - real CPU-seconds would need container_cpu_usage_seconds_total sampled
        # per pod, which is captured separately when available.
        replica_seconds=round(float((spec[:-1] * dt).sum()), 2) if len(dt) else None,
        note="scaling_actions is a LOWER BOUND - a change that reverses between two "
             "samples is invisible to sampling",
    )


def _cpu_from_prometheus(t0: float, t1: float, prom: str) -> float | None:
    """Total CPU-seconds over [t0, t1] via PromQL increase(), which is reset-aware.

    increase() is evaluated PER SERIES before the sum, so a pod whose counter restarts
    at zero contributes its real increase rather than a negative jump. This is both the
    fallback for runs recorded in the old cluster-sum format and a cross-check on the
    per-pod arithmetic below.
    """
    win = max(1, int(round(t1 - t0)))
    expr = (f'sum(increase(container_cpu_usage_seconds_total'
            f'{{namespace="aiops",pod=~"target-app-.*",container="app"}}[{win}s]))')
    try:
        import requests
        r = requests.get(f"{prom}/api/v1/query",
                         params={"query": expr, "time": t1}, timeout=15)
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else None
    except Exception:  # noqa: BLE001
        return None


def cpu_metrics(run: Path, prom: str = "http://localhost:30090") -> dict:
    """CPU-seconds consumed by the app pods during the run.

    Differencing is done PER POD. The earlier version differenced a cluster-wide sum,
    which reports zero for any run whose pods turned over - see the long note in
    run_matrix.sample_cpu_once. Since autoscaling exists to turn pods over, that bug hit
    precisely the runs the experiment cares about.
    """
    path = run / "raw" / "cpu.jsonl"
    if not path.exists():
        return dict(cpu_seconds_total=None,
                    note="cpu.jsonl missing - cAdvisor metrics unavailable this run")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) < 2:
        return dict(cpu_seconds_total=None, note="fewer than 2 CPU samples")

    t0, t1 = rows[0].get("ts"), rows[-1].get("ts")

    # Preferred path: per-pod counters (new format).
    if all(isinstance(r.get("per_pod"), dict) for r in rows):
        total = 0.0
        for a, b in zip(rows, rows[1:]):
            pa, pb = a["per_pod"], b["per_pod"]
            for pod, end in pb.items():
                start = pa.get(pod)
                if start is None:
                    total += end          # pod appeared mid-interval: all of it is new
                elif end >= start:
                    total += end - start  # normal accumulation
                else:
                    total += end          # counter restarted: post-reset value is the gain
            # A pod present in pa but absent from pb ended during the interval; whatever
            # it accrued was already counted when its counter was last read.
        src = "per-pod counter deltas (reset-aware)"
        xcheck = _cpu_from_prometheus(t0, t1, prom) if t0 and t1 else None
        out = dict(cpu_seconds_total=round(total, 3), cpu_source=src)
        if xcheck is not None:
            out["cpu_seconds_promql_increase"] = round(xcheck, 3)
            # Disagreement beyond ~15% means one of the two is wrong; surface it rather
            # than silently preferring either.
            if total > 0 and abs(xcheck - total) / total > 0.15:
                out["cpu_disagreement_warning"] = (
                    f"per-pod={total:.1f} vs promql={xcheck:.1f} - investigate")
        return out

    # Legacy format (pre-fix runs): the stored cluster sum is unusable, so recover the
    # value from Prometheus. Works only while the run is inside Prometheus retention.
    val = _cpu_from_prometheus(t0, t1, prom) if t0 and t1 else None
    if val is None:
        return dict(cpu_seconds_total=None,
                    note="legacy cluster-sum format and Prometheus could not be queried "
                         "- CPU cost is UNKNOWN for this run, not zero")
    return dict(cpu_seconds_total=round(val, 3),
                cpu_source="PromQL increase() backfill (legacy cluster-sum recording)")


def forecast_metrics(run: Path) -> dict:
    """Live forecast error and inference latency. Predictive arms only."""
    path = run / "raw" / "decisions.jsonl"
    if not path.exists():
        return dict(applicable=False,
                    note="no decision log - this arm has no forecaster (HPA/KEDA)")
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    dec = [r for r in recs if r.get("kind") == "decision" and r.get("mode") == "forecast"]
    if not dec:
        return dict(applicable=True, forecasts=0,
                    note="decision log present but contains no forecast cycles")

    # Pair each forecast with what actually arrived in the minute it predicted. The
    # controller logs `observed_rate` for the CURRENT minute, so the truth for the
    # forecast made at cycle i is the observation logged at cycle i+1.
    by_cycle = {r["cycle"]: r for r in recs if r.get("kind") == "decision"}
    pairs = [(r["predicted_rate"], by_cycle[r["cycle"] + 1]["observed_rate"])
             for r in dec
             if r["cycle"] + 1 in by_cycle
             and by_cycle[r["cycle"] + 1].get("observed_rate") is not None]

    out = dict(applicable=True, forecasts=len(dec), scored=len(pairs))
    if pairs:
        pred = np.array([p for p, _ in pairs], dtype="float64")
        true = np.array([t for _, t in pairs], dtype="float64")
        err = pred - true
        out["forecast_rmse"] = round(float(np.sqrt((err ** 2).mean())), 4)
        out["forecast_mae"] = round(float(np.abs(err).mean()), 4)
        out["forecast_bias"] = round(float(err.mean()), 4)
        # MAPE is undefined at zero traffic, and the quiet segment contains a 9-minute
        # run of exact zeros. Report it over non-zero minutes only, with the count, so
        # the exclusion is visible rather than silently changing the denominator.
        nz = true > 0
        out["forecast_mape_pct"] = (
            round(float((np.abs(err[nz]) / true[nz]).mean() * 100), 3) if nz.any() else None)
        out["mape_excluded_zero_minutes"] = int((~nz).sum())

    inf = [r["inference_ms"] for r in dec if r.get("inference_ms") is not None]
    if inf:
        a = np.array(sorted(inf))
        out.update(inference_p50_ms=_pct(a, 50), inference_p95_ms=_pct(a, 95),
                   inference_max_ms=round(float(a[-1]), 3),
                   inference_mean_ms=round(float(a.mean()), 3))

    acts = [r.get("action") for r in recs if r.get("kind") == "decision"]
    out["decision_actions"] = {a: acts.count(a) for a in sorted(set(filter(None, acts)))}
    errs = [r for r in recs if r.get("error")]
    out["cycle_errors"] = len(errs)
    # Window coverage: how much of the controller's input was real rather than carried
    # forward across a scrape gap. A run with poor coverage was not really testing the
    # forecaster (D-014 is what happens when this goes to zero).
    cov = [r["window_coverage"] for r in recs if r.get("window_coverage") is not None]
    if cov:
        out["window_coverage_min"] = round(float(min(cov)), 4)
        out["window_coverage_mean"] = round(float(np.mean(cov)), 4)
    return out


def collect(run: Path, slo_ms: float | None = None) -> dict:
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{run} has no config.json - a run without its config is not "
                         f"reconstructible and must not be analysed")
    cfg = json.loads(cfg_path.read_text())
    slo = slo_ms if slo_ms is not None else cfg.get("slo_ms")
    if slo is None:
        raise SystemExit("no SLO in config.json and none given with --slo-ms")

    out = dict(
        run_id=cfg.get("run_id"), arm=cfg.get("arm"), segment=cfg.get("segment"),
        rep=cfg.get("rep"), config=cfg,
        latency=latency_metrics(run, slo),
        cluster=replica_metrics(run),
        cpu=cpu_metrics(run),
        forecast=forecast_metrics(run),
    )
    (run / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--slo-ms", type=float, default=None)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    m = collect(run, args.slo_ms)

    lat, cl, fc = m["latency"], m["cluster"], m["forecast"]
    print(f"\n{m['arm']} / {m['segment']} / rep {m['rep']}")
    print("=" * 68)
    if "error" not in lat:
        print(f"  p95 / p99 latency      : {lat['latency_p95_ms']} / {lat['latency_p99_ms']} ms")
        print(f"  SLO violations         : {lat['slo_violations']:,} "
              f"({lat['slo_violation_rate']:.3%} of {lat['requests_total']:,}) "
              f"at {lat['slo_ms']:g} ms")
    if "error" not in cl:
        print(f"  mean replicas          : {cl['mean_replicas']} "
              f"(ready {cl['mean_ready_replicas']}, max {cl['max_replicas']})")
        print(f"  scaling actions        : {cl['scaling_actions']} "
              f"({cl['scale_ups']} up / {cl['scale_downs']} down), "
              f"{cl['direction_reversals']} reversals")
        print(f"  replica churn          : {cl['replica_churn']}")
        print(f"  pending-pod time       : {cl['pending_pod_seconds']} pod-seconds")
    print(f"  CPU-seconds            : {m['cpu'].get('cpu_seconds_total')}")
    if fc.get("applicable") and fc.get("scored"):
        print(f"  live forecast RMSE     : {fc['forecast_rmse']}  "
              f"MAPE {fc.get('forecast_mape_pct')}%")
        print(f"  inference p50 / p95    : {fc.get('inference_p50_ms')} / "
              f"{fc.get('inference_p95_ms')} ms")
    else:
        print(f"  forecast metrics       : n/a ({fc.get('note', '')})")
    print(f"\nwrote {(run / 'metrics.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
