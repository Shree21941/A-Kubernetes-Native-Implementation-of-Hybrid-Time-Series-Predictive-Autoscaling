"""Phase 7 - aggregate the matrix into mean +/- sd, and run the DoD sanity pass.

    ./.venv/bin/python experiments/aggregate.py
    ./.venv/bin/python experiments/aggregate.py --csv results/phase7_matrix.csv

Single runs are noise: container scheduling, Docker jitter and thermal drift all move the
numbers. This reduces each (arm, segment) cell over its repetitions and reports the
spread alongside the centre, because a difference smaller than its own standard deviation
is not a difference.

--------------------------------------------------------------------------------
The sanity pass, and why it is a gate rather than a report
--------------------------------------------------------------------------------
The Phase 7 Definition of Done says: "a quick sanity pass shows plausible results
(predictive should not be worse than HPA on burst latency; if it is, investigate before
analysis)."

That is the right instinct and it is worth being precise about what it can and cannot do.
It is a check for BROKEN PLUMBING, not a hypothesis test. The plausible failures it
catches are things like an arm whose metric returned <unknown> so it never scaled, a
controller that sat in warm-up for the whole run, or two autoscalers fighting. Those
produce results that are wrong in obvious directions.

What it must NOT become is a filter that discards results for disagreeing with our
expectations. If the predictive arm is genuinely worse on burst latency and the plumbing
is sound, that is a finding - and given Phase 2 (D-009: the hybrid's edge is inside noise)
and Phase 5 (the simulated controller FOLLOWED the load rather than leading it), it is a
live possibility rather than a hypothetical. So a failed sanity check prints an
INVESTIGATE banner with the specific things to check; it never deletes or hides a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "experiments" / "runs"
RESULTS = REPO / "results"

ARMS = ["hpa-cpu", "keda-rate", "predictive-prophet", "predictive-hybrid"]
SEGMENTS = ["diurnal", "burst", "quiet"]

# (label, path into metrics.json, format, lower_is_better)
METRICS = [
    ("p95 latency (ms)",      ("latency", "latency_p95_ms"),        "{:.1f}", True),
    ("p99 latency (ms)",      ("latency", "latency_p99_ms"),        "{:.1f}", True),
    ("SLO violation %",       ("latency", "slo_violation_rate"),    "{:.3%}", True),
    ("mean replicas",         ("cluster", "mean_replicas"),         "{:.2f}", True),
    ("scaling actions",       ("cluster", "scaling_actions"),       "{:.1f}", True),
    ("replica churn",         ("cluster", "replica_churn"),         "{:.1f}", True),
    ("direction reversals",   ("cluster", "direction_reversals"),   "{:.1f}", True),
    ("pending-pod seconds",   ("cluster", "pending_pod_seconds"),   "{:.1f}", True),
    ("CPU-seconds",           ("cpu", "cpu_seconds_total"),         "{:.1f}", True),
    ("live forecast RMSE",    ("forecast", "forecast_rmse"),        "{:.2f}", True),
    ("live forecast MAPE %",  ("forecast", "forecast_mape_pct"),    "{:.1f}", True),
    ("inference p50 (ms)",    ("forecast", "inference_p50_ms"),     "{:.2f}", True),
]


def load_all() -> list[dict]:
    out = []
    for d in sorted(RUNS.glob("*")):
        f = d / "metrics.json"
        if f.exists():
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                print(f"  WARNING: {d.name}/metrics.json is not valid JSON - skipped")
    return out


def dig(m: dict, path: tuple[str, str]):
    v = m.get(path[0], {})
    return v.get(path[1]) if isinstance(v, dict) else None


def cell(runs: list[dict], path: tuple[str, str]) -> tuple[float | None, float | None, int]:
    vals = [dig(m, path) for m in runs]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd, len(vals)


def sanity(by: dict) -> list[tuple[str, bool, str]]:
    """Plumbing checks. Each returns (name, ok, detail)."""
    out = []

    # 1. Completeness. Anything below 5 reps is not yet a result.
    for a in ARMS:
        for s in SEGMENTS:
            n = len(by.get((a, s), []))
            out.append((f"{a}/{s}: 5 repetitions present", n >= 5, f"n={n}"))

    # 2. Every arm actually scaled. An arm pinned at min_replicas for a whole run did not
    #    make a decision; it failed to see its metric. This is the metrics-server /
    #    KEDA-APIService silent failure that CLAUDE.md 3 warns about, and the D-014
    #    empty-rate() failure, both of which look like "chose not to scale".
    for a in ARMS:
        for s in ("diurnal", "burst"):     # quiet may legitimately need few changes
            runs = by.get((a, s), [])
            if not runs:
                continue
            mx = [dig(m, ("cluster", "max_replicas")) for m in runs]
            mx = [v for v in mx if v is not None]
            out.append((f"{a}/{s}: scaled above min_replicas at least once",
                        bool(mx) and max(mx) > 1,
                        f"max replicas observed across reps: {max(mx) if mx else 'n/a'}"))

    # 3. Predictive arms actually forecast, rather than spending the run in warm-up.
    for a in ("predictive-prophet", "predictive-hybrid"):
        for s in SEGMENTS:
            runs = by.get((a, s), [])
            if not runs:
                continue
            sc = [dig(m, ("forecast", "scored")) or 0 for m in runs]
            out.append((f"{a}/{s}: produced scored forecasts",
                        min(sc) > 0 if sc else False,
                        f"scored forecasts per rep: {sc}"))

    # 4. The controller's Monitor stage saw real data. Poor coverage means it was reading
    #    carried-forward values, so the run tested the gap-filling, not the forecaster.
    for a in ("predictive-prophet", "predictive-hybrid"):
        for s in SEGMENTS:
            runs = by.get((a, s), [])
            cov = [dig(m, ("forecast", "window_coverage_mean")) for m in runs]
            cov = [c for c in cov if c is not None]
            if cov:
                out.append((f"{a}/{s}: window coverage >= 0.95", min(cov) >= 0.95,
                            f"min mean coverage {min(cov):.3f}"))

    # 4b. Pod-scheduling stalls, which masquerade as autoscaler failure.
    #
    # Found at run 26 of the sweep: predictive-prophet/diurnal/rep3 recorded 53.8% SLO
    # violations and 6,741 timeouts, while its rep1 and rep5 siblings recorded 1.6% and
    # 2.3% with zero errors. The controller was healthy throughout - cycle_errors 0, and
    # it issued 8 scale-ups asking for up to 6 replicas. What differed was
    # pending_pod_seconds: 299.9 s against 12.6 s and 13.6 s.
    #
    # The pods were REQUESTED and could not START. An arm cannot serve traffic on pods
    # the scheduler has not placed, so the resulting latency measures the host's
    # scheduling capacity, not the forecaster's quality. Left unflagged this lands in the
    # results table as "Prophet-only collapses on diurnal load" - a plausible, publishable,
    # WRONG conclusion.
    #
    # Flagged relative to the observed distribution rather than an absolute threshold,
    # because what counts as a stall depends on the machine the sweep ran on. Across 37
    # runs: median 18.9 s, p90 68.3 s, worst-but-one 110.5 s. A run beyond 4x the median
    # AND above 150 s is a stall, not a busy cluster.
    #
    # This does NOT discard the run - CLAUDE.md 8: a failing check is a reason to look,
    # not a reason to drop data. It marks the run so the analysis reports it explicitly
    # and can quote statistics with and without it.
    pend = [(dig(m, ("cluster", "pending_pod_seconds")), m) for ms in by.values() for m in ms]
    pend = [(v, m) for v, m in pend if v is not None]
    if len(pend) >= 5:
        vals = sorted(v for v, _ in pend)
        med = vals[len(vals) // 2]
        stalled = [(v, m) for v, m in pend if v > max(150.0, 4.0 * med)]
        detail = (f"median {med:.1f}s; " + (
            ", ".join(f"{m['arm']}/{m['segment']}/rep{m['rep']}={v:.0f}s"
                      for v, m in sorted(stalled, reverse=True))
            if stalled else "no run beyond 4x median"))
        out.append(("no run suffered a pod-scheduling stall", not stalled, detail))

    # 4c. Host-suspend gaps, which are invisible to every other check.
    #
    # Found after a laptop lid-close: three diurnal runs recorded sampler spans of 7981 s,
    # 6917 s and 5988 s against a planned 2880 s, each from ONE gap of 3110-5106 s sitting
    # in the middle of the run.
    #
    # Nothing else notices. trace_replay paces on CLOCK_MONOTONIC, which does not advance
    # while the machine is suspended, so it reported 2879.7 s against 2880 s planned, with
    # zero errors and a normal shape correlation. The run looks flawless in its own log.
    # The sampler uses wall-clock time.time(), which keeps counting through a suspend -
    # hence the hole.
    #
    # The damage is to the COST axis. mean_replicas and replica_seconds are integrated
    # over sampler timestamps, so a frozen hour is averaged in at whatever replica count
    # happened to be current when the lid shut. A run that suspended while sitting at 1
    # replica reports a far cheaper arm than it was.
    #
    # The criterion is deliberately RESULT-INDEPENDENT: a wall-clock discontinuity is a
    # fact about the machine, not about how the arm performed. That is what separates
    # this from discarding a run for looking bad - contrast the prophet/diurnal/rep3
    # scheduling stall, which is flagged but KEPT because its slowness is in-cluster
    # behaviour and arguably part of the system under test.
    for (a, s), ms in sorted(by.items()):
        for m in ms:
            planned = (m["config"]["segment_minutes"] * 60.0
                       / m["config"]["compression"])
            actual = dig(m, ("cluster", "duration_s"))
            if actual is None:
                continue
            drift = actual - planned
            out.append((f"{a}/{s}/rep{m['rep']}: no host-suspend gap",
                        drift <= 60.0,
                        f"sampler span {actual:.0f}s vs planned {planned:.0f}s "
                        f"(drift {drift:+.0f}s)"))

    # 5. The DoD's own check, stated as the spec words it.
    for pred in ("predictive-hybrid", "predictive-prophet"):
        p = cell(by.get((pred, "burst"), []), ("latency", "latency_p95_ms"))
        h = cell(by.get(("hpa-cpu", "burst"), []), ("latency", "latency_p95_ms"))
        if p[0] is not None and h[0] is not None:
            # Compared against HPA's mean plus one sd: "worse" should mean worse beyond
            # the noise, not worse by a hair on a single noisy cell.
            ok = p[0] <= h[0] + (h[1] or 0.0)
            out.append((f"{pred} not worse than hpa-cpu on burst p95",
                        ok, f"{p[0]:.1f}+/-{p[1]:.1f} vs hpa {h[0]:.1f}+/-{h[1]:.1f} ms"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    runs = load_all()
    if not runs:
        print("\nNo completed runs found in experiments/runs/.")
        print("Phase 7 has not been executed - there is nothing to aggregate.")
        print("Run ./experiments/run_matrix.py once a cluster exists.\n")
        return 1

    by: dict[tuple[str, str], list[dict]] = {}
    for m in runs:
        by.setdefault((m.get("arm"), m.get("segment")), []).append(m)

    print(f"\nPhase 7 - {len(runs)} completed runs\n")
    for seg in SEGMENTS:
        present = [a for a in ARMS if by.get((a, seg))]
        if not present:
            continue
        print(f"segment: {seg}")
        print("=" * (26 + 22 * len(present)))
        print(f"  {'metric':<24}" + "".join(f"{a:>22}" for a in present))
        print("  " + "-" * (24 + 22 * len(present)))
        for label, path, fmt, _ in METRICS:
            cells = [cell(by[(a, seg)], path) for a in present]
            if all(c[0] is None for c in cells):
                continue
            row = f"  {label:<24}"
            for mean, sd, n in cells:
                row += ("{:>22}".format("n/a") if mean is None
                        else "{:>22}".format(f"{fmt.format(mean)} ± {fmt.format(sd)}"))
            print(row)
        print()

    print("\nSanity pass")
    print("=" * 88)
    checks = sanity(by)
    width = max(len(n) for n, _, _ in checks)
    n_fail = 0
    for name, ok, detail in checks:
        if not ok:
            n_fail += 1
        print(f"  {'PASS' if ok else 'CHECK':<6} {name:<{width}}  {detail}")
    print("=" * 88)
    print(f"  {len(checks) - n_fail}/{len(checks)} passed")
    if n_fail:
        print("\n  INVESTIGATE before analysing. Likely causes, in order of frequency:")
        print("    - two autoscalers active   -> ./experiments/switch_arm.sh --status")
        print("    - metric never resolved    -> kubectl describe hpa; kubectl top pods")
        print("    - KEDA APIService down     -> kubectl get apiservice v1beta1.external.metrics.k8s.io")
        print("    - scrape window empty      -> D-014; check the 1.5s scrape interval")
        print("  A failing check is a reason to look, NOT a reason to discard a run.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            heads = [l for l, _, _, _ in METRICS]
            w.writerow(["arm", "segment", "rep", "run_id"] + heads)
            for m in runs:
                w.writerow([m.get("arm"), m.get("segment"), m.get("rep"), m.get("run_id")]
                           + [dig(m, p) for _, p, _, _ in METRICS])
        print(f"\nwrote {args.csv.relative_to(REPO)} ({len(runs)} rows, one per run)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
