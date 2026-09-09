"""Phase 4 - measure PER_POD_CAPACITY: the request rate one pod can serve inside SLO.

    # against a CPU-limited local process (no cluster needed)
    ./.venv/bin/python load/calibrate.py --spawn

    # against the real cluster, deployment pinned to 1 replica
    ./.venv/bin/python load/calibrate.py --url http://localhost:30080/

This number is the denominator of the replica formula that Phase 5 implements:

    desired_replicas = ceil(predicted_request_rate / PER_POD_CAPACITY)

so guessing it would make every replica decision in Phases 5-8 wrong by a constant
factor, in a way that looks like a forecasting error rather than a calibration one.
It has to be measured.

Method. Offer a constant rate for a fixed hold, discard a warm-up prefix, record p95
latency over the remainder, then step the rate up. Capacity is the highest offered rate
whose p95 stays under the SLO; the crossing point is linearly interpolated between the
last passing step and the first failing one.

The load is open-loop, as in trace_replay.py, and for the same reason: a closed-loop
ramp would throttle itself as the server slowed and would never find the knee at all -
it would report the rate at which it gave up, not the rate at which the server did.

On the SLO. The app burns ~100 ms of CPU per request, and a pod is limited to 500m, so
a single request takes ~200 ms of wall time even with the pod completely idle. The
default p95 target of 500 ms is 2.5x that floor: high enough that it is not measuring
the service time itself, low enough to be crossed well before the queue runs away.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "load")]

from trace_replay import run as openloop_run  # noqa: E402

RESULTS = REPO / "results"

# Matches the CPU limit in k8s/app-deployment.yaml. Emulated locally with a systemd
# user scope so that a calibration taken without a cluster is comparable to one taken
# with it - cgroup v2 throttles the process the same way the kubelet throttles the pod.
CPU_QUOTA_PCT = 50           # 50% of one core == 500m
# The SLO is DERIVED from measured unloaded latency, not asserted as a constant.
# A fixed millisecond target silently encodes an assumption about service time: our
# first attempt used 500 ms, which is only 2.5x the 201 ms an idle 500m pod needs for
# one request, and cgroup throttling jitter alone breached it at 0.5 req/s. Expressing
# the SLO as a multiple of the unloaded p50 makes it a statement about QUEUEING - "a
# request may take up to 5x longer than it would on an idle pod" - which is what an
# autoscaler is actually responsible for controlling.
SLO_MULTIPLE = 5.0
BASELINE_RATE = 0.5
BASELINE_S = 20.0
WARMUP_S = 5.0
# 40s per step: p95 needs samples. At 1 req/s a 25s hold yields ~25 requests, so the
# 95th percentile IS the second-slowest observation and swings wildly run to run.
HOLD_S = 40.0
RATES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]


def poisson_arrivals(rate: float, seconds: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    out, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= seconds:
            return out
        out.append(t)


def spawn_app(port: int, iterations: int) -> subprocess.Popen:
    """Run the app under a CPU quota equal to the pod's limit."""
    if not shutil.which("systemd-run"):
        raise SystemExit("systemd-run not found; use --url against a real pod instead")
    cmd = [
        "systemd-run", "--user", "--scope", "--quiet",
        "-p", f"CPUQuota={CPU_QUOTA_PCT}%",
        str(REPO / ".venv" / "bin" / "python"), str(REPO / "app" / "main.py"),
    ]
    env = {"PORT": str(port), "WORK_ITERATIONS": str(iterations),
           "POD_NAME": "calibration", "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home()),
           "XDG_RUNTIME_DIR": f"/run/user/{__import__('os').getuid()}"}
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(2.5)
    return p


def step(url: str, rate: float, seed: int, timeout: float,
         hold: float = HOLD_S, warmup: float = WARMUP_S) -> dict:
    arrivals = poisson_arrivals(rate, warmup + hold, seed)
    rows: list = []
    stats = asyncio.run(openloop_run(url, arrivals, rows, timeout))

    # Score only the post-warm-up portion, by SEND time. Scoring by completion time
    # would push slow requests into the next bucket and flatter the measurement exactly
    # where it matters - at the knee.
    steady = [r for r in rows if r[2] >= warmup and r[4] == 200]
    errs = [r for r in rows if r[4] != 200]
    lat = np.array(sorted(r[3] for r in steady)) if steady else np.array([0.0])
    return dict(
        rate=rate, n=len(steady), errors=len(errs), shed=stats["shed"],
        p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
        p99=float(np.percentile(lat, 99)), mean=float(lat.mean()),
        achieved_rps=round(len(steady) / hold, 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=None)
    ap.add_argument("--spawn", action="store_true",
                    help="start a CPU-limited local instance instead of using a cluster")
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--iterations", type=int, default=250_000)
    ap.add_argument("--slo-p95-ms", type=float, default=None,
                    help="override the derived SLO with an absolute value")
    ap.add_argument("--slo-multiple", type=float, default=SLO_MULTIPLE,
                    help="SLO = this multiple of the measured unloaded p50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    proc = None
    if args.spawn:
        url = f"http://127.0.0.1:{args.port}/"
        proc = spawn_app(args.port, args.iterations)
        where = f"local process, CPUQuota={CPU_QUOTA_PCT}% (== 500m)"
    else:
        if not args.url:
            raise SystemExit("give --url or --spawn")
        url = args.url
        where = url

    print(f"calibrating PER_POD_CAPACITY against {where}")
    print(f"  work per request : {args.iterations:,} SHA-256 rounds")

    # Baseline: what does one request cost with the pod essentially idle? Everything
    # above this is queueing, which is the thing the SLO is really about.
    base = step(url, BASELINE_RATE, args.seed - 1, args.timeout,
                hold=BASELINE_S, warmup=2.0)
    slo = args.slo_p95_ms or round(base["p50"] * args.slo_multiple, 1)
    print(f"  unloaded p50     : {base['p50']:.1f} ms  "
          f"(n={base['n']}, p95 {base['p95']:.1f} ms)")
    print(f"  SLO              : p95 < {slo:g} ms"
          + ("" if args.slo_p95_ms else
             f"  = {args.slo_multiple:g}x the unloaded p50"))
    print(f"  each step        : {WARMUP_S:g}s warm-up + {HOLD_S:g}s measured\n")
    args.slo_p95_ms = slo
    print(f"  {'offered':>8} {'achieved':>9} {'n':>6} {'p50':>8} {'p95':>8} "
          f"{'p99':>8} {'err':>5} {'verdict':>8}")
    print("  " + "-" * 70)

    steps = []
    try:
        for i, rate in enumerate(RATES):
            s = step(url, rate, args.seed + i, args.timeout)
            s["pass"] = s["p95"] < args.slo_p95_ms and s["errors"] == 0
            steps.append(s)
            print(f"  {rate:>8.1f} {s['achieved_rps']:>9.2f} {s['n']:>6} "
                  f"{s['p50']:>8.1f} {s['p95']:>8.1f} {s['p99']:>8.1f} "
                  f"{s['errors']:>5} {'ok' if s['pass'] else 'BREACH':>8}",
                  flush=True)
            if not s["pass"]:
                break
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    passing = [s for s in steps if s["pass"]]
    failing = [s for s in steps if not s["pass"]]
    if not passing:
        print("\nEvery step breached the SLO - lower --iterations or raise the SLO.")
        return 1

    last_ok, first_bad = passing[-1], (failing[0] if failing else None)
    if first_bad:
        # Linear interpolation of the p95 curve between the bracketing steps.
        f = ((args.slo_p95_ms - last_ok["p95"])
             / max(first_bad["p95"] - last_ok["p95"], 1e-9))
        capacity = last_ok["rate"] + f * (first_bad["rate"] - last_ok["rate"])
        bracketed = True
    else:
        capacity = last_ok["rate"]
        bracketed = False

    out = dict(
        per_pod_capacity_rps=round(capacity, 3),
        per_pod_capacity_rpm=round(capacity * 60, 1),
        slo_p95_ms=args.slo_p95_ms, slo_multiple=args.slo_multiple,
        unloaded_p50_ms=round(base["p50"], 1), work_iterations=args.iterations,
        cpu_limit="500m" if args.spawn else "cluster pod limit",
        measured_against=where, bracketed=bracketed, steps=steps,
        note=("measured against a CPU-quota-limited local process, NOT a pod; "
              "re-run with --url once the cluster exists"
              if args.spawn else "measured against a real single-replica pod"),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / ("phase4_capacity_local.json" if args.spawn
                      else "phase4_capacity.json")
    dest.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print(f"  PER_POD_CAPACITY = {capacity:.2f} req/s  "
          f"({capacity * 60:.0f} req/min) at p95 < {args.slo_p95_ms:g} ms")
    if not bracketed:
        print("  WARNING: the ramp never breached the SLO - this is a LOWER BOUND. "
              "Extend RATES.")
    print("=" * 72)

    frozen = json.loads((REPO / "load" / "segments.json").read_text())["segments"]
    print("\n  Replicas the frozen segments would need at this capacity")
    print(f"  (compression 10x => offered req/s = trace req/min / 6)")
    for name, seg in frozen.items():
        peak_rps = seg["max_rpm"] / 6.0
        mean_rps = seg["mean_rpm"] / 6.0
        print(f"    {name:<8} peak {peak_rps:5.1f} req/s -> "
              f"{int(np.ceil(peak_rps / capacity)):>2} pods   "
              f"mean {mean_rps:5.1f} req/s -> "
              f"{int(np.ceil(mean_rps / capacity)):>2} pods")
    print(f"\nwrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
