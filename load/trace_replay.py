"""Phase 4 - replay a frozen NASA segment against the cluster.

    ./.venv/bin/python load/trace_replay.py --segment burst
    ./.venv/bin/python load/trace_replay.py --segment quiet --url http://127.0.0.1:18080/
    ./.venv/bin/python load/trace_replay.py --segment diurnal --dry-run   # schedule only

Writes experiments/<run-id>/raw/requests.csv (one row per request) and summary.json.

--------------------------------------------------------------------------------
Why this is OPEN-LOOP, and why that matters more than the tool choice
--------------------------------------------------------------------------------
The phase notes recommend Locust. We do not use it, and the reason is methodological
rather than stylistic (DEVIATIONS.md D-011).

Locust is closed-loop: a fixed population of simulated users each send a request, WAIT
for the response, then send the next. When the server slows down, a closed-loop
generator therefore sends fewer requests per second - the offered load silently drops in
response to the server's own distress.

That is fatal for this specific experiment. Phases 6-8 compare a predictive autoscaler
against reactive HPA. HPA's characteristic failure is being late: load arrives, latency
climbs, and only then does it add pods. Under a closed-loop generator that latency spike
would throttle the offered load, hiding the very overload we are trying to measure, and
flattering the reactive arm exactly when it is performing worst.

This generator instead sends requests on a schedule fixed in advance from the trace. If
the server cannot keep up, requests queue and latency rises - which is what a real
population of users hitting a slow service actually produces.

--------------------------------------------------------------------------------
Time compression, and the one thing it does NOT compress
--------------------------------------------------------------------------------
One trace-minute is replayed in 60/C wall-seconds. Everything downstream - forecast
horizon, cooldowns, evaluation windows - must use the same compressed clock
(DEVIATIONS.md D-010).

But pod startup does not compress. It is physical, and stays whatever it is in wall
seconds. So compression does not merely shorten the experiment; it changes which
real-world regime the experiment emulates:

    emulated real-world pod startup = C x (measured wall-clock startup)

At C=10, a pod that really takes 4 s to become Ready behaves like a 40-second startup
against a 60-second horizon - which is squarely the regime DEVIATIONS.md D-004 is about.
That is a feature, but only if it is stated: report results in compressed trace-time and
say what wall-clock startup they correspond to.

--------------------------------------------------------------------------------
Arrivals are Poisson, not evenly paced
--------------------------------------------------------------------------------
Within each trace-minute the rate is held constant and arrival times are drawn from a
Poisson process. Evenly-spaced arrivals at the same mean rate would understate queueing:
real request arrivals clump, and the clumping is what actually fills a pod's queue and
produces the tail latency that P95/P99 measure.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import aiohttp

REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "load" / "segments.json"
EXPERIMENTS = REPO / "experiments"

# One trace-minute -> 6 wall-seconds. See D-010 for how this was chosen: large enough
# that a 24-hour segment finishes in 2.4 hours, small enough that physical pod startup
# still maps to a plausible real-world horizon fraction.
DEFAULT_COMPRESSION = 10.0

# Magnitude multiplier on the request COUNT per trace-minute. Frozen at 1.0, and that
# is a deliberate decision rather than an unset default - see DEVIATIONS.md D-013.
#
# The phase notes ask us to multiply the trace up until a single pod saturates, because
# NASA's ~400 req/min peak would not trouble a cluster. That goal is met here, but by a
# different knob: compression already multiplies the OFFERED RATE by C. At C=10 the
# segments peak at 14.2-22.8 req/s against a measured per-pod capacity of 3.25 req/s, so
# one pod is already 4.4x-7.0x oversubscribed at peak. The remaining gap was closed by
# making each request cost more CPU (WORK_ITERATIONS, D-012) rather than by sending more
# requests, because the generator and the cluster share one laptop: raising the count
# spends the load generator's CPU competing with the pods it is supposed to be measuring,
# whereas raising per-request work spends the CLUSTER's CPU, which is what we want loaded.
#
# Kept as a live parameter (`--magnitude`) because Phase 5's replica formula must stay
# correct if it is ever raised - the forecaster predicts trace req/min, and the offered
# rate is `y * magnitude * compression / 60` req/s.
DEFAULT_MAGNITUDE = 1.0

# Requests may pile up if the cluster is overloaded. This cap exists only to stop the
# generator exhausting file descriptors and dying mid-experiment. Hitting it is recorded
# as `shed`, never silently absorbed - shedding IS the overload signal, and hiding it
# would reintroduce exactly the closed-loop bias this generator avoids.
MAX_INFLIGHT = 4000


def build_schedule(y: list[int], compression: float, magnitude: float,
                   seed: int) -> list[float]:
    """Arrival offsets in wall-seconds for the whole segment."""
    rng = random.Random(seed)
    window = 60.0 / compression          # wall-seconds per trace-minute
    arrivals: list[float] = []
    for i, count in enumerate(y):
        expected = count * magnitude
        if expected <= 0:
            continue                      # a zero-traffic minute really sends nothing
        rate = expected / window          # requests per wall-second
        t = i * window
        end = t + window
        # Exponential gaps => Poisson process with the right mean count per window.
        while True:
            t += rng.expovariate(rate)
            if t >= end:
                break
            arrivals.append(t)
    return arrivals


async def _one(session: aiohttp.ClientSession, url: str, idx: int,
               planned: float, t0: float, rows: list) -> None:
    sent = time.perf_counter() - t0
    start = time.perf_counter()
    status, err = 0, ""
    try:
        async with session.get(url) as resp:
            await resp.read()             # drain, or latency excludes the body
            status = resp.status
    except Exception as e:                # noqa: BLE001
        err = f"{type(e).__name__}"
    latency = time.perf_counter() - start
    rows.append((idx, round(planned, 4), round(sent, 4),
                 round(latency * 1000, 3), status, err))


async def run(url: str, arrivals: list[float], rows: list, timeout_s: float) -> dict:
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    shed = 0
    inflight: set[asyncio.Task] = set()

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t0 = time.perf_counter()
        for idx, planned in enumerate(arrivals):
            # Absolute deadline, not cumulative sleeps: sleeping by deltas accumulates
            # drift, and over a 24-hour segment the schedule would slide by minutes.
            delay = planned - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)

            inflight = {t for t in inflight if not t.done()}
            if len(inflight) >= MAX_INFLIGHT:
                shed += 1
                continue
            inflight.add(asyncio.create_task(
                _one(session, url, idx, planned, t0, rows)))

        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        wall = time.perf_counter() - t0

    return dict(wall_seconds=round(wall, 2), shed=shed)


def summarise(rows: list, y: list[int], compression: float,
              magnitude: float) -> dict:
    lat = sorted(r[3] for r in rows if r[4] == 200)
    errs = [r for r in rows if r[4] != 200]

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return round(lat[min(int(len(lat) * p / 100), len(lat) - 1)], 2)

    # Achieved requests per trace-minute, for the shape check.
    window = 60.0 / compression
    achieved = [0] * len(y)
    for r in rows:
        i = int(r[2] // window)
        if 0 <= i < len(achieved):
            achieved[i] += 1

    target = [c * magnitude for c in y]
    both = [(a, t) for a, t in zip(achieved, target)]
    n = len(both)
    ma = sum(a for a, _ in both) / n
    mt = sum(t for _, t in both) / n
    cov = sum((a - ma) * (t - mt) for a, t in both)
    va = sum((a - ma) ** 2 for a, _ in both) ** 0.5
    vt = sum((t - mt) ** 2 for _, t in both) ** 0.5
    corr = cov / (va * vt) if va and vt else 0.0

    return dict(
        requests_sent=len(rows),
        requests_ok=len(lat),
        requests_error=len(errs),
        error_kinds=dict(sorted(
            {(r[5] or f"HTTP{r[4]}"): sum(1 for q in errs
                                          if (q[5] or f"HTTP{q[4]}") == (r[5] or f"HTTP{r[4]}"))
             for r in errs}.items())),
        latency_ms=dict(
            mean=round(statistics.mean(lat), 2) if lat else 0.0,
            p50=pct(50), p90=pct(90), p95=pct(95), p99=pct(99),
            max=round(lat[-1], 2) if lat else 0.0,
        ),
        shape_correlation=round(corr, 4),
        achieved_per_trace_minute=achieved,
        target_per_trace_minute=[round(t, 2) for t in target],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segment", required=True, choices=["diurnal", "burst", "quiet"])
    ap.add_argument("--url", default="http://localhost:30080/")
    ap.add_argument("--compression", type=float, default=DEFAULT_COMPRESSION)
    ap.add_argument("--magnitude", type=float, default=DEFAULT_MAGNITUDE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--minutes", type=int, default=None,
                    help="replay only the first N trace-minutes (smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the schedule without sending anything")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args()

    frozen = json.loads(SEGMENTS.read_text())
    seg = frozen["segments"][args.segment]
    y = seg["y"][: args.minutes] if args.minutes else seg["y"]

    arrivals = build_schedule(y, args.compression, args.magnitude, args.seed)
    window = 60.0 / args.compression
    wall_planned = len(y) * window
    peak_rps = max(y) * args.magnitude / window

    print(f"segment            : {args.segment}  ({seg['sha256_16']})")
    print(f"trace window       : {seg['first_minute']} -> {seg['last_minute']}")
    print(f"trace minutes      : {len(y):,}")
    print(f"compression        : {args.compression:g}x  "
          f"(1 trace-minute = {window:g} wall-seconds)")
    print(f"magnitude          : {args.magnitude:g}x")
    print(f"planned requests   : {len(arrivals):,}")
    print(f"planned wall time  : {wall_planned:,.0f} s ({wall_planned / 60:.1f} min)")
    print(f"mean / peak rate   : {len(arrivals) / wall_planned:.1f} / {peak_rps:.1f} req/s")
    print(f"target             : {args.url}")

    # Saturation check. The phase brief requires the replayed load to be heavy enough
    # that a single pod genuinely saturates; asserting that is not the same as showing
    # it, so print the ratio against the MEASURED capacity every run. If capacity has
    # not been measured yet there is nothing honest to print, so print nothing.
    cap_file = next((p for p in (REPO / "results" / "phase4_capacity.json",
                                 REPO / "results" / "phase4_capacity_local.json")
                     if p.exists()), None)
    if cap_file:
        cap = json.loads(cap_file.read_text())
        c = cap["per_pod_capacity_rps"]
        mean_rps = len(arrivals) / wall_planned
        print(f"per-pod capacity   : {c:.2f} req/s  ({cap_file.name}"
              + (", NOT pod-measured" if "local" in cap_file.name else "") + ")")
        print(f"saturation         : peak {peak_rps / c:.1f}x one pod "
              f"(needs {-(-peak_rps // c):.0f} pods)   "
              f"mean {mean_rps / c:.1f}x (needs {-(-mean_rps // c):.0f})")

    if args.dry_run:
        print("\ndry run - nothing sent.")
        return 0

    run_id = (f"{datetime.now():%Y%m%d-%H%M%S}-{args.segment}"
              + (f"-{args.tag}" if args.tag else ""))
    outdir = EXPERIMENTS / run_id
    (outdir / "raw").mkdir(parents=True, exist_ok=True)

    rows: list = []
    print(f"\nreplaying -> experiments/{run_id}/ ...", flush=True)
    stats = asyncio.run(run(args.url, arrivals, rows, args.timeout))
    rows.sort(key=lambda r: r[0])

    with open(outdir / "raw" / "requests.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "planned_s", "sent_s", "latency_ms", "status", "error"])
        w.writerows(rows)

    summary = dict(
        run_id=run_id, segment=args.segment, segment_sha=seg["sha256_16"],
        url=args.url, compression=args.compression, magnitude=args.magnitude,
        seed=args.seed, trace_minutes=len(y),
        planned_requests=len(arrivals), **stats,
        **summarise(rows, y, args.compression, args.magnitude),
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    lat = summary["latency_ms"]
    print(f"\n  wall time        : {stats['wall_seconds']:,.1f} s "
          f"(planned {wall_planned:,.0f} s)")
    print(f"  sent / ok / err  : {summary['requests_sent']:,} / "
          f"{summary['requests_ok']:,} / {summary['requests_error']:,}"
          + (f"   shed {stats['shed']:,}" if stats["shed"] else ""))
    if summary["error_kinds"]:
        print(f"  errors           : {summary['error_kinds']}")
    print(f"  latency ms       : mean {lat['mean']}  p50 {lat['p50']}  "
          f"p90 {lat['p90']}  p95 {lat['p95']}  p99 {lat['p99']}  max {lat['max']}")
    print(f"  shape corr       : {summary['shape_correlation']:.4f}  "
          f"(achieved vs trace, per trace-minute)")
    print(f"\nwrote experiments/{run_id}/summary.json and raw/requests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
