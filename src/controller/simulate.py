"""Phase 5 - replay a frozen segment through the real forecaster and planner.

    ./.venv/bin/python src/controller/simulate.py --segment burst
    ./.venv/bin/python src/controller/simulate.py --segment quiet

WHAT IS REAL HERE, AND WHAT IS NOT. This matters more than the plot.

  REAL : the Phase 2 hybrid, loaded from the bundle, forecasting one trace-minute ahead
         from a 60-minute window - the same code path main.py runs in-cluster.
  REAL : the planner, with the shipping safety layer.
  REAL : the segment, byte-identical to what the load generator replays.
  SIMULATED : the cluster. Pods become Ready after `pod_cold_start_wall_s`, and serve
         `per_pod_capacity` each. There is no scheduler, no kubelet, no queueing model,
         no CPU contention.
  SIMULATED : the Monitor stage. The controller is fed the trace's own counts instead of
         Prometheus samples of real traffic. Phase 4 measured the generator reproducing
         the trace shape at r = 0.9496 against a Poisson ceiling of 0.9476, so this is a
         good approximation of what Prometheus would see - but it is an approximation,
         and it removes scrape noise the real loop will face.

So this establishes that THE CONTROL LOGIC is proactive. It does not establish that the
system is, and it is not a substitute for the Phase 5 Definition-of-Done item, which
requires the controller running in-cluster against a real replay. Any figure produced
here must carry the word SIMULATED.

Why build it anyway: it makes the forecaster/planner interaction debuggable in seconds
rather than in 18-minute cluster runs, and it produces the lead-time number that tells
us whether the cluster experiment is even worth running at this compression factor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "controller")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import Config  # noqa: E402
from planner import Action, Planner  # noqa: E402

RESULTS = REPO / "results"


class FakeCluster:
    """Pods that take real wall-time to become Ready.

    This is the only part of the simulation that matters, because it is the part the
    compression factor interacts with (D-010): the scale-up decision is made on the
    compressed clock, and the pod arrives on the wall clock.
    """

    def __init__(self, cfg, start_replicas: int) -> None:
        self.cfg = cfg
        self.spec = start_replicas
        self._ready_at: list[float] = [0.0] * start_replicas   # wall time each pod ready

    def scale_to(self, n: int, now_wall: float) -> None:
        if n > self.spec:
            for _ in range(n - self.spec):
                self._ready_at.append(now_wall + self.cfg.pod_cold_start_wall_s)
        elif n < self.spec:
            # Remove the newest pods first - the ones most likely still starting.
            self._ready_at.sort()
            del self._ready_at[n:]
        self.spec = n

    def ready(self, now_wall: float) -> int:
        return sum(1 for t in self._ready_at if t <= now_wall)


def simulate(cfg: Config, y: list[int], fc, reactive: bool = False) -> dict:
    planner = Planner(cfg)
    cluster = FakeCluster(cfg, cfg.min_replicas)
    lb = cfg.look_back_trace_min

    rows = []
    for i in range(len(y)):
        now_wall = i * cfg.control_period_wall_s
        warming = i < lb

        if warming or reactive:
            # Reactive reference: act on the PREVIOUS trace-minute, not the current one.
            #
            # An earlier version used y[i] here, which handed the reactive arm the true
            # rate for the very minute it was being scored on - perfect knowledge of the
            # present, which no reactive autoscaler has. Real HPA reads a metric window
            # that has already closed. Scoring against an oracle-reactive strawman would
            # have made the predictive arm look worse than it is, in a way that happens
            # to flatter nobody but still invalidates the comparison.
            #
            # This is still only a REFERENCE LINE, not the Phase 6 baseline. The real
            # comparison uses actual HPA and KEDA in-cluster, with their own metric
            # windows, sync periods and stabilisation behaviour - none of which are
            # modelled here.
            predicted = float(y[i - 1]) if i > 0 else float(y[0])
            mode = "reactive" if reactive else "warmup"
            inf_ms = 0.0
        else:
            window = np.array(y[i - lb + 1:i + 1], dtype="float64")
            ts = np.array([TRACE0 + np.timedelta64(k, "m")
                           for k in range(i - lb + 1, i + 1)], dtype="datetime64[m]")
            f = fc.predict(window, ts, TRACE0 + np.timedelta64(i + cfg.horizon_trace_min, "m"))
            predicted, mode, inf_ms = f.value, "forecast", f.total_ms

        d = planner.plan(predicted, current=cluster.spec, now=now_wall,
                         warming_up=False)
        if d.action in (Action.SCALE_UP, Action.SCALE_DOWN):
            cluster.scale_to(d.target, now_wall)

        ready = cluster.ready(now_wall)
        # Capacity actually available during this trace-minute, in trace units.
        served = ready * cfg.per_pod_capacity_rpm_trace
        rows.append(dict(i=i, actual=y[i], predicted=predicted, mode=mode,
                         spec=cluster.spec, ready=ready, desired=d.desired_clamped,
                         action=d.action.value, served_capacity=served,
                         deficit=max(0.0, y[i] - served), inference_ms=inf_ms))
    return rows


TRACE0 = np.datetime64("1995-08-24T06:52", "m")   # overwritten per segment in main()


def lead_time(rows: list[dict], cfg: Config, max_lag: int = 15) -> dict:
    """Cross-correlation lag between the replica series and the load series.

    An earlier version of this function compared the argmax of each series - "when did
    replicas peak" vs "when did load peak" - and reported a lead of +26 trace-minutes on
    the burst segment. That number was an artifact, and an impossible one: the forecaster
    sees exactly ONE minute ahead, so it cannot anticipate anything by 26. What the
    argmax actually found was the start of a plateau, because the replica count reached 4
    early and stayed there while load kept climbing.

    The defensible question is which lag best ALIGNS the two series:

        lag k maximises corr(replicas[t], load[t + k])

    k > 0 means the replica curve resembles the load curve k minutes in the FUTURE - the
    pods lead. k < 0 means it resembles the past - the pods follow. A purely reactive
    controller should land at k <= 0, and a t+1 forecaster's ceiling is k = +1, since
    that is all the information it has.
    """
    lb = cfg.look_back_trace_min
    actual = np.array([r["actual"] for r in rows], dtype="float64")[lb:]
    spec = np.array([r["spec"] for r in rows], dtype="float64")[lb:]
    if len(actual) <= 2 * max_lag + 2:
        return dict(error="post-warm-up region too short for a lag estimate")
    if spec.std() == 0:
        return dict(error="replica count never changed after warm-up")

    best, curve = None, {}
    for k in range(-max_lag, max_lag + 1):
        # Positive k compares replicas now against load k minutes later.
        if k >= 0:
            a, s = actual[k:], spec[:len(spec) - k]
        else:
            a, s = actual[:len(actual) + k], spec[-k:]
        if len(a) < 10 or s.std() == 0 or a.std() == 0:
            continue
        r = float(np.corrcoef(s, a)[0, 1])
        curve[k] = round(r, 4)
        if best is None or r > best[1]:
            best = (k, r)

    # Two ways this estimate can be worthless, both of which must be said out loud
    # rather than left for the reader to notice in the lag curve.
    at_boundary = abs(best[0]) >= max_lag
    weak = best[1] < 0.5
    return dict(
        lead_trace_min=best[0], correlation_at_best_lag=round(best[1], 4),
        lead_wall_s=best[0] * cfg.control_period_wall_s,
        correlation_at_zero_lag=curve.get(0),
        correlation_at_horizon_lag=curve.get(cfg.horizon_trace_min),
        horizon_trace_min=cfg.horizon_trace_min,
        # The argmax ran into the edge of the search, so the true optimum is outside it
        # and this number is a floor, not an estimate.
        argmax_at_search_boundary=at_boundary,
        # A flat, low correlation curve means the replica series does not track load at
        # ANY lag, so asking which lag is best is asking the wrong question.
        correlation_too_weak_to_interpret=weak,
        lag_curve=curve,
    )


def main() -> int:
    global TRACE0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segment", default="burst", choices=["diurnal", "burst", "quiet"])
    ap.add_argument("--prophet-mode", default="table")
    args = ap.parse_args()

    cfg = Config()
    seg = json.loads((REPO / "load" / "segments.json").read_text())["segments"][args.segment]
    y = seg["y"]
    TRACE0 = np.datetime64(seg["first_minute"].replace(" ", "T")[:16], "m")

    from forecaster import Forecaster
    fc = Forecaster(cfg.bundle, prophet_mode=args.prophet_mode)

    pred_rows = simulate(cfg, y, fc, reactive=False)
    reac_rows = simulate(cfg, y, fc, reactive=True)

    lead = lead_time(pred_rows, cfg)
    lb = cfg.look_back_trace_min

    def deficit(rows):
        d = [r["deficit"] for r in rows[lb:]]
        return dict(minutes_under_capacity=sum(1 for x in d if x > 0),
                    total_deficit=round(sum(d), 1))

    dp, dr = deficit(pred_rows), deficit(reac_rows)
    inf = [r["inference_ms"] for r in pred_rows if r["mode"] == "forecast"]

    print(f"\nSIMULATED controller - segment '{args.segment}' "
          f"({len(y)} trace-minutes, {lb} of them warm-up)")
    print("=" * 78)
    print(f"  {'':<28} {'predictive':>14} {'reactive':>14}")
    print(f"  {'minutes under capacity':<28} "
          f"{dp['minutes_under_capacity']:>14} {dr['minutes_under_capacity']:>14}")
    print(f"  {'total unserved (req)':<28} "
          f"{dp['total_deficit']:>14.1f} {dr['total_deficit']:>14.1f}")
    print(f"  {'peak replicas':<28} "
          f"{max(r['spec'] for r in pred_rows):>14} "
          f"{max(r['spec'] for r in reac_rows):>14}")
    print(f"  {'scaling actions':<28} "
          f"{sum(1 for r in pred_rows if r['action'].startswith('scale')):>14} "
          f"{sum(1 for r in reac_rows if r['action'].startswith('scale')):>14}")
    print("-" * 78)
    lead_r = lead_time(reac_rows, cfg)
    if "error" not in lead:
        v = lead["lead_trace_min"]
        print(f"  best-alignment lag, predictive : {v:+d} trace-min "
              f"({lead['lead_wall_s']:+.1f}s wall, r={lead['correlation_at_best_lag']})")
        if "error" not in lead_r:
            print(f"  best-alignment lag, reactive   : "
                  f"{lead_r['lead_trace_min']:+d} trace-min "
                  f"(r={lead_r['correlation_at_best_lag']})")
        print(f"  forecaster's ceiling is +{cfg.horizon_trace_min} "
              f"(it sees one trace-minute ahead)")
        print(f"  r at lag 0 / lag +1            : "
              f"{lead['correlation_at_zero_lag']} / "
              f"{lead['correlation_at_horizon_lag']}")
        if lead["correlation_too_weak_to_interpret"]:
            print("  => LAG IS NOT INTERPRETABLE: peak r < 0.5, so the replica curve "
                  "does not\n     track load at any lag. The lead question does not "
                  "apply to this run.")
        elif lead["argmax_at_search_boundary"]:
            print(f"  => argmax hit the +/-{15} search boundary; treat as a bound, "
                  "not an estimate.")
        else:
            print("  " + ("=> replicas LEAD the load" if v > 0 else
                          "=> replicas FOLLOW the load" if v < 0 else
                          "=> replicas track the load with no lead"))
    else:
        print(f"  lead: {lead['error']}")
    if inf:
        print(f"  live inference: median {sorted(inf)[len(inf) // 2]:.2f} ms "
              f"({sorted(inf)[len(inf) // 2] / (cfg.control_period_wall_s * 1000):.2%} "
              f"of the {cfg.control_period_wall_s:g}s cycle)")
    print("=" * 78)
    print("  SIMULATED: pods are modelled, not scheduled; Monitor is the trace itself,")
    print(f"  not Prometheus; cold start is a {cfg.pod_cold_start_wall_s:g}s placeholder"
          + ("" if cfg.pod_cold_start_measured else ", NOT a measurement") + ".")

    # -- figure ---------------------------------------------------------------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    x = np.arange(len(y))

    ax = axes[0]
    ax.plot(x, y, lw=1.2, color="#1f77b4", label="actual request rate")
    ax.plot(x, [r["predicted"] for r in pred_rows], lw=1.0, color="#ff7f0e",
            alpha=0.85, label="forecast (t+1)")
    ax.axvspan(0, lb, color="#cccccc", alpha=0.45)
    ax.text(lb / 2, ax.get_ylim()[1] * 0.92, "warm-up\n(reactive)", ha="center",
            fontsize=8, color="#555555")
    ax.set_ylabel("requests / trace-minute")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(f"SIMULATED predictive autoscaler - segment '{args.segment}'  "
                 f"(C={cfg.compression:g}x, cold start "
                 f"{cfg.pod_cold_start_wall_s:g}s placeholder)")

    ax = axes[1]
    ax.step(x, [r["spec"] for r in pred_rows], where="post", lw=1.5,
            color="#d62728", label="replicas requested (predictive)")
    ax.step(x, [r["ready"] for r in pred_rows], where="post", lw=1.0,
            color="#d62728", alpha=0.45, ls="--", label="replicas Ready")
    ax.step(x, [r["spec"] for r in reac_rows], where="post", lw=1.2,
            color="#7f7f7f", alpha=0.8, label="replicas requested (reactive ref.)")
    ax.axvspan(0, lb, color="#cccccc", alpha=0.45)
    ax.set_ylabel("replicas")
    ax.set_xlabel("trace-minute")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, len(y) - 1)

    fig.tight_layout()
    out = RESULTS / f"fig_phase5_sim_{args.segment}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    summary = dict(segment=args.segment, segment_sha=seg["sha256_16"],
                   simulated=True, config=cfg.to_dict(), lead=lead,
                   lead_reactive=lead_r,
                   predictive=dp, reactive=dr,
                   inference_ms_median=(sorted(inf)[len(inf) // 2] if inf else None),
                   rows=pred_rows)
    (RESULTS / f"phase5_sim_{args.segment}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out.relative_to(REPO)} and "
          f"results/phase5_sim_{args.segment}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
