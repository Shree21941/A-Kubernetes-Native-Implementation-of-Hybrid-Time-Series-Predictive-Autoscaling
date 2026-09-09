"""Phase 5 - does the compression factor destroy the predictive advantage?

    ./.venv/bin/python src/controller/sweep_compression.py

The Phase 5 simulation found the controller FOLLOWING the load rather than leading it,
which for a forecaster with real skill (R^2 = 0.71) is surprising until the two clocks
are written side by side (D-010):

    forecast horizon   = 1 trace-minute = 60/C wall-seconds   <- compresses
    pod cold start     = whatever it physically is            <- does NOT compress

    usable head start  = 60/C - cold_start   wall-seconds

At C=10 the horizon is 6 wall-seconds and a pod needs ~5 of them, so the entire
advantage of knowing the future is about one second. At C=1 the same pod against a
60-second horizon leaves 55 seconds of genuine head start. The forecaster is identical in
both cases; only the clock differs.

This script measures that curve instead of asserting it, because the answer decides
whether C=10 is a legitimate setting for Phases 6-8 or one that guarantees a null result.

Note what is NOT being asked. This does not ask whether our controller beats HPA - that
is Phase 6, in-cluster, against real baselines. It asks whether the experimental setup
leaves room for ANY t+1 predictive method to demonstrate an advantage. A setup in which
the answer is no would produce a clean, publishable-looking, meaningless comparison.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "controller")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import simulate as S  # noqa: E402
from config import Config  # noqa: E402

RESULTS = REPO / "results"
FACTORS = [1.0, 2.0, 5.0, 10.0, 20.0, 60.0]
SEGMENT = "diurnal"


def main() -> int:
    seg = json.loads((REPO / "load" / "segments.json").read_text())["segments"][SEGMENT]
    y = seg["y"]
    S.TRACE0 = np.datetime64(seg["first_minute"].replace(" ", "T")[:16], "m")

    from forecaster import Forecaster
    base = Config()
    fc = Forecaster(base.bundle, prophet_mode="table")

    print(f"\nCompression sweep - segment '{SEGMENT}', "
          f"cold start {base.pod_cold_start_wall_s:g}s wall "
          f"({'measured' if base.pod_cold_start_measured else 'PLACEHOLDER'})")
    print("=" * 88)
    print("  per-pod capacity is scaled with C so that pods-needed stays constant;")
    print("  the only variable is how much of the horizon survives pod startup.\n")
    print(f"  {'C':>5} {'horizon(s)':>11} {'headstart(s)':>13} {'pods@peak':>10} "
          f"{'pred deficit':>13} {'reac deficit':>13} {'advantage':>11}")
    print("  " + "-" * 84)

    rows = []
    for c in FACTORS:
        # Hold the SCALING PROBLEM constant while varying the clock.
        #
        # Compression changes two things at once: the head start (60/C wall-seconds) and
        # the offered rate (y * C / 60 req/s), and therefore how many pods the segment
        # needs. A naive sweep measures both and can conclude nothing about either - the
        # first version of this script showed zero deficit for both arms at C=1, not
        # because prediction was perfect but because at C=1 the peak fits in a single
        # pod and there is no scaling problem to get right.
        #
        # Scaling per-pod capacity with C keeps pods-needed invariant, so the only
        # variable left is how much of the horizon survives pod startup.
        cap = base.per_pod_capacity_rps * (c / base.compression)
        cfg = replace(base, compression=c, per_pod_capacity_rps=cap)
        horizon_wall = cfg.horizon_trace_min * cfg.trace_minute_wall_s
        head = horizon_wall - cfg.pod_cold_start_wall_s

        p = S.simulate(cfg, y, fc, reactive=False)
        r = S.simulate(cfg, y, fc, reactive=True)
        lb = cfg.look_back_trace_min
        dp = sum(x["deficit"] for x in p[lb:])
        dr = sum(x["deficit"] for x in r[lb:])
        # Positive = the predictive arm left fewer requests unserved.
        adv = (dr - dp) / dr * 100 if dr else 0.0
        acts = sum(1 for x in p if x["action"].startswith("scale"))
        rows.append(dict(compression=c, per_pod_capacity_rps=round(cap, 3),
                         peak_pods_needed=int(np.ceil(max(y) / cfg.per_pod_capacity_rpm_trace)),
                         horizon_wall_s=horizon_wall,
                         head_start_wall_s=head, pred_deficit=round(dp, 1),
                         reac_deficit=round(dr, 1), advantage_pct=round(adv, 2),
                         pred_actions=acts))
        print(f"  {c:>5.0f} {horizon_wall:>11.1f} {head:>13.1f} "
              f"{rows[-1]['peak_pods_needed']:>10} "
              f"{dp:>13.1f} {dr:>13.1f} {adv:>10.1f}%")

    print("=" * 88)

    # The flat region is a limitation of this model, not a result about the world.
    flat = {r["advantage_pct"] for r in rows if r["head_start_wall_s"] > 0}
    if len(flat) == 1:
        print("  READ THE FLAT REGION AS 'NO ANSWER', NOT AS 'NO EFFECT'.")
        print("  Every C with a positive head start scores identically because this")
        print("  simulation evaluates capacity ONCE PER TRACE-MINUTE: a pod that becomes")
        print("  Ready 55 wall-seconds early and one that makes it by 1 wall-second are")
        print("  both simply 'Ready' for that minute. Sub-trace-minute timing - which is")
        print("  exactly what the head start consists of - is below this model's")
        print("  resolution. Answering the head-start question needs the real cluster,")
        print("  where requests arrive continuously and a pod that is late by 5 seconds")
        print("  drops 5 seconds of requests. That is Phase 6/7, not this script.\n")

    neg = [r for r in rows if r["head_start_wall_s"] <= 0]
    if neg:
        print(f"  At C >= {min(r['compression'] for r in neg):g} the head start is "
              f"NEGATIVE: a pod ordered on the forecast arrives after the minute it was\n"
              f"  forecast for. No t+1 predictive method can help there, however good "
              f"the forecaster.")
    print("  SIMULATED - see simulate.py for what is and is not modelled.")

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    cs = [r["compression"] for r in rows]
    ax1.plot(cs, [r["advantage_pct"] for r in rows], "o-", color="#d62728", lw=1.8,
             label="predictive advantage (% fewer unserved requests)")
    ax1.axhline(0, color="#666666", lw=1, ls="--")
    ax1.set_xscale("log")
    ax1.set_xticks(cs)
    ax1.set_xticklabels([f"{c:g}x" for c in cs])
    ax1.set_xlabel("compression factor C")
    ax1.set_ylabel("predictive advantage (%)", color="#d62728")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(cs, [r["head_start_wall_s"] for r in rows], "s--", color="#1f77b4",
             lw=1.5, label="usable head start (wall s)")
    ax2.axhline(0, color="#1f77b4", lw=0.8, ls=":")
    ax2.set_ylabel("horizon minus pod cold start (wall seconds)", color="#1f77b4")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax1.set_title(f"SIMULATED - the predictive advantage is a property of the CLOCK\n"
                  f"segment '{SEGMENT}', pod cold start "
                  f"{base.pod_cold_start_wall_s:g}s wall (placeholder)")
    fig.tight_layout()
    out = RESULTS / "fig_phase5_compression_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    (RESULTS / "phase5_compression_sweep.json").write_text(json.dumps(
        dict(segment=SEGMENT, simulated=True,
             pod_cold_start_wall_s=base.pod_cold_start_wall_s,
             pod_cold_start_measured=base.pod_cold_start_measured,
             rows=rows), indent=2))
    print(f"\nwrote {out.relative_to(REPO)} and results/phase5_compression_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
