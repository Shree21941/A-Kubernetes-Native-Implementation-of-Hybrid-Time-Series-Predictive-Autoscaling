"""Phase 8 - the five figures that carry the argument.

    ./.venv/bin/python src/eval/figures.py            # all five into results/figures/
    ./.venv/bin/python src/eval/figures.py --only 1

Figures, and what each one is FOR:

  1  request rate + replica count over time, per arm
     The central claim of the whole project is visible here or nowhere: does the
     replica line move BEFORE the demand line or after it? A table of p95 numbers can
     be argued with; a controller visibly lagging its own load cannot.

  2  latency CDF per arm
     p95 and p99 are two samples from a distribution. The CDF shows the whole shape,
     including the part where two arms have equal p95 and completely different tails.

  3  cost (CPU-seconds) vs SLO violation - the trade-off frontier
     Any arm can win on latency by running more pods. Plotting both axes at once is
     what stops that from looking like a victory.

  4  scaling actions per arm - the oscillation story
     PROJECT_PHASES.md calls stability a first-class result, not a footnote: a
     controller that reaches the right replica count by thrashing is a bad controller,
     because every action costs a cold start.

  5  forecast vs actual during a burst
     Where the model works and where it breaks. Aggregate MAPE hides whether the error
     is a small constant offset or a total miss at the one moment that mattered.

ON RUN SELECTION. Time-series figures need ONE run, not five. Picking the best-looking
one would be dishonest, so figures 1 and 5 use the run whose SLO violation rate is the
MEDIAN of its cell - representative by construction, chosen before looking at the shape
of the curve. Runs flagged by the Phase 7 integrity pass are excluded from that choice,
since a host-suspend gap would draw a flat line through the middle of the plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in this environment; write files only

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments" / "runs"
FIGS = REPO / "results" / "figures"

ARMS = ["hpa-cpu", "keda-rate", "predictive-prophet", "predictive-hybrid"]
SEGMENTS = ["diurnal", "burst", "quiet"]
CAPACITY_RPS = 3.135          # measured in Phase 4 against a real pod
SLO_MS = 593.7                # 5x the unloaded p50, measured

# Okabe-Ito: distinguishable in greyscale and for the common colour-vision deficiencies.
COLOR = {"hpa-cpu": "#D55E00", "keda-rate": "#0072B2",
         "predictive-prophet": "#E69F00", "predictive-hybrid": "#009E73"}
LABEL = {"hpa-cpu": "HPA (CPU)", "keda-rate": "KEDA (req/s)",
         "predictive-prophet": "Prophet-only", "predictive-hybrid": "Prophet+LSTM"}

FLAGGED = {("predictive-prophet", "diurnal", 3), ("hpa-cpu", "diurnal", 1),
           ("hpa-cpu", "diurnal", 5), ("predictive-prophet", "diurnal", 5)}


def load_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS.iterdir()):
        f = d / "metrics.json"
        if f.exists():
            m = json.load(f.open())
            m["_dir"] = d
            out.append(m)
    return out


def median_run(runs, arm, seg):
    """The run whose SLO rate is the median of its cell - representative, not cherry-picked."""
    sel = [m for m in runs if m["arm"] == arm and m["segment"] == seg
           and (arm, seg, m["rep"]) not in FLAGGED]
    if not sel:
        return None
    sel.sort(key=lambda m: m["latency"]["slo_violation_rate"])
    return sel[len(sel) // 2]


def read_requests(d: Path):
    sent, lat, ok = [], [], []
    with (d / "raw" / "requests.csv").open() as fh:
        for r in csv.DictReader(fh):
            try:
                s = float(r["sent_s"]); ms = float(r["latency_ms"])
            except (ValueError, KeyError, TypeError):
                continue
            sent.append(s); lat.append(ms); ok.append(r.get("status") == "200")
    return np.array(sent), np.array(lat), np.array(ok)


def read_replicas(d: Path):
    ts, spec, ready = [], [], []
    for line in (d / "raw" / "replicas.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ts.append(r["ts"]); spec.append(r["spec"]); ready.append(r["ready"])
    ts = np.array(ts)
    return ts - ts[0], np.array(spec, float), np.array(ready, float)


def rate_series(sent: np.ndarray, span: float, bin_s: float = 6.0):
    """Offered request rate in req/s, binned at one trace-minute (6 s wall at C=10)."""
    edges = np.arange(0, span + bin_s, bin_s)
    counts, _ = np.histogram(sent, bins=edges)
    return edges[:-1] + bin_s / 2, counts / bin_s


# ---------------------------------------------------------------- figure 1
def fig_rate_replicas(runs, seg="burst"):
    # sharey matters here. With independent y-axes each arm's line fills its own panel and
    # they all look equally well-matched to demand; a shared axis is what makes an arm
    # sitting two pods below the others visible as a difference rather than a redraw.
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(11, 11), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        m = median_run(runs, arm, seg)
        if m is None:
            continue
        d = m["_dir"]
        sent, _lat, _ok = read_requests(d)
        t, spec, ready = read_replicas(d)
        span = max(t[-1], sent.max())
        bt, br = rate_series(sent, span)

        # Left axis: demand, expressed in the units the controller actually decides in -
        # pods required = offered rate / measured per-pod capacity. Plotting req/s against
        # a replica count on one axis would be comparing unlike things.
        need = br / CAPACITY_RPS
        ax.fill_between(bt, 0, need, color="0.85", label="pods required by offered load")
        ax.plot(t, spec, color=COLOR[arm], lw=1.8, label="replicas requested")
        ax.plot(t, ready, color=COLOR[arm], lw=1.0, ls=":", alpha=0.8,
                label="replicas ready")
        ax.set_ylabel("pods")
        ax.set_title(f"{LABEL[arm]}   ({seg}, rep{m['rep']}, "
                     f"SLO viol {m['latency']['slo_violation_rate']*100:.2f}%)",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel(f"wall seconds  (compression 10x: 1 s = 10 trace-seconds)")
    fig.suptitle(f"Demand vs provisioned pods - {seg} segment\n"
                 f"grey = pods the offered load needs; line = pods the arm asked for",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = FIGS / f"fig1_rate_replicas_{seg}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2
def fig_latency_cdf(runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, seg in zip(axes, SEGMENTS):
        for arm in ARMS:
            # Pool every rep: the CDF of the arm's behaviour on this segment, not of one
            # lucky run. Flagged runs excluded - their latency reflects a stalled host.
            pooled = []
            for m in runs:
                if m["arm"] != arm or m["segment"] != seg:
                    continue
                if (arm, seg, m["rep"]) in FLAGGED:
                    continue
                _s, lat, _ok = read_requests(m["_dir"])
                pooled.append(lat)
            if not pooled:
                continue
            v = np.sort(np.concatenate(pooled))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v),
                    color=COLOR[arm], lw=1.6, label=LABEL[arm])
        ax.axvline(SLO_MS, color="0.3", ls="--", lw=1.2)
        ax.text(SLO_MS * 1.06, 0.06, f"SLO {SLO_MS:.0f} ms", fontsize=8, color="0.3",
                rotation=90)
        ax.set_xscale("log")
        ax.set_xlabel("request latency (ms, log scale)")
        ax.set_title(seg, fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of requests ≤ x")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Latency distribution per arm (all reps pooled, flagged runs excluded)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIGS / "fig2_latency_cdf.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 3
def fig_cost_slo(runs):
    # ONE PANEL PER SEGMENT, and the x-axes are deliberately NOT shared.
    #
    # CPU-seconds is a total, so it scales with how much work the segment contains: a
    # diurnal run is 48 wall-minutes and lands near 900, a quiet run is 9 and lands near
    # 150. Plotting all three on one axis puts the segments in three separate clusters
    # and makes the eye compare a quiet run against a diurnal one - a comparison with no
    # meaning, since they served different traffic for different durations. Cost is only
    # comparable BETWEEN ARMS WITHIN A SEGMENT, so that is the only comparison the
    # figure is allowed to show.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, seg in zip(axes, SEGMENTS):
        for m in runs:
            if m["segment"] != seg:
                continue
            cpu = m.get("cpu", {}).get("cpu_seconds_total")
            if cpu is None:
                continue
            slo = m["latency"]["slo_violation_rate"] * 100
            flag = (m["arm"], m["segment"], m["rep"]) in FLAGGED
            ax.scatter(cpu, max(slo, 1e-3), s=70 if not flag else 100,
                       c=COLOR[m["arm"]],
                       edgecolors="red" if flag else "white",
                       linewidths=1.8 if flag else 0.7, alpha=0.9, zorder=3)
        # Cell means as large crosses, so the eye reads the arm's central tendency
        # rather than being pulled to whichever single run sits furthest out.
        for arm in ARMS:
            pts = [(m["cpu"]["cpu_seconds_total"],
                    m["latency"]["slo_violation_rate"] * 100)
                   for m in runs if m["arm"] == arm and m["segment"] == seg
                   and m.get("cpu", {}).get("cpu_seconds_total") is not None]
            if pts:
                ax.scatter(st.mean([p[0] for p in pts]),
                           max(st.mean([p[1] for p in pts]), 1e-3),
                           marker="X", s=180, c=COLOR[arm],
                           edgecolors="black", linewidths=1.1, zorder=5)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("CPU-seconds consumed  (cost)")
        ax.set_title(seg, fontsize=11)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("SLO violation %  (symlog)")
    h1 = [plt.Line2D([], [], color=COLOR[a], marker="o", ls="", label=LABEL[a])
          for a in ARMS]
    h1 += [plt.Line2D([], [], color="0.4", marker="X", ls="", label="cell mean"),
           plt.Line2D([], [], color="w", marker="o", mec="red", mew=1.8, ls="",
                      label="flagged run")]
    axes[0].legend(handles=h1, fontsize=8, loc="lower left")
    fig.suptitle("Cost vs SLO within each segment - down-and-left is better.  "
                 "Cost is NOT comparable across segments, so the panels do not share an axis.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = FIGS / "fig3_cost_vs_slo.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 4
def fig_scaling_actions(runs):
    T95 = {4: 2.776, 3: 3.182, 2: 4.303}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in ((axes[0], "scaling_actions", "scaling actions"),
                           (axes[1], "direction_reversals",
                            "direction reversals (thrash)")):
        w = 0.2
        for i, arm in enumerate(ARMS):
            means, errs = [], []
            for seg in SEGMENTS:
                v = [m["cluster"][key] for m in runs
                     if m["arm"] == arm and m["segment"] == seg]
                means.append(st.mean(v) if v else 0)
                if len(v) > 1:
                    errs.append(T95.get(len(v) - 1, 2.0) * st.stdev(v) / math.sqrt(len(v)))
                else:
                    errs.append(0)
            x = np.arange(len(SEGMENTS)) + (i - 1.5) * w
            ax.bar(x, means, w, yerr=errs, capsize=3,
                   color=COLOR[arm], label=LABEL[arm], alpha=0.92)
        ax.set_xticks(np.arange(len(SEGMENTS))); ax.set_xticklabels(SEGMENTS)
        ax.set_title(title + "   (mean, 95% CI)", fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("count per run")
    axes[0].legend(fontsize=8)
    fig.suptitle("Oscillation: a controller that thrashes to the right answer "
                 "pays a cold start for every action", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = FIGS / "fig4_scaling_actions.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 5
def fig_forecast_vs_actual(runs, seg="burst"):
    preds = ["predictive-prophet", "predictive-hybrid"]
    fig, axes = plt.subplots(len(preds), 1, figsize=(11, 7.2), sharex=True)
    for ax, arm in zip(axes, preds):
        m = median_run(runs, arm, seg)
        if m is None:
            continue
        rows = [json.loads(l) for l in
                (m["_dir"] / "raw" / "decisions.jsonl").read_text().splitlines() if l.strip()]
        dec = [r for r in rows if r.get("kind") == "decision"]
        obs = np.array([r["observed_rate"] for r in dec], float)
        pred = np.array([r["predicted_rate"] for r in dec], float)
        cyc = np.arange(len(dec))

        # The forecast made at cycle t is a claim about cycle t+1, so it must be shifted
        # by one to sit above the value it was predicting. Plotting them unshifted is the
        # classic way to make a lagging forecaster look clairvoyant.
        ax.plot(cyc, obs, color="0.25", lw=1.6, label="observed rate (req/trace-min)")
        ax.plot(cyc[1:], pred[:-1], color=COLOR[arm], lw=1.5, ls="--",
                label="forecast for this minute, made one minute earlier")
        err = obs[1:] - pred[:-1]
        ax.fill_between(cyc[1:], obs[1:], pred[:-1],
                        where=err >= 0, color="#D55E00", alpha=0.18,
                        label="under-forecast (risk of SLO breach)")
        ax.fill_between(cyc[1:], obs[1:], pred[:-1],
                        where=err < 0, color="#0072B2", alpha=0.18,
                        label="over-forecast (wasted pods)")
        ax.set_ylabel("req / trace-minute")
        ax.set_title(f"{LABEL[arm]}   MAPE {m['forecast']['forecast_mape_pct']:.1f}%  "
                     f"bias {m['forecast']['forecast_bias']:+.2f}  "
                     f"(rep{m['rep']})", fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
    axes[-1].set_xlabel("control cycle (1 per trace-minute = 6 s wall)")
    fig.suptitle(f"Forecast vs actual - {seg} segment", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIGS / f"fig5_forecast_vs_actual_{seg}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=int, default=0, help="render just one figure (1-5)")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    print(f"loaded {len(runs)} runs")

    todo = {
        1: lambda: [fig_rate_replicas(runs, "burst"), fig_rate_replicas(runs, "diurnal")],
        2: lambda: [fig_latency_cdf(runs)],
        3: lambda: [fig_cost_slo(runs)],
        4: lambda: [fig_scaling_actions(runs)],
        5: lambda: [fig_forecast_vs_actual(runs, "burst"),
                    fig_forecast_vs_actual(runs, "diurnal")],
    }
    for n, fn in todo.items():
        if args.only and n != args.only:
            continue
        for p in fn():
            print(f"  wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
