"""Generate examiner figures strictly from saved measurements and experiment logs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
PAIR = Path(__file__).resolve().parent / "live_pair"
INK = "#182A3A"
TEAL = "#007378"
ORANGE = "#C27124"
BLUE = "#3878A8"
PALE = "#EEF4F4"
GREY = "#667781"
RED = "#A1372D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.edgecolor": "#C5D0D3",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_run_metrics():
    scored = json.loads((PAIR / "scored_results.json").read_text())
    return {(r["arm"], r["rep"]): r for r in scored["runs"]}


def add_values(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=INK)


def paired_bars(rows, metric, title, ylabel, path, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    x = [0, 1]
    width = 0.34
    fixed = [rows[("fixed_horizon", rep)][metric] for rep in (1, 2)]
    aware = [rows[("readiness_aware", rep)][metric] for rep in (1, 2)]
    b1 = ax.bar([i - width / 2 for i in x], fixed, width, color=BLUE,
                label="Fixed one-step", zorder=3)
    b2 = ax.bar([i + width / 2 for i in x], aware, width, color=TEAL,
                label="Readiness-aware (3-step)", zorder=3)
    add_values(ax, b1, fmt)
    add_values(ax, b2, fmt)
    ax.set_xticks(x, ["Pair 1 · seed 1001", "Pair 2 · seed 1002"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, pad=12)
    ax.grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_live_latency(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, title in zip(
        axes,
        ("latency_p95_ms", "latency_p99_ms"),
        ("P95 response latency", "P99 response latency"),
    ):
        x = [0, 1]
        width = 0.34
        fixed = [rows[("fixed_horizon", rep)][metric] for rep in (1, 2)]
        aware = [rows[("readiness_aware", rep)][metric] for rep in (1, 2)]
        bars_a = ax.bar([i - width / 2 for i in x], fixed, width, color=BLUE,
                        label="Fixed one-step", zorder=3)
        bars_b = ax.bar([i + width / 2 for i in x], aware, width, color=TEAL,
                        label="Readiness-aware (3-step)", zorder=3)
        add_values(ax, bars_a)
        add_values(ax, bars_b)
        ax.set_xticks(x, ["Seed 1001", "Seed 1002"])
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel("Latency (ms)")
        ax.grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Same burst trace and seed within each pair", x=0.04, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "live_pair_latency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_slo(rows):
    paired_bars(rows, "slo_violations_pct", "SLO violations by matched run",
                "Requests exceeding 593.7 ms (%)", OUT / "live_pair_slo.png", "{:.2f}")


def plot_resources(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    metrics = [("replica_seconds_est", "Estimated replica-seconds"),
               ("pending_seconds_est", "Estimated pending-pod seconds")]
    for ax, (metric, title) in zip(axes, metrics):
        x = [0, 1]
        width = 0.34
        fixed = [rows[("fixed_horizon", rep)][metric] for rep in (1, 2)]
        aware = [rows[("readiness_aware", rep)][metric] for rep in (1, 2)]
        b1 = ax.bar([i - width / 2 for i in x], fixed, width, color=BLUE,
                    label="Fixed one-step", zorder=3)
        b2 = ax.bar([i + width / 2 for i in x], aware, width, color=TEAL,
                    label="Readiness-aware", zorder=3)
        add_values(ax, b1, "{:.0f}")
        add_values(ax, b2, "{:.0f}")
        ax.set_xticks(x, ["Seed 1001", "Seed 1002"])
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel("Seconds (estimated from 6 s sampling)")
        ax.grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "live_pair_resource_use.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_and_inference(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    groups = ["fixed_horizon", "readiness_aware"]
    labels = ["Fixed one-step", "Readiness-aware"]
    colors = [BLUE, TEAL]
    for rep, marker in ((1, "o"), (2, "s")):
        for group, label, color, x in zip(groups, labels, colors, (0, 1)):
            row = rows[(group, rep)]
            axes[0].scatter(x, row["forecast_rmse"], color=color, marker=marker,
                            s=70, label=f"{label}, seed {1000+rep}" if rep == 1 else None,
                            zorder=3)
            axes[1].scatter(x, row["inference_p95_ms"], color=color, marker=marker,
                            s=70, zorder=3)
    axes[0].set_xticks([0, 1], labels)
    axes[0].set_ylabel("RMSE (requests / trace-minute)")
    axes[0].set_title("Forecast error at each arm's horizon", loc="left")
    axes[1].set_xticks([0, 1], labels)
    axes[1].set_ylabel("Inference p95 (ms)")
    axes[1].set_title("Controller inference overhead", loc="left")
    for ax in axes:
        ax.grid(axis="y", color="#DCE4E6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(0.02, 0.98,
                 "Different horizons (1 vs 3 steps):\nnot a like-for-like model score.",
                 transform=axes[0].transAxes, va="top", fontsize=8, color=GREY)
    handles = [plt.Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
                          label="Fixed"),
               plt.Line2D([], [], marker="o", color="none", markerfacecolor=TEAL,
                          label="Readiness-aware"),
               plt.Line2D([], [], marker="o", color=INK, linestyle="none",
                          label="Seed 1001"),
               plt.Line2D([], [], marker="s", color=INK, linestyle="none",
                          label="Seed 1002")]
    fig.legend(handles=handles, frameon=False, ncols=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "forecast_overhead.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_readiness_horizon():
    measured = json.loads((ROOT / "experiments" / "measured_constants.json").read_text())
    slow = json.loads((Path(__file__).resolve().parent / "slow_start_coldstart.json").read_text())
    interval = 6.0
    values = [float(measured["pod_cold_start_p95_wall_s"]),
              float(slow["all_ready"]["p95"])]
    labels = ["Natural startup\n(5 trials)", "Controlled delay\n(20 trials)"]
    horizons = [max(1, math.ceil((value + 0.01) / interval)) for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    bars = axes[0].bar(labels, values, color=[BLUE, ORANGE], width=0.55, zorder=3)
    axes[0].axhline(interval, color=RED, linestyle="--", linewidth=1.5,
                    label="Control interval = 6 s")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.2,
                     f"{value:.3f} s", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Time until all 5 pods Ready (s)")
    axes[0].set_title("Measured readiness p95", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
    axes[0].spines[["top", "right"]].set_visible(False)

    bars = axes[1].bar(labels, horizons, color=[BLUE, ORANGE], width=0.55, zorder=3)
    axes[1].axhline(1, color=GREY, linestyle=":", linewidth=1.2,
                    label="Fixed one-step baseline")
    for bar, horizon in zip(bars, horizons):
        axes[1].text(bar.get_x() + bar.get_width() / 2, horizon + 0.08,
                     f"{horizon} steps", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Selected recursive forecast horizon (steps)")
    axes[1].set_ylim(0, max(horizons) + 1)
    axes[1].set_title("Horizon derived from readiness", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("At compression 10, one trace-minute equals 6 wall-seconds",
                 x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "readiness_horizon.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_request_errors(rows):
    ordered = [rows[(arm, rep)] for arm in ("fixed_horizon", "readiness_aware")
               for rep in (1, 2)]
    labels = ["Fixed r1", "Fixed r2", "Aware r1", "Aware r2"]
    rates = [100 * row["requests_error"] / max(row["requests_total"], 1)
             for row in ordered]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar(labels, rates, color=[BLUE, BLUE, TEAL, TEAL], width=0.62,
                  zorder=3)
    for bar, row in zip(bars, ordered):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00015,
                f"{row['requests_error']} error(s)", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel("Transport / HTTP errors (% of received requests)")
    ax.set_title("Request outcomes: one error observed in optimized rep 2", loc="left")
    ax.set_ylim(0, max(rates, default=0.01) * 1.7 + 0.0005)
    ax.grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "request_errors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_accuracy():
    table = ROOT / "results/presentation_nasa_20260929/tables/table_nasa_accuracy.csv"
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    paper = {("ARIMA", 1): 14.010, ("Bi-LSTM", 1): 13.551,
             ("ARIMA", 5): 15.414, ("Bi-LSTM", 5): 14.390}
    names = [("ARIMA", 1), ("Bi-LSTM", 1), ("ARIMA", 5), ("Bi-LSTM", 5)]
    actual = {(r["model"], int(r["steps"])): float(r["RMSE"]) for r in rows}
    labels = [f"{model}\n{step}-step" for model, step in names]
    x = list(range(len(names)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    a = ax.bar([i - width / 2 for i in x], [paper[k] for k in names], width,
               color=GREY, label="Paper Table 4", zorder=3)
    b = ax.bar([i + width / 2 for i in x], [actual[k] for k in names], width,
               color=TEAL, label="Full rerun, seed 42", zorder=3)
    add_values(ax, a)
    add_values(ax, b)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (requests / minute)")
    ax.set_title("NASA forecast baseline reproduction", loc="left")
    ax.grid(axis="y", color="#DCE4E6", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "nasa_baseline_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, title, detail, color=TEAL, title_color="white"):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.03,rounding_size=0.08",
                         linewidth=1.6, edgecolor=color, facecolor=PALE)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.33, detail, ha="center", va="center",
            fontsize=8, color=INK, wrap=True)
    return box


def arrow(ax, start, end, color=GREY, rad=0.0, lw=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture(title, optimized=False):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.2, 6.62, title, fontsize=17, fontweight="bold", color=INK)
    add_box(ax, (0.25, 4.95), (2.1, 1.0), "Workload", "NASA trace / open-loop requests", BLUE)
    add_box(ax, (3.05, 4.95), (2.0, 1.0), "Service route", "NodePort / Kubernetes Service", BLUE)
    add_box(ax, (5.75, 4.95), (2.35, 1.0), "Application pods", "HTTP service replicas", BLUE)
    add_box(ax, (9.0, 4.95), (2.0, 1.0), "Responses", "latency / success / errors", BLUE)
    arrow(ax, (2.35, 5.45), (3.05, 5.45), BLUE)
    arrow(ax, (5.05, 5.45), (5.75, 5.45), BLUE)
    arrow(ax, (8.1, 5.45), (9.0, 5.45), BLUE)

    add_box(ax, (0.35, 2.85), (2.0, 1.05), "Metric collection", "throughput + pod state" if not optimized else "http_requests_total", TEAL)
    add_box(ax, (2.95, 2.85), (2.0, 1.05), "Prometheus", "time-series store / query API", TEAL)
    if optimized:
        add_box(ax, (5.55, 2.85), (2.1, 1.05), "Hybrid forecaster", "Prophet + residual Bi-LSTM\n1-step model", TEAL)
        add_box(ax, (8.25, 2.55), (2.35, 1.65), "Readiness-aware planner",
                "lead = ceil((Ready p95 + overhead) / tick)\nrecursive forecast; pending/Ready guard", ORANGE)
        add_box(ax, (11.15, 2.85), (1.55, 1.05), "Executor", "RBAC-limited\nscale patch", TEAL)
        ax.text(9.4, 2.25, "OPTIMIZATION INTRODUCED HERE", ha="center", va="center",
                fontsize=9, fontweight="bold", color=ORANGE)
    else:
        add_box(ax, (5.55, 2.85), (2.1, 1.05), "Bi-LSTM", "10-step input; 1/5-step output", TEAL)
        add_box(ax, (8.25, 2.55), (2.35, 1.65), "Adaptation manager",
                "CDT 60 s\nRRS 60%; pods_min 10; 500 req/min/pod", TEAL)
        add_box(ax, (11.15, 2.85), (1.55, 1.05), "Kubernetes API", "replica\nupdate", TEAL)
    arrow(ax, (6.9, 4.95), (1.35, 3.9), GREY, rad=-0.2)
    arrow(ax, (2.35, 3.38), (2.95, 3.38), TEAL)
    arrow(ax, (4.95, 3.38), (5.55, 3.38), TEAL)
    arrow(ax, (7.65, 3.38), (8.25, 3.38), ORANGE if optimized else TEAL)
    arrow(ax, (10.6, 3.38), (11.15, 3.38), ORANGE if optimized else TEAL)
    arrow(ax, (11.9, 2.85), (7.0, 4.95), ORANGE if optimized else TEAL, rad=-0.22)
    ax.text(0.35, 1.15,
            "Closed loop: observed application metrics -> forecast -> replica decision -> Kubernetes scale action -> changed serving capacity.",
            fontsize=10, color=INK)
    note = ("The model is held constant across the live A/B; only the planning horizon changes.\n"
            "Controlled test: startup delay 10 s; measured all-Ready p95 12.715 s; C=10 gives 6 s/tick; horizon=3.") if optimized else (
            "Paper model comparison: NASA / FIFA request traces, chronological 70/30 split,\n"
            "w=10; Bi-LSTM/LSTM versus ARIMA; autoscaler uses fixed workload-per-pod assumption, CDT, RRS.")
    ax.text(0.35, 0.38, note, fontsize=9, color=GREY, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / ("architecture_optimized.png" if optimized else "architecture_baseline.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_run_metrics()
    plot_live_latency(rows)
    plot_slo(rows)
    plot_resources(rows)
    plot_forecast_and_inference(rows)
    plot_readiness_horizon()
    plot_request_errors(rows)
    plot_baseline_accuracy()
    architecture("Baseline paper system: Bi-LSTM forecasting + CDT/RRS planning", False)
    architecture("Implemented optimization: readiness-derived forecast horizon", True)
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()