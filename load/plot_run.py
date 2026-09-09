"""Phase 4 - plot a replay run: offered load vs trace, and the latency distribution.

    ./.venv/bin/python load/plot_run.py experiments/<run-id>

This is the visual evidence for the Definition-of-Done item "replaying a segment
produces a load curve whose shape matches the trace".

On reading the correlation. Achieved counts will never correlate 1.000 with the trace,
and they should not: arrivals are drawn from a Poisson process, so the count in a
trace-minute whose target is m has standard deviation sqrt(m). The attainable ceiling is

    corr_max = sd(target) / sqrt(sd(target)^2 + mean(target))

which this script computes alongside the measured value. A run at or near that ceiling
is sampling correctly; a run materially BELOW it means the generator failed to keep up -
the schedule slipped, or requests were shed - and that is a real defect worth chasing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO / args.run
    summary = json.loads((run / "summary.json").read_text())

    achieved = np.array(summary["achieved_per_trace_minute"], dtype=float)
    target = np.array(summary["target_per_trace_minute"], dtype=float)

    # Poisson ceiling on the correlation - see the module docstring.
    sd, mean = target.std(), target.mean()
    corr_max = sd / np.sqrt(sd ** 2 + mean) if sd > 0 else 0.0
    corr = summary["shape_correlation"]

    lat = []
    with open(run / "raw" / "requests.csv") as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "200":
                lat.append(float(row["latency_ms"]))
    lat = np.array(sorted(lat))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(target, lw=1.6, color="#1f77b4", label="trace (target)")
    ax.plot(achieved, lw=0.9, color="#d62728", alpha=0.85, label="achieved")
    ax.set_title(f"Offered load vs trace - segment '{summary['segment']}' "
                 f"({summary['compression']:g}x compression)   "
                 f"r = {corr:.4f}  (Poisson ceiling {corr_max:.4f})")
    ax.set_xlabel("trace minute")
    ax.set_ylabel("requests / trace-minute")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, len(target) - 1)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.scatter(target, achieved, s=10, alpha=0.5, color="#1f77b4",
               edgecolors="none")
    lim = float(max(target.max(), achieved.max()))
    ax.plot([0, lim], [0, lim], ls="--", color="#d62728", lw=1)
    ax.set_title("Achieved vs target per trace-minute "
                 "(spread is Poisson arrival noise, not scheduling error)")
    ax.set_xlabel("target")
    ax.set_ylabel("achieved")
    ax.grid(alpha=0.25)

    ax = axes[2]
    if len(lat):
        q = np.arange(1, len(lat) + 1) / len(lat) * 100
        ax.plot(lat, q, lw=1.4, color="#1f77b4")
        for p, c in ((50, "#999999"), (95, "#ff7f0e"), (99, "#d62728")):
            v = float(np.percentile(lat, p))
            ax.axvline(v, color=c, ls="--", lw=1)
            ax.text(v, 4, f" p{p}={v:.1f}ms", color=c, fontsize=9, rotation=90,
                    va="bottom")
        ax.set_xscale("log")
    ax.set_title(f"Latency distribution - {summary['requests_ok']:,} successful "
                 f"requests, {summary['requests_error']:,} errors")
    ax.set_xlabel("latency (ms, log scale)")
    ax.set_ylabel("percentile")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = args.out or (run / "fig_replay.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print(f"  shape correlation : {corr:.4f}")
    print(f"  Poisson ceiling   : {corr_max:.4f}")
    verdict = ("at the sampling ceiling - generator kept the schedule"
               if corr >= corr_max - 0.02
               else "BELOW the ceiling - the generator fell behind, investigate")
    print(f"  verdict           : {verdict}")
    print(f"  wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
