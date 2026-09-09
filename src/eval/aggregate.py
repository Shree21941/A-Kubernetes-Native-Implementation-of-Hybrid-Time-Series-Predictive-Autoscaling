"""Phase 8 - turn 60 run directories into one defensible comparison table.

    ./.venv/bin/python src/eval/aggregate.py                 # print + write results/
    ./.venv/bin/python src/eval/aggregate.py --exclude-flagged

Writes results/phase7_table.md, results/phase7_table.csv, results/phase7_summary.json.

WHY THIS EXISTS SEPARATELY FROM experiments/aggregate.py. That script is a PRE-FLIGHT
check: it asks "is this dataset trustworthy?" and prints pass/fail. This one assumes the
answer was yes and asks "what does it say?". Keeping them apart means the integrity
checks cannot quietly become a knob you turn until the results look good.

CONFIDENCE INTERVALS, NOT JUST STANDARD DEVIATION. Phase 7 required "mean +/- standard
deviation or a confidence interval". With n=5 the two are very different claims: sd
describes the spread of the five runs we did, a CI describes where the true mean of the
underlying process probably sits. The comparison we actually want to make is "is arm A's
mean different from arm B's mean", so the CI is the honest instrument. With n=5 the
normal approximation is wrong - the 95% multiplier is Student's t with 4 degrees of
freedom, 2.776, not 1.96. Using 1.96 here would understate every interval by 42%.

n=5 IS SMALL AND THE INTERVALS ARE WIDE. That is not a defect in the reporting, it is
the truth about the dataset, and overlapping intervals are reported as overlapping
rather than being resolved by picking a different statistic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments" / "runs"
RESULTS = REPO / "results"

ARMS = ["hpa-cpu", "keda-rate", "predictive-prophet", "predictive-hybrid"]
SEGMENTS = ["diurnal", "burst", "quiet"]

# Student's t, 95% two-sided, by degrees of freedom (n-1). n=5 -> df=4 -> 2.776.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}

# (label, path in metrics.json, format, lower-is-better)
METRICS = [
    ("p95 latency (ms)",      ("latency", "latency_p95_ms"),        "{:.1f}",  True),
    ("p99 latency (ms)",      ("latency", "latency_p99_ms"),        "{:.1f}",  True),
    ("SLO violation %",       ("latency", "slo_violation_rate"),    "{:.3f}",  True),
    ("mean replicas",         ("cluster", "mean_replicas"),         "{:.2f}",  True),
    ("CPU-seconds",           ("cpu", "cpu_seconds_total"),         "{:.1f}",  True),
    ("replica-seconds",       ("cluster", "replica_seconds"),       "{:.0f}",  True),
    ("scaling actions",       ("cluster", "scaling_actions"),       "{:.1f}",  True),
    ("replica churn",         ("cluster", "replica_churn"),         "{:.1f}",  True),
    ("direction reversals",   ("cluster", "direction_reversals"),   "{:.1f}",  True),
    ("pending-pod seconds",   ("cluster", "pending_pod_seconds"),   "{:.1f}",  True),
    ("forecast MAPE %",       ("forecast", "forecast_mape_pct"),    "{:.1f}",  True),
    ("forecast RMSE",         ("forecast", "forecast_rmse"),        "{:.2f}",  True),
    ("inference p50 (ms)",    ("forecast", "inference_p50_ms"),     "{:.2f}",  True),
    ("inference p95 (ms)",    ("forecast", "inference_p95_ms"),     "{:.2f}",  True),
]

# Runs flagged by the Phase 7 integrity pass. Kept in the primary table - CLAUDE.md 8:
# a failing check is a reason to look, not to discard - but --exclude-flagged reruns the
# whole analysis without them so the reader can see how much any conclusion depends on
# them. A finding that survives both is a finding; one that does not is a coin flip.
FLAGGED = {
    ("predictive-prophet", "diurnal", 3):
        "pod-scheduling stall: 299.9s pending vs 18.9s median across the sweep",
    ("hpa-cpu", "diurnal", 1): "host suspend mid-run (5106s); cluster metrics rescored",
    ("hpa-cpu", "diurnal", 5): "host suspend mid-run (4038s); cluster metrics rescored",
    ("predictive-prophet", "diurnal", 5):
        "host suspend mid-run (3110s); cluster metrics rescored",
}


def dig(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS.iterdir()):
        f = d / "metrics.json"
        if f.exists():
            m = json.load(f.open())
            m["_dir"] = d.name
            out.append(m)
    return out


def summarise(vals: list[float]) -> dict | None:
    """mean, sd, half-width of the 95% CI, and median, for one cell."""
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    n = len(vals)
    mean = st.mean(vals)
    if n == 1:
        return dict(n=1, mean=mean, sd=0.0, ci=None, median=mean,
                    lo=mean, hi=mean)
    sd = st.stdev(vals)
    # Half-width of the CI on the MEAN: t * sd / sqrt(n). Note the sqrt(n) - the interval
    # shrinks with more runs even though the spread sd does not.
    ci = T95.get(n - 1, 2.0) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, ci=ci, median=st.median(vals),
                lo=mean - ci, hi=mean + ci)


def build(runs: list[dict], exclude_flagged: bool) -> dict:
    cells: dict = {}
    dropped = []
    for arm in ARMS:
        for seg in SEGMENTS:
            sel = [m for m in runs if m["arm"] == arm and m["segment"] == seg]
            if exclude_flagged:
                keep = []
                for m in sel:
                    if (arm, seg, m["rep"]) in FLAGGED:
                        dropped.append(f"{arm}/{seg}/rep{m['rep']}")
                    else:
                        keep.append(m)
                sel = keep
            per = {}
            for label, path, _fmt, _lo in METRICS:
                vals = [dig(m, path) for m in sel]
                # SLO is stored as a fraction; the table reports percent.
                if path == ("latency", "slo_violation_rate"):
                    vals = [v * 100 for v in vals if v is not None]
                per[label] = summarise(vals)
            cells[(arm, seg)] = dict(runs=len(sel), metrics=per)
    return dict(cells=cells, dropped=dropped)


def fmt_cell(s: dict | None, fmt: str) -> str:
    if s is None:
        return "n/a"
    if s["ci"] is None:
        return fmt.format(s["mean"])
    return f"{fmt.format(s['mean'])} ± {fmt.format(s['ci'])}"


def render_markdown(built: dict, exclude_flagged: bool) -> str:
    lines = ["# Phase 7 results - arms x segments", ""]
    lines.append("Mean ± half-width of the 95% confidence interval on the mean "
                 "(Student's t, n=5, df=4, t=2.776).")
    lines.append("")
    if exclude_flagged:
        lines.append(f"**Flagged runs EXCLUDED** ({len(built['dropped'])}): "
                     + ", ".join(built["dropped"]))
    else:
        lines.append("**All runs included**, flagged ones among them. "
                     "See `--exclude-flagged` for the sensitivity check.")
    lines.append("")
    for seg in SEGMENTS:
        lines += [f"## segment: {seg}", "",
                  "| metric | " + " | ".join(ARMS) + " |",
                  "|---|" + "---|" * len(ARMS)]
        for label, path, fmt, _lo in METRICS:
            row = [fmt_cell(built["cells"][(a, seg)]["metrics"][label], fmt)
                   for a in ARMS]
            lines.append(f"| {label} | " + " | ".join(row) + " |")
        ns = {built["cells"][(a, seg)]["runs"] for a in ARMS}
        lines += ["", f"n per arm: {sorted(ns)}", ""]
    return "\n".join(lines)


def write_csv(built: dict, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "arm", "metric", "n", "mean", "sd",
                    "ci95_halfwidth", "ci95_lo", "ci95_hi", "median"])
        for seg in SEGMENTS:
            for arm in ARMS:
                for label, _p, _f, _lo in METRICS:
                    s = built["cells"][(arm, seg)]["metrics"][label]
                    if s is None:
                        continue
                    w.writerow([seg, arm, label, s["n"],
                                f"{s['mean']:.6g}", f"{s['sd']:.6g}",
                                "" if s["ci"] is None else f"{s['ci']:.6g}",
                                f"{s['lo']:.6g}", f"{s['hi']:.6g}",
                                f"{s['median']:.6g}"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="drop the runs flagged by the Phase 7 integrity pass")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    runs = load_runs()
    print(f"loaded {len(runs)} runs from {RUNS}")

    built = build(runs, args.exclude_flagged)
    md = render_markdown(built, args.exclude_flagged)

    suffix = "_excl_flagged" if args.exclude_flagged else ""
    (RESULTS / f"phase7_table{suffix}.md").write_text(md + "\n")
    write_csv(built, RESULTS / f"phase7_table{suffix}.csv")

    summary = {f"{a}|{s}": {
        "n": built["cells"][(a, s)]["runs"],
        "metrics": {k: v for k, v in built["cells"][(a, s)]["metrics"].items()}}
        for a in ARMS for s in SEGMENTS}
    (RESULTS / f"phase7_summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    print(md)
    print(f"\nwrote results/phase7_table{suffix}.md, .csv, "
          f"and phase7_summary{suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
