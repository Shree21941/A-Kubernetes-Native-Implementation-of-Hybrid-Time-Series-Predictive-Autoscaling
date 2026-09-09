"""Phase 1 verification - look at the series before trusting it.

    ./.venv/bin/python src/data/explore.py

Reads data/processed/nasa_requests_per_min.csv and produces:

  results/fig_phase1_full_series.png   - the whole two months (cf. paper Figure 3)
  results/fig_phase1_typical_week.png  - one week, to expose the daily cycle
  results/fig_phase1_erin.png          - the Hurricane Erin outage in context

plus a printed table of the longest zero-traffic runs, and an explicit pass/fail
check that the Hurricane Erin outage is present.

Why bother plotting at all: the summary statistics printed by preprocess_nasa.py
(row count, total requests, min/mean/max) would all look perfectly healthy on a series
whose minutes were shuffled, shifted by an hour, or off by a day. The shape is the only
thing that catches those. Eyeballing the daily cycle is a real test - if the peaks do
not land in the working day, the timezone handling is wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this machine; write files, never pop a window

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# CLAUDE.md 6: the server was down for Hurricane Erin and recorded no requests from
# 01/Aug/1995:14:52:01 to 03/Aug/1995:04:36:13. The paper keeps this window as genuine
# zero traffic rather than synthesising data, and so do we.
ERIN_FROM = pd.Timestamp("1995-08-01 14:52")
ERIN_TO = pd.Timestamp("1995-08-03 04:36")

# 1995-07-03 was a Monday, so this is a clean Mon-Sun week well clear of both the
# Hurricane Erin outage and the July trace truncation.
WEEK_FROM = pd.Timestamp("1995-07-03 00:00")
WEEK_TO = pd.Timestamp("1995-07-09 23:59")


def zero_runs(s: pd.Series) -> pd.DataFrame:
    """Return every maximal run of consecutive zero-traffic minutes.

    A "gap" in this dataset is not a missing row - preprocess_nasa.py guarantees the
    index is complete - it is a run of zeros. Those runs are the thing worth auditing,
    because they are where the two very different causes live: a genuine server outage
    versus the end of a trace file.
    """
    is_zero = (s.to_numpy() == 0)
    if not is_zero.any():
        return pd.DataFrame(columns=["start", "end", "minutes"])

    # Find the boundaries by looking at where the boolean flips. Padding with False on
    # both ends makes runs that touch the start or end of the series fall out of the
    # same diff, with no special-casing.
    padded = np.concatenate(([False], is_zero, [False]))
    flips = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = flips[0::2], flips[1::2]  # ends are exclusive

    return pd.DataFrame(
        {
            "start": s.index[starts],
            "end": s.index[ends - 1],
            "minutes": ends - starts,
        }
    ).sort_values("minutes", ascending=False, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv", type=Path,
        default=REPO / "data" / "processed" / "nasa_requests_per_min.csv",
    )
    ap.add_argument("--outdir", type=Path, default=REPO / "results")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found - run src/data/preprocess_nasa.py first.")
        return 1

    df = pd.read_csv(args.csv, parse_dates=["ds"])
    s = df.set_index("ds")["y"]
    args.outdir.mkdir(parents=True, exist_ok=True)

    ok = True

    # --- Structural checks -------------------------------------------------
    expected_index = pd.date_range(s.index[0], s.index[-1], freq="min")
    continuous = len(s) == len(expected_index) and s.index.equals(expected_index)
    monotonic = s.index.is_monotonic_increasing
    dupes = int(s.index.duplicated().sum())

    print()
    print("Phase 1 - series verification")
    print("=" * 74)
    print(f"  rows                      : {len(s):,}")
    print(f"  range                     : {s.index[0]}  ->  {s.index[-1]}")
    print(f"  index continuous (1 min)  : {continuous}")
    print(f"  index sorted ascending    : {monotonic}")
    print(f"  duplicate timestamps      : {dupes}")
    print(f"  timezone-naive            : {s.index.tz is None}")
    print(f"  total requests            : {int(s.sum()):,}")
    print(f"  req/min  min/mean/max     : {s.min()} / {s.mean():.2f} / {s.max()}")
    print()

    if not (continuous and monotonic and dupes == 0 and s.index.tz is None):
        print("FAIL: index is not a clean, continuous, naive 1-minute grid.")
        ok = False

    # --- Hour-of-day profile: the timezone sanity test ---------------------
    by_hour = s.groupby(s.index.hour).mean()
    peak_hour, trough_hour = int(by_hour.idxmax()), int(by_hour.idxmin())
    print("  mean req/min by hour of day (local, -0400):")
    for h in range(0, 24, 3):
        bar = "#" * int(by_hour[h] / by_hour.max() * 40)
        print(f"    {h:02d}:00  {by_hour[h]:6.1f}  {bar}")
    print(f"  busiest hour {peak_hour:02d}:00, quietest hour {trough_hour:02d}:00")
    # A web server's traffic peaks during the working day. If the peak landed at 03:00
    # we would have applied the -0400 offset in the wrong direction, or mixed naive and
    # aware timestamps - a bug that no count-based check can see.
    if not (8 <= peak_hour <= 20):
        print(f"FAIL: peak hour {peak_hour:02d}:00 is not in the working day - "
              f"suspect the timezone handling.")
        ok = False
    print()

    # --- Zero-traffic runs -------------------------------------------------
    runs = zero_runs(s)
    total_zero = int((s == 0).sum())
    print(f"  zero-traffic minutes      : {total_zero:,} ({total_zero / len(s):.2%})")
    print(f"  distinct zero runs        : {len(runs):,}")
    print()
    print("  10 longest zero-traffic runs:")
    print(f"    {'start':<20} {'end':<20} {'minutes':>8}  {'duration':>12}")
    for r in runs.head(10).itertuples():
        hours, mins = divmod(int(r.minutes), 60)
        print(f"    {str(r.start):<20} {str(r.end):<20} {r.minutes:>8,}  "
              f"{hours:>6}h {mins:02d}m")
    print()

    # --- Hurricane Erin ----------------------------------------------------
    # Find the run that covers the documented outage, then report its true extent
    # rather than asserting the documented boundary minutes are themselves zero: the
    # outage began at 14:52:01, so minute 14:52 can legitimately hold the requests
    # that arrived in its first second.
    covering = runs[(runs["start"] <= ERIN_TO) & (runs["end"] >= ERIN_FROM)]
    print("  Hurricane Erin outage check")
    print(f"    documented (CLAUDE.md 6) : {ERIN_FROM}  ->  {ERIN_TO}")
    if covering.empty:
        print("    FAIL: no zero-traffic run overlaps the documented outage window.")
        ok = False
    else:
        run = covering.iloc[0]
        hours, mins = divmod(int(run["minutes"]), 60)
        interior = s.loc[ERIN_FROM + pd.Timedelta(minutes=1):ERIN_TO - pd.Timedelta(minutes=1)]
        print(f"    observed zero run        : {run['start']}  ->  {run['end']}")
        print(f"    length                   : {run['minutes']:,} minutes "
              f"({hours}h {mins:02d}m)")
        print(f"    y at {ERIN_FROM} : {int(s.loc[ERIN_FROM])}  "
              f"(nonzero is expected - outage starts at :01s)")
        print(f"    y at {ERIN_TO} : {int(s.loc[ERIN_TO])}  "
              f"(nonzero is expected - service returns at :13s)")
        print(f"    interior minutes all zero: {bool((interior == 0).all())} "
              f"({len(interior):,} minutes checked)")
        if not (interior == 0).all():
            print("    FAIL: the documented outage window contains traffic.")
            ok = False
        else:
            print("    PASS: outage present as a contiguous zero block.")
    print()

    # --- Plots -------------------------------------------------------------
    # Full series - the direct visual comparison against the paper's Figure 3.
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(s.index, s.to_numpy(), linewidth=0.25, color="#1f77b4")
    ax.set_title("NASA-HTTP: requests per minute, 1 Jul - 31 Aug 1995 "
                 "(cf. paper Figure 3)")
    ax.set_xlabel("time (local, UTC-0400)")
    ax.set_ylabel("requests / minute")
    ax.set_xlim(s.index[0], s.index[-1])
    ax.set_ylim(bottom=0)
    for _, r in runs.head(3).iterrows():
        ax.axvspan(r["start"], r["end"], color="#d62728", alpha=0.18, zorder=0)
    ax.text(
        0.005, 0.95,
        "red bands: 3 longest zero-traffic runs",
        transform=ax.transAxes, va="top", fontsize=8, color="#d62728",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p_full = args.outdir / "fig_phase1_full_series.png"
    fig.savefig(p_full, dpi=140)
    plt.close(fig)

    # Typical week - the full series is too dense to show the daily cycle; this is
    # where you actually see the structure Prophet is meant to learn.
    wk = s.loc[WEEK_FROM:WEEK_TO]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(wk.index, wk.to_numpy(), linewidth=0.6, color="#1f77b4")
    ax.set_title(f"Typical week: {WEEK_FROM.date()} (Mon) - {WEEK_TO.date()} (Sun)")
    ax.set_xlabel("time (local, UTC-0400)")
    ax.set_ylabel("requests / minute")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_xlim(wk.index[0], wk.index[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p_week = args.outdir / "fig_phase1_typical_week.png"
    fig.savefig(p_week, dpi=140)
    plt.close(fig)

    # Erin in context - a day either side, so the outage reads as an outage rather
    # than as the end of the data.
    erin = s.loc[ERIN_FROM - pd.Timedelta(days=1):ERIN_TO + pd.Timedelta(days=1)]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(erin.index, erin.to_numpy(), linewidth=0.6, color="#1f77b4")
    ax.axvspan(ERIN_FROM, ERIN_TO, color="#d62728", alpha=0.18)
    ax.set_title("Hurricane Erin outage (red) with one day either side - "
                 "kept as genuine zero traffic, not imputed")
    ax.set_xlabel("time (local, UTC-0400)")
    ax.set_ylabel("requests / minute")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.set_xlim(erin.index[0], erin.index[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p_erin = args.outdir / "fig_phase1_erin.png"
    fig.savefig(p_erin, dpi=140)
    plt.close(fig)

    for p in (p_full, p_week, p_erin):
        print(f"  wrote {p.relative_to(REPO)}")
    print("=" * 74)

    if not ok:
        print("\nVERIFICATION FAILED - see FAIL lines above.")
        return 1
    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
