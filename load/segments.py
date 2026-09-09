"""Phase 4 - the three frozen evaluation segments.

    ./.venv/bin/python load/segments.py          # freeze + plot
    ./.venv/bin/python load/segments.py --show   # print what is frozen

Every autoscaler arm in Phases 6-8 replays exactly these three windows, so they are
written out to load/segments.json as literal per-minute request counts rather than as
date ranges to be re-derived. If a later phase regenerated them from the CSV and the CSV
had changed, the arms would silently face different workloads and the comparison would
be void without anything appearing to be wrong.

All three are cut from the TEST portion (from 1995-08-13 09:36), which no model in
Phase 2 was trained on. Using a training window would let the forecaster's arm replay
data it had memorised.

Why these three, specifically:

  diurnal  1995-08-28, a full 24 h. Of the 18 complete days in the test portion this is
           the only one with ZERO zero-traffic minutes, and its hourly profile
           correlates 0.988 with the test-set mean. It is the "does the autoscaler track
           an ordinary day" case.

  burst    1995-08-24 06:52-09:51. Contains the sharpest sustained ramp in the test
           portion measured in absolute terms: a prior-hour median of 14.5 req/min rises
           to a 10-minute peak of 76 within minutes. Sharper ratios exist (7.3x on
           08-26) but from a base of 5.5 req/min, where the absolute load is too small
           to make an autoscaler do anything.

  quiet    1995-08-31 15:50-17:19. A near-total dropout in the middle of a busy
           afternoon: per-10-minute means run 98.5, 96.5, 73.3, **8.4**, 87.9 - a
           collapse to near zero and a full recovery inside 30 minutes, including a
           9-minute run of exact zeros. This is the flapping test. An autoscaler that
           scales down eagerly will remove pods it needs again 10 minutes later, and
           will be caught here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src" / "models")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dataset import load_series, split_at  # noqa: E402

OUT = REPO / "load" / "segments.json"
RESULTS = REPO / "results"

SEGMENTS = {
    "diurnal": dict(
        start="1995-08-28 10:00", end="1995-08-28 17:59",
        purpose="ordinary daily cycle: ramp -> peak -> decline",
        why="8h window of 1995-08-28, the only complete test-set day with zero "
            "zero-traffic minutes. Chosen by maximising min(rise, fall) around a "
            "mid-window peak, so the segment exercises scale-UP and scale-DOWN "
            "roughly equally: smoothed load climbs 37.3 req/min into the 15:00 peak "
            "(raw max 118) and falls 35.8 after it. Shortened from the full 24h - "
            "see DEVIATIONS.md D-018 for why and what it costs.",
    ),
    "burst": dict(
        start="1995-08-24 06:52", end="1995-08-24 09:51",
        purpose="sharp ramp: can the autoscaler get ahead of a spike",
        why="largest absolute ramp in the test portion - prior-hour median "
            "14.5 req/min to a 10-minute peak of 76",
    ),
    "quiet": dict(
        start="1995-08-31 15:50", end="1995-08-31 17:19",
        purpose="dropout and recovery: scale-down correctness and flapping",
        why="traffic collapses from ~97 req/min to a 9-minute run of exact zeros "
            "and recovers to ~88 within 30 minutes",
    ),
}


def build() -> dict:
    s = load_series()
    cut = split_at(len(s))
    test_start = s.index[cut]

    out = {"source": "data/processed/nasa_requests_per_min.csv",
           "test_split_starts": str(test_start), "segments": {}}

    for name, meta in SEGMENTS.items():
        g = s.loc[meta["start"]:meta["end"]]
        if g.index[0] < test_start:
            raise ValueError(
                f"segment '{name}' starts {g.index[0]}, before the test split at "
                f"{test_start} - it would replay data the models trained on"
            )
        y = [int(v) for v in g.to_numpy()]
        # Checksum over the actual values, so a later phase can assert it is replaying
        # the same bytes rather than trusting the date range.
        digest = hashlib.sha256(
            ",".join(map(str, y)).encode()
        ).hexdigest()[:16]

        out["segments"][name] = dict(
            **meta,
            minutes=len(y),
            first_minute=str(g.index[0]),
            last_minute=str(g.index[-1]),
            total_requests=int(sum(y)),
            mean_rpm=round(float(g.mean()), 2),
            min_rpm=int(g.min()), max_rpm=int(g.max()),
            zero_minutes=int((g == 0).sum()),
            sha256_16=digest,
            y=y,
        )
    return out


def plot(frozen: dict) -> None:
    s = load_series()
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = list(frozen["segments"])
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3.2 * len(names)))
    for ax, name in zip(axes, names):
        seg = frozen["segments"][name]
        g = s.loc[seg["first_minute"]:seg["last_minute"]]
        ax.plot(g.index, g.to_numpy(), lw=0.8, color="#1f77b4")
        ax.fill_between(g.index, 0, g.to_numpy(), alpha=0.15, color="#1f77b4")
        ax.set_title(f"{name} - {seg['purpose']}  "
                     f"({seg['minutes']} min, mean {seg['mean_rpm']} req/min, "
                     f"max {seg['max_rpm']}, {seg['zero_minutes']} zero minutes)",
                     fontsize=10)
        ax.set_ylabel("req / min")
        ax.set_xlim(g.index[0], g.index[-1])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.suptitle("Phase 4 - frozen evaluation segments (all from the test split)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_phase4_segments.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true",
                    help="print the frozen file instead of rebuilding it")
    args = ap.parse_args()

    if args.show:
        if not OUT.exists():
            print("not frozen yet - run without --show")
            return 1
        frozen = json.loads(OUT.read_text())
    else:
        frozen = build()
        OUT.write_text(json.dumps(frozen, indent=2))
        plot(frozen)

    print()
    print("Phase 4 - frozen evaluation segments")
    print("=" * 100)
    print(f"{'name':<9} {'window':<38} {'min':>5} {'mean':>7} {'max':>5} "
          f"{'zeros':>6} {'total':>8} {'sha':>17}")
    print("-" * 100)
    for name, seg in frozen["segments"].items():
        print(f"{name:<9} {seg['first_minute'][:16]} -> {seg['last_minute'][11:16]:<19} "
              f"{seg['minutes']:>5} {seg['mean_rpm']:>7.1f} {seg['max_rpm']:>5} "
              f"{seg['zero_minutes']:>6} {seg['total_requests']:>8,} "
              f"{seg['sha256_16']:>17}")
    print("=" * 100)
    for name, seg in frozen["segments"].items():
        print(f"\n  {name}: {seg['purpose']}")
        print(f"    {seg['why']}")
    if not args.show:
        print(f"\nwrote {OUT.relative_to(REPO)} and "
              f"results/fig_phase4_segments.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
