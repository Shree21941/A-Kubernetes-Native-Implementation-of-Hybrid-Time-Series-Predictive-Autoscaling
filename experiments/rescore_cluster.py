"""Recompute the cluster/replica block of every run's metrics.json, in place.

    ./.venv/bin/python experiments/rescore_cluster.py [--dry-run]

WHY THIS EXISTS. Three diurnal runs in the Phase 7 sweep were recorded while the host
was suspended (laptop lid) for 3110-5106 s in the middle of a 2880 s run. Every metric
in the cluster block is an integral over sampler timestamps, so a frozen hour was being
averaged in at whatever replica count was current when the machine went to sleep.
replica_seconds - the cost axis - came out 1.9x to 2.7x too high.

collect.replica_metrics now excises intervals longer than 10 s before doing any
time-weighted arithmetic. This script re-applies that corrected scorer to runs that were
scored before the fix existed. It recomputes from raw/replicas.jsonl, which was always
correct; only the arithmetic over it was wrong.

WHY IT ONLY TOUCHES THE CLUSTER BLOCK. The cpu block of the early runs was itself
recovered from Prometheus after a separate bug, and Prometheus retention is 2 days.
Re-running the full collector would re-query Prometheus for runs whose data may have
aged out, replacing good backfilled numbers with nulls. Latency and forecast metrics are
read from files that have not changed and do not need recomputing. So: cluster only.

The original values are preserved under cluster.rescore_previous so nothing is
destroyed - a corrected number that cannot be compared against what it replaced is not
much better than the wrong one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import collect  # noqa: E402

RUNS = REPO / "experiments" / "runs"

KEYS_OF_INTEREST = ("mean_replicas", "replica_seconds", "duration_s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    args = ap.parse_args()

    changed = suspended = 0
    for d in sorted(RUNS.iterdir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.load(f.open())
        old = m.get("cluster", {})
        new = collect.replica_metrics(d)

        deltas = [k for k in KEYS_OF_INTEREST
                  if old.get(k) is not None and new.get(k) is not None
                  and abs(old[k] - new[k]) > 0.05]
        if new.get("suspend_gaps"):
            suspended += 1

        if not deltas:
            continue
        changed += 1
        label = f"{m['arm']}/{m['segment']}/rep{m['rep']}"
        gaps = new.get("suspend_gaps", 0)
        print(f"  {label:38s} gaps={gaps} "
              f"({new.get('suspend_gap_seconds', 0):.0f}s frozen)")
        for k in deltas:
            print(f"      {k:20s} {old[k]:>10.2f} -> {new[k]:>10.2f}")

        if not args.dry_run:
            new["rescore_previous"] = {k: old.get(k) for k in KEYS_OF_INTEREST}
            new["rescore_note"] = (
                "cluster metrics recomputed with suspend-gap excision; previous values "
                "integrated over wall-clock time that included a host suspend")
            m["cluster"] = new
            f.write_text(json.dumps(m, indent=2))

    verb = "would change" if args.dry_run else "changed"
    print(f"\n  {changed} run(s) {verb}; {suspended} run(s) contain a suspend gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
