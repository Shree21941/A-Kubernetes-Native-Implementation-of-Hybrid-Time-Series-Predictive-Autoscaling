"""Phase 1 - turn the raw NASA-HTTP access logs into a per-minute request-rate series.

    ./.venv/bin/python src/data/preprocess_nasa.py

Input : data/raw/NASA_access_log_Jul95.gz, data/raw/NASA_access_log_Aug95.gz
Output: data/processed/nasa_requests_per_min.csv  (columns: ds,y)

The whole job is "count log lines per minute", but every later phase consumes this one
file, so the script is written to *prove* it did that correctly rather than to assume
it. It reports the number of lines it refused to parse, the timezone offsets it saw,
the per-file date ranges, and - most importantly - whether the total it counted matches
the documented 3,461,612 lines in the two traces. A parser that silently drops 5% of
July would look exactly like a bad model in Phase 2.

Design notes that matter:

* We count EVERY log line as one request, including 4xx and 5xx responses. The scaling
  signal is load *arriving at the server*, and a request that 404s still consumed a
  connection, a thread and CPU. Filtering by status would understate demand.

* We never parse a full datetime per line. 3.46 M `strptime` calls are slow and
  pointless when the output only has minute resolution: instead we slice the 17-byte
  minute prefix out of the timestamp and use it as a dict key, then convert the ~89 k
  distinct keys once at the end.
"""

from __future__ import annotations

import argparse
import gzip
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Documented line counts from the Internet Traffic Archive. Phase 0 verified the files
# against these; we re-state them here so this script can attribute a shortfall to the
# parser rather than to the data.
DOCUMENTED_LINES = {
    "NASA_access_log_Jul95.gz": 1_891_714,
    "NASA_access_log_Aug95.gz": 1_569_898,
}

# Explicit month map instead of strptime's %b: %b is locale-dependent, and a machine
# with a non-English locale would silently fail to parse "Jul". The logs are always
# English regardless of where we run them.
MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# A well-formed CLF timestamp field is exactly 28 bytes: [dd/Mon/yyyy:HH:MM:SS +zzzz]
#                                                        0         1         2
#                                                        0123456789012345678901234567
TS_LEN = 28
MINUTE_SLICE = slice(1, 18)   # b"01/Jul/1995:00:00"  - the dict key
TZ_SLICE = slice(22, 27)      # b"-0400"


def minute_key_to_datetime(key: str) -> datetime:
    """Convert a b'dd/Mon/yyyy:HH:MM' key into a naive datetime.

    Naive on purpose. Every record in both traces carries the same -0400 offset (the
    script asserts this below), so stripping the offset leaves a consistent local-time
    index. Mixing tz-aware and tz-naive datetimes is a classic silent bug once Prophet
    and pandas reindexing get involved, so we make the whole pipeline naive at the
    single point where the timezone is known to be uniform.
    """
    return datetime(
        year=int(key[7:11]),
        month=MONTHS[key[3:6]],
        day=int(key[0:2]),
        hour=int(key[12:14]),
        minute=int(key[15:17]),
    )


def count_minutes(path: Path) -> tuple[Counter, int, int, Counter]:
    """Count requests per minute in one gzipped log.

    Returns (counts_by_minute_key, lines_read, lines_skipped, timezone_offsets_seen).
    """
    counts: Counter = Counter()
    offsets: Counter = Counter()
    read = 0
    skipped = 0

    # 'rb' + manual slicing: the logs are not UTF-8 clean (they contain raw bytes from
    # malformed client requests), so decoding every line would either raise or cost a
    # latin-1 decode we do not need. We only ever touch the ASCII timestamp field.
    with gzip.open(path, "rb") as fh:
        for raw in fh:
            read += 1
            i = raw.find(b"[")
            if i == -1:
                skipped += 1
                continue
            ts = raw[i:i + TS_LEN]
            # Structural validation, not a regex: a regex over 3.46 M lines is ~10x
            # slower and buys nothing here, because the two anchors below already
            # pin the field length and shape.
            if len(ts) != TS_LEN or ts[-1:] != b"]" or ts[21:22] != b" ":
                skipped += 1
                continue
            try:
                key = ts[MINUTE_SLICE].decode("ascii")
                # Cheap correctness gate: the separators must be where CLF says they
                # are. Catches truncated or interleaved records without a full parse.
                if key[2] != "/" or key[6] != "/" or key[11] != ":" or key[14] != ":":
                    skipped += 1
                    continue
                if key[3:6] not in MONTHS:
                    skipped += 1
                    continue
            except UnicodeDecodeError:
                skipped += 1
                continue
            counts[key] += 1
            offsets[ts[TZ_SLICE].decode("ascii", "replace")] += 1

    return counts, read, skipped, offsets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data" / "raw")
    ap.add_argument(
        "--out", type=Path,
        default=REPO / "data" / "processed" / "nasa_requests_per_min.csv",
    )
    args = ap.parse_args()

    total_counts: Counter = Counter()
    total_offsets: Counter = Counter()
    total_read = 0
    total_skipped = 0
    per_file: list[tuple[str, int, int, int, str, str]] = []

    for fname in DOCUMENTED_LINES:
        path = args.raw_dir / fname
        if not path.exists():
            print(f"ERROR: missing input {path}")
            return 1

        print(f"parsing {fname} ...", flush=True)
        counts, read, skipped, offsets = count_minutes(path)

        # Sort the DATETIMES, never the raw keys. The keys are dd/Mon/yyyy strings, so
        # a lexicographic sort puts "01/Aug" before "01/Jul" and orders days before
        # months. An earlier version of this script did exactly that, built the minute
        # grid from the resulting bogus min/max, and silently dropped all of July in
        # the reindex - it reported a plausible-looking 44,640-row CSV.
        stamps = [minute_key_to_datetime(k) for k in counts]
        per_file.append(
            (fname, read, skipped, sum(counts.values()), min(stamps), max(stamps))
        )
        total_counts.update(counts)
        total_offsets.update(offsets)
        total_read += read
        total_skipped += skipped

    # --- Build the continuous minute grid ---------------------------------
    observed = pd.Series(
        {minute_key_to_datetime(k): v for k, v in total_counts.items()},
        dtype="int64",
    ).sort_index()
    start = observed.index[0].to_pydatetime()
    end = observed.index[-1].to_pydatetime()

    # The reindex is the whole point of this step: a minute with no traffic emits no
    # log lines, so it is simply ABSENT above. Filling with 0 turns "not in the file"
    # into the assertion we actually mean - "no requests arrived" - and gives the
    # models the evenly spaced index they require.
    grid = pd.date_range(start=start, end=end, freq="min")
    series = observed.reindex(grid, fill_value=0)
    series.index.name = "ds"
    series.name = "y"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_csv(args.out, index=True)

    # --- Sanity report ----------------------------------------------------
    documented_total = sum(DOCUMENTED_LINES.values())
    counted_total = int(series.sum())
    zero_minutes = int((series == 0).sum())

    print()
    print("Phase 1 - NASA log -> per-minute request rate")
    print("=" * 74)
    print(f"{'file':<28} {'lines read':>12} {'skipped':>9} {'counted':>12}")
    for fname, read, skipped, counted, first, last in per_file:
        print(f"{fname:<28} {read:>12,} {skipped:>9,} {counted:>12,}")
        print(f"{'':<28} range {first}  ->  {last}")
    print("-" * 74)
    print(f"{'TOTAL':<28} {total_read:>12,} {total_skipped:>9,} {counted_total:>12,}")
    print()
    print(f"  timezone offsets seen : {dict(total_offsets)}")
    print(f"  documented log lines  : {documented_total:,}")
    print(f"  requests counted      : {counted_total:,}  "
          f"({counted_total / documented_total:.4%} of documented)")
    print()
    print(f"  output                : {args.out.relative_to(REPO)}")
    print(f"  rows (minutes)        : {len(series):,}")
    print(f"  date range            : {series.index[0]}  ->  {series.index[-1]}")
    print(f"  span                  : {(end - start).days} days "
          f"{(end - start).seconds // 3600} hours")
    print(f"  req/min  min / mean / max : "
          f"{series.min()} / {series.mean():.2f} / {series.max()}")
    print(f"  minutes with zero traffic : {zero_minutes:,} "
          f"({zero_minutes / len(series):.2%})")
    print("=" * 74)

    # --- Definition-of-Done gates ----------------------------------------
    ok = True

    # Every skipped line is a request we failed to count. A handful is expected (the
    # July trace genuinely ends mid-record); a flood means the parser is wrong.
    if total_skipped > 10:
        print(f"\nFAIL: skipped {total_skipped:,} lines - parser is dropping records.")
        ok = False

    if set(total_offsets) != {"-0400"}:
        print(f"\nFAIL: mixed timezone offsets {dict(total_offsets)} - the naive "
              f"index above would be inconsistent.")
        ok = False

    # 99.9% rather than 100%: the documented count includes the truncated final record.
    if counted_total < documented_total * 0.999:
        print(f"\nFAIL: counted {counted_total:,} of {documented_total:,} documented "
              f"lines - investigate before Phase 2.")
        ok = False

    if not ok:
        return 1

    print("\nAll sanity gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
