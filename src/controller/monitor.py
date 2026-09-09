"""Phase 5 - Monitor: read the recent request rate out of Prometheus.

The forecaster needs the last 60 TRACE-minutes as counts-per-trace-minute. Prometheus
stores a counter scraped in WALL time, so this module is where the two clocks meet.

    one trace-minute  =  60/C wall-seconds        (6 s at C=10)
    counts in trace-minute i = increase(http_requests_total[6s]) at the end of bucket i

We ask for a range query at `step = trace_minute_wall_s` so Prometheus does the bucketing
itself, then convert rate (per second) to a count per trace-minute.

--------------------------------------------------------------------------------
Why the scrape interval had to change
--------------------------------------------------------------------------------
`rate()` and `increase()` need at least two samples inside the lookback window. At the
original 15 s scrape interval a 6 s window contains zero or one sample, so every query
would return empty and the controller would see a permanent outage. The scrape interval
is part of the compressed clock and is now 1.5 s (D-014), giving four samples per bucket.

--------------------------------------------------------------------------------
Gaps must not crash the loop, and must not be silently invented either
--------------------------------------------------------------------------------
A missed scrape, a pod restart or a Prometheus reload leaves holes. The loop has to keep
running, but a hole is not a zero: a minute with no data and a minute with no traffic
mean opposite things, and filling the first with 0 teaches the model that demand
collapsed. So gaps are:

  * filled by carrying the last observed value forward (persistence, the right
    assumption for a metric that is continuous in time), and
  * COUNTED, and returned alongside the window, so the controller can log how much of
    its input was real and refuse to act when too little of it was.

A genuine zero-traffic minute is distinguishable because the counter IS present and
simply does not increase - that yields a real 0, not a gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import requests


@dataclass
class Window:
    values: np.ndarray          # counts per trace-minute, oldest first
    timestamps: np.ndarray      # datetime64[m], trace-clock labels
    gaps: int                   # buckets with no data, filled by carry-forward
    stale: bool                 # the most recent bucket was itself a gap
    query_ms: float

    @property
    def coverage(self) -> float:
        return 1.0 - self.gaps / max(len(self.values), 1)


class Monitor:
    def __init__(self, cfg, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.http = session or requests.Session()
        self.metric = "http_requests_total"
        self.selector = f'namespace="{cfg.namespace}"'

    def _range_query(self, query: str, start: float, end: float,
                     step: float) -> list[tuple[float, float]]:
        r = self.http.get(
            f"{self.cfg.prometheus_url}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=5.0,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"prometheus: {payload.get('error')}")
        result = payload["data"]["result"]
        if not result:
            return []
        # sum() over pods collapses to a single series; if more than one comes back the
        # query is wrong and averaging them would hide it.
        if len(result) > 1:
            raise RuntimeError(f"expected 1 series, got {len(result)} - check the query")
        return [(float(t), float(v)) for t, v in result[0]["values"]]

    def fetch(self, trace_now: np.datetime64, now_wall: float | None = None) -> Window:
        """Window of `look_back` trace-minutes ending at (and including) trace_now."""
        cfg = self.cfg
        step = cfg.trace_minute_wall_s
        n = cfg.look_back_trace_min
        end = now_wall if now_wall is not None else time.time()
        start = end - (n - 1) * step

        # rate() over exactly one bucket, summed across pods. Using rate()*window rather
        # than increase() because increase() extrapolates at series boundaries and can
        # return fractional counts that look like corrupted data in the logs.
        q = (f'sum(rate({self.metric}{{{self.selector}}}[{step:g}s]))')

        t0 = time.perf_counter()
        points = self._range_query(q, start, end, step)
        query_ms = (time.perf_counter() - t0) * 1000.0

        by_bucket = {round((t - start) / step): v for t, v in points}
        values = np.zeros(n, dtype="float64")
        gaps = 0
        last = None
        for i in range(n):
            if i in by_bucket:
                last = by_bucket[i] * step        # req/s -> requests in this bucket
                values[i] = last
            else:
                gaps += 1
                # Carry forward, not zero. See the module docstring.
                values[i] = last if last is not None else 0.0

        ts = np.array([trace_now - np.timedelta64(n - 1 - i, "m") for i in range(n)],
                      dtype="datetime64[m]")
        return Window(values=values, timestamps=ts, gaps=gaps,
                      stale=(n - 1) not in by_bucket, query_ms=query_ms)

    def healthy(self) -> tuple[bool, str]:
        try:
            r = self.http.get(f"{self.cfg.prometheus_url}/-/ready", timeout=3.0)
            return (r.status_code == 200, f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
