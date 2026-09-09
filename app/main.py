"""The target application - a small HTTP service that is scalable and observable.

This is the workload every later phase scales. It has exactly three jobs:

  1. Serve HTTP and count the requests, so Prometheus can compute a request rate.
     That rate is the Monitor stage of the MAPE loop and the signal Phase 5 forecasts.
  2. Burn a measurable, tunable amount of CPU per request, so that offered load
     actually shows up as CPU utilisation. Without this, native HPA - which scales on
     CPU - would have nothing to react to, and the comparison arm would be meaningless.
  3. Start fast and report readiness honestly, because Phase 3 measures how long a pod
     takes to become Ready and that number underpins our contribution (DEVIATIONS D-004).

Deliberately dependency-free: Python standard library only. Two reasons. A `pip install`
layer would add tens of megabytes to the image and seconds to container start, both of
which would inflate the cold-start measurement this phase exists to take. And the
Prometheus text exposition format is simple enough that hand-writing it is clearer than
pulling in a client library whose behaviour we would then have to explain.

Endpoints:
    GET /            do CPU work, return 200
    GET /work?n=..   same, with an explicit iteration count
    GET /metrics     Prometheus exposition format
    GET /healthz     liveness  - is the process alive?
    GET /readyz      readiness - should this pod receive traffic?
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PORT = int(os.environ.get("PORT", "8080"))

# CPU work per request, in SHA-256 rounds over a small buffer. This is the knob that
# sets per-pod capacity: more iterations means fewer requests per second per pod, which
# means the replica count Phase 5 computes responds visibly to load. Calibrated in
# Phase 4 against the CPU limit in k8s/app-deployment.yaml - see results/.
WORK_ITERATIONS = int(os.environ.get("WORK_ITERATIONS", "12000"))

# Simulated application boot time. Real services load models, warm caches and open
# connection pools before they can serve. Ours cannot honestly claim zero, and setting
# it explicitly means the cold-start measurement has a known, documented app component
# rather than an accidental one. Default 0 keeps the baseline measurement clean.
STARTUP_DELAY_S = float(os.environ.get("STARTUP_DELAY_S", "0"))

POD_NAME = os.environ.get("POD_NAME", socket.gethostname())

_process_start = time.time()
_ready = threading.Event()

# All counters are guarded by one lock. Contention is irrelevant here - the CPU work
# dominates by orders of magnitude - and one obvious lock is easier to reason about
# than several clever ones.
_lock = threading.Lock()
_requests_total: dict[tuple[str, str, int], int] = {}
_work_seconds_total = 0.0
_inflight = 0

# Histogram buckets in seconds, chosen around the latency this app actually produces
# under load. Buckets far from the real distribution make a histogram useless: every
# observation lands in the same one.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_hist_counts = [0] * len(_BUCKETS)
_hist_sum = 0.0
_hist_count = 0


def _burn(iterations: int) -> str:
    """Consume CPU deterministically and return something, so it cannot be optimised out."""
    buf = b"aiops-phase3-workload"
    h = b""
    for _ in range(iterations):
        buf = hashlib.sha256(buf).digest()
        h = buf
    return h.hex()[:16]


def _observe(method: str, path: str, status: int, elapsed: float) -> None:
    global _hist_sum, _hist_count
    with _lock:
        key = (method, path, status)
        _requests_total[key] = _requests_total.get(key, 0) + 1
        _hist_sum += elapsed
        _hist_count += 1
        for i, edge in enumerate(_BUCKETS):
            if elapsed <= edge:
                _hist_counts[i] += 1


def _render_metrics() -> bytes:
    with _lock:
        reqs = dict(_requests_total)
        hist = list(_hist_counts)
        hsum, hcount = _hist_sum, _hist_count
        work_s = _work_seconds_total
        inflight = _inflight

    out: list[str] = []
    out.append("# HELP http_requests_total Total HTTP requests handled.")
    out.append("# TYPE http_requests_total counter")
    if not reqs:
        # Emit the series at zero even before the first request. A counter that only
        # appears after traffic starts makes `rate()` return nothing for the first
        # scrape interval, which looks exactly like a broken ServiceMonitor.
        out.append(f'http_requests_total{{method="GET",path="/",status="200",'
                   f'pod="{POD_NAME}"}} 0')
    for (method, path, status), n in sorted(reqs.items()):
        out.append(f'http_requests_total{{method="{method}",path="{path}",'
                   f'status="{status}",pod="{POD_NAME}"}} {n}')

    out.append("# HELP http_request_duration_seconds Request latency.")
    out.append("# TYPE http_request_duration_seconds histogram")
    cumulative = 0
    for edge, count in zip(_BUCKETS, hist):
        cumulative += count
        out.append(f'http_request_duration_seconds_bucket{{pod="{POD_NAME}",'
                   f'le="{edge}"}} {cumulative}')
    out.append(f'http_request_duration_seconds_bucket{{pod="{POD_NAME}",le="+Inf"}} '
               f'{hcount}')
    out.append(f'http_request_duration_seconds_sum{{pod="{POD_NAME}"}} {hsum:.6f}')
    out.append(f'http_request_duration_seconds_count{{pod="{POD_NAME}"}} {hcount}')

    out.append("# HELP app_work_seconds_total CPU seconds spent in the work function.")
    out.append("# TYPE app_work_seconds_total counter")
    out.append(f'app_work_seconds_total{{pod="{POD_NAME}"}} {work_s:.6f}')

    out.append("# HELP app_inflight_requests Requests currently being served.")
    out.append("# TYPE app_inflight_requests gauge")
    out.append(f'app_inflight_requests{{pod="{POD_NAME}"}} {inflight}')

    out.append("# HELP app_uptime_seconds Seconds since process start.")
    out.append("# TYPE app_uptime_seconds gauge")
    out.append(f'app_uptime_seconds{{pod="{POD_NAME}"}} '
               f'{time.time() - _process_start:.3f}')

    return ("\n".join(out) + "\n").encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "aiops-target/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Per-request logging to stderr would itself become a bottleneck at the request
        # rates Phase 4 drives, and would distort the very latency we are measuring.
        pass

    def _send(self, status: int, body: bytes, ctype: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        global _work_seconds_total, _inflight
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/metrics":
            self._send(200, _render_metrics(),
                       "text/plain; version=0.0.4; charset=utf-8")
            return

        if route == "/healthz":
            # Liveness: the process is running and can accept a connection. It must NOT
            # depend on readiness, or a slow start would get the container killed and
            # restarted forever instead of simply waiting.
            self._send(200, b"ok\n")
            return

        if route == "/readyz":
            if _ready.is_set():
                self._send(200, b"ready\n")
            else:
                self._send(503, b"starting\n")
            return

        # --- the actual workload -----------------------------------------
        start = time.perf_counter()
        with _lock:
            _inflight += 1
        try:
            qs = parse_qs(parsed.query)
            n = int(qs.get("n", [WORK_ITERATIONS])[0])
            digest = _burn(n)
            status = 200
            body = json.dumps({
                "pod": POD_NAME, "iterations": n, "digest": digest,
            }).encode() + b"\n"
        except Exception as e:  # noqa: BLE001
            status = 500
            body = f"error: {e}\n".encode()
        finally:
            elapsed = time.perf_counter() - start
            with _lock:
                _inflight -= 1
                _work_seconds_total += elapsed

        # Collapse the path label so a load generator hitting /work?n=1 and /work?n=2
        # does not create unbounded label cardinality in Prometheus.
        _observe("GET", "/work" if route.startswith("/work") else "/", status, elapsed)
        self._send(status, body, "application/json")


def main() -> None:
    if STARTUP_DELAY_S > 0:
        time.sleep(STARTUP_DELAY_S)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True

    # Bind first, then mark ready. Ordering matters: if readiness flipped before the
    # socket was listening, Kubernetes would route traffic to a pod that refuses
    # connections, and the cold-start number would be optimistic by however long the
    # gap was.
    _ready.set()
    print(f"listening on :{PORT}  pod={POD_NAME}  "
          f"work_iterations={WORK_ITERATIONS}  boot={time.time() - _process_start:.3f}s",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
