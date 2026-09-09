"""
Polls `kubectl get pods` every 5 seconds and logs, per timestamp, how many
pods exist for each deployment we're watching. Run this in parallel with the
k6 load test (start it first, stop it with Ctrl+C after the test finishes).

USAGE:
    python3 poll_pod_counts.py

OUTPUT:
    pod_counts.csv  -- timestamp, deployment, replica_count
"""

import subprocess
import time
import csv
from datetime import datetime, timezone

WATCH_DEPLOYMENTS = [
    "frontend",
    "recommendationservice",
    "currencyservice",
    "cartservice",
    "checkoutservice",
    "productcatalogservice",
]

OUTFILE = "pod_counts2.csv"
POLL_INTERVAL_SEC = 5


def get_replica_counts():
    """Returns {deployment_name: ready_replica_count} via kubectl."""
    counts = {}
    for dep in WATCH_DEPLOYMENTS:
        try:
            out = subprocess.run(
                ["kubectl", "get", "deployment", dep,
                 "-o", "jsonpath={.status.readyReplicas}"],
                capture_output=True, text=True, timeout=5,
            )
            val = out.stdout.strip()
            counts[dep] = int(val) if val else 0
        except Exception:
            counts[dep] = None
    return counts


def main():
    print(f"Polling every {POLL_INTERVAL_SEC}s -> {OUTFILE}  (Ctrl+C to stop)")
    with open(OUTFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "deployment", "ready_replicas"])
        f.flush()

        try:
            while True:
                ts = datetime.now(timezone.utc).isoformat()
                counts = get_replica_counts()
                for dep, n in counts.items():
                    writer.writerow([ts, dep, n])
                f.flush()
                print(f"[{ts}] " + "  ".join(f"{d}={n}" for d, n in counts.items()))
                time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\nStopped. Saved to", OUTFILE)


if __name__ == "__main__":
    main()
