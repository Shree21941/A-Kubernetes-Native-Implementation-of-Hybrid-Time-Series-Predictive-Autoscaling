"""Phase 3 - measure how long a pod actually takes to become Ready.

    ./.venv/bin/python src/eval/coldstart.py

This is the empirical foundation of our contribution (DEVIATIONS.md D-004). The baseline
forecasts 60 seconds ahead and never checks whether a pod can be Ready inside that
minute. If pods need 45 s, only ~15 s of genuine head start remains - and Phase 2 showed
the hybrid forecaster spends ~3 s of that just predicting. If pods need 90 s, a 60 s
horizon cannot help at all: the forecast is correct and useless.

What is measured. The deployment is scaled 1 -> 5, and we record wall-clock time until
the Deployment reports all replicas Ready, repeated several times. Alongside the single
headline number we record the breakdown Kubernetes itself reports per pod:

    created -> scheduled     the scheduler placing the pod on a node
    scheduled -> started     kubelet creating and starting the container
    started -> ready         the readiness probe passing

The breakdown matters because only some of it is reducible. Scheduling and container
start are Kubernetes' floor; probe delay is ours to configure. Reporting one aggregate
number would hide which is which, and the whole point of the later contribution is to
argue about that budget.

Scope, stated honestly. The image is pre-loaded onto every node by `kind load
docker-image`, so this measures the STEADY-STATE scale-out case - the one an autoscaler
actually lives in, since after the first pod the image is cached. A genuinely cold node
that must pull the image would be substantially slower, and that is a different number
we are not claiming to have measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from kubernetes import client, config

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

NAMESPACE = "aiops"
DEPLOYMENT = "target-app"
POLL_S = 0.1  # 100 ms: fine enough that the instrument is not the bottleneck


def _now() -> float:
    return time.perf_counter()


def _iso(ts) -> float | None:
    """Kubernetes timestamps -> epoch seconds. They have 1-second granularity."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=ts.tzinfo or timezone.utc).timestamp()
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def scale(apps: client.AppsV1Api, replicas: int) -> None:
    apps.patch_namespaced_deployment_scale(
        DEPLOYMENT, NAMESPACE, {"spec": {"replicas": replicas}}
    )


def ready_count(apps: client.AppsV1Api) -> int:
    st = apps.read_namespaced_deployment_status(DEPLOYMENT, NAMESPACE).status
    return st.ready_replicas or 0


def pod_names(core: client.CoreV1Api) -> set[str]:
    pods = core.list_namespaced_pod(NAMESPACE, label_selector=f"app={DEPLOYMENT}")
    return {p.metadata.name for p in pods.items}


def settle(apps: client.AppsV1Api, core: client.CoreV1Api, target: int,
           timeout: float = 180.0) -> None:
    """Wait until exactly `target` pods exist AND are Ready.

    Both conditions are required. Checking only readyReplicas would let a trial start
    while pods from the previous trial were still terminating, and those pods compete
    for the same CPU - which would inflate the next measurement for a reason that has
    nothing to do with cold start.
    """
    deadline = _now() + timeout
    while _now() < deadline:
        if ready_count(apps) == target and len(pod_names(core)) == target:
            time.sleep(1.0)  # brief quiet period so the cluster is genuinely idle
            if ready_count(apps) == target and len(pod_names(core)) == target:
                return
        time.sleep(0.25)
    raise TimeoutError(f"deployment did not settle at {target} replicas in {timeout}s")


def one_trial(apps: client.AppsV1Api, core: client.CoreV1Api,
              frm: int, to: int, timeout: float = 180.0) -> dict:
    settle(apps, core, frm)
    before = pod_names(core)

    t0 = _now()
    scale(apps, to)

    first_ready_at: float | None = None
    all_ready_at: float | None = None
    seen_ready: set[str] = set()

    deadline = t0 + timeout
    while _now() < deadline:
        pods = core.list_namespaced_pod(
            NAMESPACE, label_selector=f"app={DEPLOYMENT}"
        ).items
        for p in pods:
            if p.metadata.name in before or p.metadata.name in seen_ready:
                continue
            conds = {c.type: c for c in (p.status.conditions or [])}
            if conds.get("Ready") and conds["Ready"].status == "True":
                seen_ready.add(p.metadata.name)
                if first_ready_at is None:
                    first_ready_at = _now() - t0
        if ready_count(apps) >= to:
            all_ready_at = _now() - t0
            break
        time.sleep(POLL_S)

    if all_ready_at is None:
        raise TimeoutError(f"only {ready_count(apps)}/{to} replicas Ready after {timeout}s")

    # Per-pod breakdown from Kubernetes' own timestamps (1 s granularity).
    breakdown = []
    for p in core.list_namespaced_pod(
        NAMESPACE, label_selector=f"app={DEPLOYMENT}"
    ).items:
        if p.metadata.name in before:
            continue
        conds = {c.type: c for c in (p.status.conditions or [])}
        created = _iso(p.metadata.creation_timestamp)
        sched = _iso(conds["PodScheduled"].last_transition_time) if "PodScheduled" in conds else None
        rdy = _iso(conds["Ready"].last_transition_time) if "Ready" in conds else None
        started = None
        for cs in (p.status.container_statuses or []):
            if cs.state and cs.state.running:
                started = _iso(cs.state.running.started_at)
        breakdown.append(dict(
            pod=p.metadata.name,
            node=p.spec.node_name,
            schedule_s=(sched - created) if (sched and created) else None,
            start_s=(started - sched) if (started and sched) else None,
            probe_s=(rdy - started) if (rdy and started) else None,
            total_s=(rdy - created) if (rdy and created) else None,
        ))

    scale(apps, frm)
    return dict(
        first_ready_s=round(first_ready_at, 3) if first_ready_at else None,
        all_ready_s=round(all_ready_at, 3),
        n_new_pods=len(breakdown),
        pods=breakdown,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="frm", type=int, default=1)
    ap.add_argument("--to", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    config.load_kube_config()
    apps, core = client.AppsV1Api(), client.CoreV1Api()

    print(f"cold-start: scaling {DEPLOYMENT} {args.frm} -> {args.to}, "
          f"{args.repeats} trials\n")

    trials = []
    for i in range(1, args.repeats + 1):
        t = one_trial(apps, core, args.frm, args.to)
        t["trial"] = i
        trials.append(t)
        print(f"  trial {i}: first pod Ready {t['first_ready_s']:6.2f}s   "
              f"all {args.to} Ready {t['all_ready_s']:6.2f}s", flush=True)

    settle(apps, core, args.frm)

    allr = [t["all_ready_s"] for t in trials]
    firstr = [t["first_ready_s"] for t in trials if t["first_ready_s"] is not None]

    def stats(xs):
        return dict(
            mean=round(statistics.mean(xs), 3),
            median=round(statistics.median(xs), 3),
            stdev=round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
            min=round(min(xs), 3), max=round(max(xs), 3),
        )

    # Aggregate the per-pod breakdown across every pod of every trial.
    parts = {k: [p[k] for t in trials for p in t["pods"] if p.get(k) is not None]
             for k in ("schedule_s", "start_s", "probe_s", "total_s")}

    summary = dict(
        scale_from=args.frm, scale_to=args.to, repeats=args.repeats,
        all_ready=stats(allr), first_ready=stats(firstr) if firstr else None,
        per_pod={k: stats(v) for k, v in parts.items() if v},
        note="image pre-loaded on all nodes; steady-state scale-out, not a cold pull",
        trials=trials,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase3_coldstart.json").write_text(json.dumps(summary, indent=2))

    print()
    print("Pod cold-start latency")
    print("=" * 72)
    print(f"{'metric':<34} {'mean':>8} {'median':>8} {'sd':>7} {'min':>7} {'max':>7}")
    print("-" * 72)
    s = summary["all_ready"]
    print(f"{'time until all ' + str(args.to) + ' Ready (s)':<34} "
          f"{s['mean']:>8.2f} {s['median']:>8.2f} {s['stdev']:>7.2f} "
          f"{s['min']:>7.2f} {s['max']:>7.2f}")
    if summary["first_ready"]:
        s = summary["first_ready"]
        print(f"{'time until FIRST pod Ready (s)':<34} "
              f"{s['mean']:>8.2f} {s['median']:>8.2f} {s['stdev']:>7.2f} "
              f"{s['min']:>7.2f} {s['max']:>7.2f}")
    print("-" * 72)
    labels = {"schedule_s": "  per pod: created -> scheduled",
              "start_s": "  per pod: scheduled -> started",
              "probe_s": "  per pod: started -> Ready (probe)",
              "total_s": "  per pod: created -> Ready"}
    for k, lab in labels.items():
        if k in summary["per_pod"]:
            s = summary["per_pod"][k]
            print(f"{lab:<34} {s['mean']:>8.2f} {s['median']:>8.2f} "
                  f"{s['stdev']:>7.2f} {s['min']:>7.2f} {s['max']:>7.2f}")
    print("=" * 72)

    # The comparison that actually matters for the project.
    m = summary["all_ready"]["mean"]
    print(f"\nAgainst the paper's 60 s forecast horizon:")
    print(f"  time to scale {args.frm} -> {args.to}          : {m:.1f} s")
    print(f"  remaining head start           : {60 - m:.1f} s "
          f"({(60 - m) / 60:.0%} of the horizon)")
    print(f"  minus hybrid prediction (~3 s) : {60 - m - 3:.1f} s")
    if m >= 60:
        print("  => the horizon is TOO SHORT: pods cannot be Ready in time.")
    elif m > 30:
        print("  => over half the horizon is consumed by pod startup.")
    print(f"\nwrote results/phase3_coldstart.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
