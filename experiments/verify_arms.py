"""Phase 6 - assert the arms differ ONLY in how they decide, not in what they are allowed.

    ./.venv/bin/python experiments/verify_arms.py            # static: parse the manifests
    ./.venv/bin/python experiments/verify_arms.py --cluster  # also check the live cluster

Two kinds of check, and the static ones are the important ones because they can run
before a cluster exists and they catch the failures that are invisible afterwards.

WHY THIS IS A SCRIPT AND NOT A CAREFUL READ-THROUGH. Parity between arms is the entire
validity of Phase 7. An arm with maxReplicas 12 instead of 10 wins on SLO and loses on
cost, and the write-up would attribute that to its algorithm. Nothing errors, nothing
looks wrong, and the difference is one character in a YAML file. Checks like this are
worth more than the code they test.

The specific ways parity can break, all checked below:

  * different replica bounds                -> the extra capacity does the winning
  * different per-pod capacity assumptions  -> the arms disagree about the QUESTION,
                                               not the answer
  * different stabilisation windows         -> a difference in willingness to shrink
                                               reads as a difference in prediction
  * KEDA metricType left at its default     -> HPA multiplies by replica count and the
                                               arm runs away to maxReplicas (CLAUDE.md 3)
  * an uncompressed timing constant         -> one arm lives on a different clock (D-010)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src" / "controller")]

import yaml  # noqa: E402

from config import Config  # noqa: E402

RESULTS = REPO / "results"
NS = "aiops"
DEPLOY = "target-app"
COMPRESSION = 10.0

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def load_docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster", action="store_true",
                    help="additionally verify the live cluster (needs kubectl)")
    args = ap.parse_args()

    cfg = Config()
    k8s = REPO / "k8s"

    hpa = next(d for d in load_docs(k8s / "arm-hpa-cpu.yaml")
               if d["kind"] == "HorizontalPodAutoscaler")
    keda = next(d for d in load_docs(k8s / "arm-keda-rate.yaml")
                if d["kind"] == "ScaledObject")
    ctrl = next(d for d in load_docs(k8s / "controller.yaml")
                if d["kind"] == "Deployment")
    ctrl_env = {e["name"]: e.get("value")
                for e in ctrl["spec"]["template"]["spec"]["containers"][0]["env"]}

    # -- 1. replica bounds identical -------------------------------------------------
    bounds = {
        "hpa-cpu": (hpa["spec"]["minReplicas"], hpa["spec"]["maxReplicas"]),
        "keda-rate": (keda["spec"]["minReplicaCount"], keda["spec"]["maxReplicaCount"]),
        "predictive": (int(ctrl_env["MIN_REPLICAS"]), int(ctrl_env["MAX_REPLICAS"])),
    }
    uniq = set(bounds.values())
    check("replica bounds identical across all three arms", len(uniq) == 1,
          "; ".join(f"{k}={v}" for k, v in bounds.items()))

    # The predictive controller's own defaults must agree with what the manifest sets,
    # or a locally-run controller and an in-cluster one are different arms.
    check("controller manifest bounds match config.py defaults",
          bounds["predictive"] == (cfg.min_replicas, cfg.max_replicas),
          f"manifest={bounds['predictive']} config.py=({cfg.min_replicas}, {cfg.max_replicas})")

    # -- 2. same capacity assumption --------------------------------------------------
    keda_threshold = float(keda["spec"]["triggers"][0]["metadata"]["threshold"])
    ctrl_capacity = float(ctrl_env["PER_POD_CAPACITY_RPS"])
    check("KEDA threshold == controller PER_POD_CAPACITY_RPS",
          abs(keda_threshold - ctrl_capacity) < 1e-9,
          f"keda={keda_threshold} controller={ctrl_capacity}")
    check("controller manifest capacity matches config.py default",
          abs(ctrl_capacity - cfg.per_pod_capacity_rps) < 1e-9,
          f"manifest={ctrl_capacity} config.py={cfg.per_pod_capacity_rps}")

    # The capacity must be the MEASURED one, not a round number someone typed.
    # Prefer the POD-measured file over the local-process one. calibrate.py --url writes
    # phase4_capacity.json; --spawn writes phase4_capacity_local.json. Once the cluster
    # measurement exists it supersedes the local emulation entirely, and checking against
    # the stale file made this assertion fail for the right value (3.135 measured on a
    # pod vs 3.245 emulated on the host) - a check that fails on correct configuration
    # is worse than no check, because the next person silences it.
    cap_file = next((p for p in (RESULTS / "phase4_capacity.json",
                                 RESULTS / "phase4_capacity_local.json")
                     if p.exists()), None)
    if cap_file:
        cap_data = json.loads(cap_file.read_text())
        measured = cap_data["per_pod_capacity_rps"]
        check("capacity traces back to the Phase 4 measurement",
              abs(ctrl_capacity - measured) < 5e-4,
              f"config={ctrl_capacity} measured={measured} ({cap_file.name})")
        # And it must be the POD measurement, not the host emulation, before any result
        # is reported. D-012 flagged the local value as provisional for exactly this.
        check("capacity was measured against a real pod, not a local process",
              cap_file.name == "phase4_capacity.json",
              f"using {cap_file.name}"
              + ("" if cap_file.name == "phase4_capacity.json"
                 else " - re-run load/calibrate.py --url against the cluster"))

    # -- 3. KEDA metric semantics -----------------------------------------------------
    mt = keda["spec"]["triggers"][0].get("metricType")
    check("KEDA metricType explicitly AverageValue (not defaulted)",
          mt == "AverageValue",
          f"metricType={mt!r} - with 'Value', HPA multiplies by replica count "
          f"(CLAUDE.md 3)")

    # -- 4. both reactive arms equally willing to shrink -------------------------------
    hb = hpa["spec"]["behavior"]
    kb = keda["spec"]["advanced"]["horizontalPodAutoscalerConfig"]["behavior"]
    check("HPA and KEDA share identical scaleUp behaviour",
          hb["scaleUp"] == kb["scaleUp"], f"hpa={hb['scaleUp']} keda={kb['scaleUp']}")
    check("HPA and KEDA share identical scaleDown behaviour",
          hb["scaleDown"] == kb["scaleDown"],
          f"hpa={hb['scaleDown']} keda={kb['scaleDown']}")

    # -- 5. the reactive arms' scale-down matches the predictive planner ---------------
    check("reactive scaleDown window == predictive scale-down cooldown",
          abs(hb["scaleDown"]["stabilizationWindowSeconds"]
              - cfg.scale_down_cooldown_wall_s) < 1e-6,
          f"reactive={hb['scaleDown']['stabilizationWindowSeconds']}s "
          f"predictive={cfg.scale_down_cooldown_wall_s}s")

    # -- 6. every timing constant lives on the compressed clock ------------------------
    # Each entry: (label, configured wall seconds, the real-time value it derives from).
    timings = [
        ("KEDA pollingInterval", keda["spec"]["pollingInterval"], 30.0),
        ("KEDA cooldownPeriod", keda["spec"]["cooldownPeriod"], 300.0),
        ("HPA scaleDown stabilization", hb["scaleDown"]["stabilizationWindowSeconds"], 300.0),
    ]
    for label, actual, real_time in timings:
        want = real_time / COMPRESSION
        check(f"compressed clock: {label}", abs(actual - want) < 1e-6,
              f"{actual}s, expected {want:g}s (= {real_time:g}s / C={COMPRESSION:g})")

    kind_cfg = (k8s / "kind-config.yaml").read_text()
    m = re.search(r"horizontal-pod-autoscaler-sync-period:\s*\"?([0-9a-z]+)\"?", kind_cfg)
    check("HPA sync period compressed in kind-config.yaml",
          bool(m) and m.group(1) in ("1500ms", "1.5s"),
          f"found {m.group(1) if m else 'NOTHING'} - default 15s would give the "
          f"predictive arm a 2.5x timing advantage over both reactive arms (D-016)")

    check("controller COMPRESSION matches the arms' compression",
          abs(float(ctrl_env["COMPRESSION"]) - COMPRESSION) < 1e-9,
          f"controller={ctrl_env['COMPRESSION']} arms assume {COMPRESSION:g}")

    # -- 6b. the two predictive arms are one Deployment, not two -----------------------
    # The ablation's validity depends on the arms differing in exactly one thing. Two
    # Deployments would eventually drift in a second detail; one Deployment with a
    # patched env var cannot. It also makes it structurally impossible for both
    # predictive arms to be active simultaneously.
    check("FORECASTER is a controller env var (one Deployment for both predictive arms)",
          "FORECASTER" in ctrl_env,
          f"value={ctrl_env.get('FORECASTER')!r} - switch_arm.sh patches this")
    check("controller manifest FORECASTER matches config.py default",
          ctrl_env.get("FORECASTER") == cfg.forecaster,
          f"manifest={ctrl_env.get('FORECASTER')!r} config.py={cfg.forecaster!r}")

    switch = (REPO / "experiments" / "switch_arm.sh").read_text()
    for arm in ("hpa-cpu", "keda-rate", "predictive-prophet", "predictive-hybrid"):
        check(f"switch_arm.sh knows arm '{arm}'", arm in switch,
              "an arm missing from ALL_ARMS is an arm teardown can silently miss")

    # -- 7. all arms target the same Deployment ---------------------------------------
    targets = {
        "hpa-cpu": hpa["spec"]["scaleTargetRef"]["name"],
        "keda-rate": keda["spec"]["scaleTargetRef"]["name"],
        "predictive": ctrl_env["DEPLOYMENT"],
    }
    check("all arms target the same Deployment", len(set(targets.values())) == 1,
          "; ".join(f"{k}={v}" for k, v in targets.items()))

    # -- 8. live cluster ---------------------------------------------------------------
    if args.cluster:
        try:
            out = subprocess.run(
                ["kubectl", "get", "hpa", "-n", NS, "-o",
                 f"jsonpath={{range .items[?(@.spec.scaleTargetRef.name=='{DEPLOY}')]}}"
                 "{.metadata.name}{'\\n'}{end}"],
                capture_output=True, text=True, timeout=15)
            hpas = [x for x in out.stdout.split("\n") if x.strip()]
            so = subprocess.run(
                ["kubectl", "get", "scaledobject", "-n", NS, "-o",
                 "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}"],
                capture_output=True, text=True, timeout=15)
            sos = [x for x in so.stdout.split("\n") if x.strip()]
            ctl = subprocess.run(
                ["kubectl", "get", "deploy", "predictive-autoscaler", "-n", NS,
                 "-o", "jsonpath={.spec.replicas}"],
                capture_output=True, text=True, timeout=15)
            ctl_n = int(ctl.stdout) if ctl.returncode == 0 and ctl.stdout.strip() else 0

            # A KEDA ScaledObject creates an HPA, so it would be double-counted.
            n_active = len(sos) + ctl_n + max(0, len(hpas) - len(sos))
            check("exactly one autoscaler active on the Deployment", n_active <= 1,
                  f"hpas={hpas} scaledobjects={sos} predictive_replicas={ctl_n}")
        except Exception as e:  # noqa: BLE001
            check("live cluster reachable", False, f"{type(e).__name__}: {e}")
    else:
        print("  (static checks only - pass --cluster to also verify the live cluster)\n")

    # -- report ------------------------------------------------------------------------
    print("Phase 6 - arm parity")
    print("=" * 92)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL':<5} {name:<{width}}  {detail}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("=" * 92)
    print(f"  {len(results) - n_fail}/{len(results)} passed")
    if n_fail:
        print("\n  Parity is the validity of Phase 7. Do not collect data until this is clean.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
