"""Phase 7 step 8 - propagate the REAL measurements into everything that consumes them.

    ./.venv/bin/python experiments/wire_measurements.py --check   # preflight, changes nothing
    ./.venv/bin/python experiments/wire_measurements.py           # patch the manifests

Until this runs, the experiment is configured with placeholders:

    PER_POD_CAPACITY_RPS = 3.245   measured against a CPUQuota=50% LOCAL PROCESS (D-012)
    POD_COLD_START_WALL_S = 5.0    a guess; POD_COLD_START_MEASURED=0 flags it as such
    SLO = 142.7 ms                 derived from the local process's unloaded p50

Those numbers came from a laptop process under a cgroup quota, not from a pod on a KIND
node. They are the right ORDER of magnitude and there is no reason to believe they are
the right VALUE - a pod pays for kube-proxy, the CNI, the container runtime and
contention with two other nodes' workloads, none of which a bare process pays.

--------------------------------------------------------------------------------
Why this is a script and not four hand edits
--------------------------------------------------------------------------------
The same two constants appear in four places that must agree:

    k8s/controller.yaml        the predictive arms' capacity and cold start
    k8s/arm-keda-rate.yaml     the KEDA arm's threshold - MUST equal the above, or the
                               arms differ in their capacity assumption rather than in
                               their timing, and D-017's parity guarantee is void
    src/controller/config.py   the defaults a locally-run controller would use
    the SLO used to score every run

Hand-editing four files is how three of them end up right. And the failure is silent:
every arm still runs, every number still looks plausible, and the comparison is simply
measuring something other than what it claims.

--------------------------------------------------------------------------------
--check is the preflight gate
--------------------------------------------------------------------------------
Run before the matrix. It fails if the measurement files are missing, if any consumer
still carries a placeholder, or if the four places disagree. A 25-hour sweep built on a
placeholder is 25 hours of data that has to be caveated in every table it appears in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
COLDSTART = RESULTS / "phase3_coldstart.json"
CAPACITY = RESULTS / "phase4_capacity.json"          # the POD-measured one, not _local
OUT = REPO / "experiments" / "measured_constants.json"

CONTROLLER = REPO / "k8s" / "controller.yaml"
KEDA = REPO / "k8s" / "arm-keda-rate.yaml"

PLACEHOLDERS = {"per_pod_capacity_rps": 3.245, "pod_cold_start_wall_s": 5.0,
                "slo_ms": 142.7}


def load_measurements() -> dict:
    missing = [p.name for p in (COLDSTART, CAPACITY) if not p.exists()]
    if missing:
        raise SystemExit(
            f"MISSING MEASUREMENTS: {', '.join(missing)}\n"
            f"  {COLDSTART.name} comes from  ./.venv/bin/python src/eval/coldstart.py\n"
            f"  {CAPACITY.name} comes from   ./.venv/bin/python load/calibrate.py "
            f"--url http://localhost:30080/\n"
            f"Both must be taken against the RUNNING CLUSTER. Refusing to wire "
            f"placeholders."
        )

    cs = json.loads(COLDSTART.read_text())
    cap = json.loads(CAPACITY.read_text())

    # The capacity file must state it came from a pod. calibrate.py writes
    # phase4_capacity_local.json under --spawn and phase4_capacity.json under --url, so
    # the filename already encodes this - but check the recorded note too, because a file
    # can be copied and a filename lied about more easily than a field can.
    note = cap.get("note", "")
    if "local process" in note or cap.get("cpu_limit") == "500m" and "--spawn" in note:
        raise SystemExit(
            f"{CAPACITY.name} says it was measured against a local process, not a pod:\n"
            f"  note: {note}\n"
            f"Re-run calibrate.py with --url against the cluster."
        )

    return dict(
        per_pod_capacity_rps=float(cap["per_pod_capacity_rps"]),
        per_pod_capacity_rpm=float(cap["per_pod_capacity_rpm"]),
        slo_ms=float(cap["slo_p95_ms"]),
        unloaded_p50_ms=float(cap["unloaded_p50_ms"]),
        capacity_bracketed=bool(cap.get("bracketed", False)),
        capacity_measured_against=cap.get("measured_against"),
        # all_ready mean is the number the cooldown floor needs: the time until the
        # cluster has actually responded to a scale-up decision.
        pod_cold_start_wall_s=float(cs["all_ready"]["mean"]),
        pod_cold_start_median_s=float(cs["all_ready"]["median"]),
        pod_cold_start_sd_s=float(cs["all_ready"]["stdev"]),
        pod_first_ready_wall_s=(float(cs["first_ready"]["mean"])
                                if cs.get("first_ready") else None),
        cold_start_trials=int(cs["repeats"]),
        cold_start_scale=f"{cs['scale_from']}->{cs['scale_to']}",
    )


def patch(path: Path, subs: list[tuple[str, str, str]]) -> list[str]:
    """subs: (regex, replacement, human label). Every one must match exactly once."""
    text = path.read_text()
    changed = []
    for pattern, repl, label in subs:
        new, n = re.subn(pattern, repl, text)
        if n != 1:
            raise SystemExit(f"{path.name}: pattern for '{label}' matched {n} times, "
                             f"expected exactly 1 - refusing to guess.\n  {pattern}")
        if new != text:
            changed.append(label)
        text = new
    path.write_text(text)
    return changed


def do_wire(m: dict) -> None:
    cap = f"{m['per_pod_capacity_rps']:.4f}".rstrip("0").rstrip(".")
    cs = f"{m['pod_cold_start_wall_s']:.3f}".rstrip("0").rstrip(".")

    changed = patch(CONTROLLER, [
        (r'(- \{name: PER_POD_CAPACITY_RPS, value: ")[^"]*(")',
         rf'\g<1>{cap}\g<2>', "controller PER_POD_CAPACITY_RPS"),
        (r'(- \{name: POD_COLD_START_WALL_S, value: ")[^"]*(")',
         rf'\g<1>{cs}\g<2>', "controller POD_COLD_START_WALL_S"),
        (r'(- \{name: POD_COLD_START_MEASURED, value: ")[^"]*(")',
         r'\g<1>1\g<2>', "controller POD_COLD_START_MEASURED"),
    ])
    print(f"  k8s/controller.yaml       : {', '.join(changed) or 'already current'}")

    changed = patch(KEDA, [
        (r'(threshold: ")[^"]*(")', rf'\g<1>{cap}\g<2>', "KEDA threshold"),
    ])
    print(f"  k8s/arm-keda-rate.yaml    : {', '.join(changed) or 'already current'}")

    OUT.write_text(json.dumps(m, indent=2))
    print(f"  experiments/measured_constants.json written")


def do_check(m: dict) -> int:
    fails = []

    def ck(name: str, ok: bool, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL':<5} {name:<46} {detail}")
        if not ok:
            fails.append(name)

    print("\nMeasured values")
    print(f"    per-pod capacity : {m['per_pod_capacity_rps']:.3f} req/s "
          f"({m['per_pod_capacity_rpm']:.1f} req/min)  vs placeholder "
          f"{PLACEHOLDERS['per_pod_capacity_rps']}")
    print(f"    SLO              : p95 < {m['slo_ms']:.1f} ms "
          f"(5x unloaded p50 {m['unloaded_p50_ms']:.1f} ms)  vs placeholder "
          f"{PLACEHOLDERS['slo_ms']}")
    print(f"    pod cold start   : {m['pod_cold_start_wall_s']:.2f} s "
          f"(median {m['pod_cold_start_median_s']:.2f}, sd {m['pod_cold_start_sd_s']:.2f}, "
          f"n={m['cold_start_trials']}, {m['cold_start_scale']})  vs placeholder "
          f"{PLACEHOLDERS['pod_cold_start_wall_s']}")
    print()

    ctext = CONTROLLER.read_text()
    ktext = KEDA.read_text()

    def env_of(name: str) -> str | None:
        mm = re.search(rf'- \{{name: {name}, value: "([^"]*)"', ctext)
        return mm.group(1) if mm else None

    c_cap = env_of("PER_POD_CAPACITY_RPS")
    c_cs = env_of("POD_COLD_START_WALL_S")
    c_meas = env_of("POD_COLD_START_MEASURED")
    k_thr = re.search(r'threshold: "([^"]*)"', ktext)
    k_thr = k_thr.group(1) if k_thr else None

    ck("controller capacity == measured",
       c_cap is not None and abs(float(c_cap) - m["per_pod_capacity_rps"]) < 5e-4,
       f"manifest={c_cap} measured={m['per_pod_capacity_rps']:.4f}")
    ck("controller cold start == measured",
       c_cs is not None and abs(float(c_cs) - m["pod_cold_start_wall_s"]) < 5e-3,
       f"manifest={c_cs} measured={m['pod_cold_start_wall_s']:.3f}")
    ck("POD_COLD_START_MEASURED == 1", c_meas == "1",
       f"manifest={c_meas} (0 means the controller warns it is running on a guess)")
    ck("KEDA threshold == controller capacity",
       k_thr is not None and c_cap is not None and abs(float(k_thr) - float(c_cap)) < 5e-4,
       f"keda={k_thr} controller={c_cap}")

    # Explicitly assert the placeholders are GONE. A value that merely differs from the
    # placeholder could still be a second guess; a value equal to it is definitely stale.
    ck("capacity is not the local-process placeholder",
       c_cap is not None and abs(float(c_cap) - PLACEHOLDERS["per_pod_capacity_rps"]) > 1e-6,
       f"placeholder was {PLACEHOLDERS['per_pod_capacity_rps']}")
    ck("cold start is not the 5.0s placeholder",
       c_cs is not None and abs(float(c_cs) - PLACEHOLDERS["pod_cold_start_wall_s"]) > 1e-6,
       f"placeholder was {PLACEHOLDERS['pod_cold_start_wall_s']}")

    ck("measured_constants.json exists for run_matrix", OUT.exists(),
       str(OUT.relative_to(REPO)))
    if OUT.exists():
        saved = json.loads(OUT.read_text())
        ck("measured_constants.json matches the measurement files",
           abs(saved.get("per_pod_capacity_rps", -1) - m["per_pod_capacity_rps"]) < 5e-4
           and abs(saved.get("slo_ms", -1) - m["slo_ms"]) < 5e-3,
           "re-run without --check if stale")

    ck("capacity ramp bracketed the SLO", m["capacity_bracketed"],
       "if false the capacity is a LOWER BOUND - extend RATES in calibrate.py")

    # The finding that decides whether C=10 is honest (D-004, D-010).
    horizon_wall = 60.0 / 10.0
    head = horizon_wall - m["pod_cold_start_wall_s"]
    print()
    print(f"  Head start at C=10: horizon {horizon_wall:.1f}s - cold start "
          f"{m['pod_cold_start_wall_s']:.2f}s = {head:+.2f}s")
    if head <= 0:
        print("  >> NEGATIVE. A pod ordered on the t+1 forecast becomes Ready AFTER the")
        print("     minute it was forecast for. No t+1 predictive method can be proactive")
        print("     at this compression factor. This does not block the matrix - it is a")
        print("     RESULT (D-004/D-010) - but it must be reported, not averaged away.")
    elif head < 0.5 * horizon_wall:
        print(f"  >> Pod startup consumes {m['pod_cold_start_wall_s'] / horizon_wall:.0%} "
              f"of the forecast horizon.")

    print()
    if fails:
        print(f"  {len(fails)} check(s) FAILED - do not start the matrix.")
        print(f"  Run without --check to propagate the measured values.")
        return 1
    print("  All wiring checks passed. The matrix will consume measured values.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify wiring without changing anything (preflight gate)")
    args = ap.parse_args()

    m = load_measurements()
    if args.check:
        return do_check(m)

    print("\nPropagating measured constants")
    do_wire(m)
    print()
    return do_check(m)


if __name__ == "__main__":
    raise SystemExit(main())
