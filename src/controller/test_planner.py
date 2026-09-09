"""Phase 5 - tests for the safety layer.

    ./.venv/bin/python src/controller/test_planner.py

The planner is pure logic, so it can be tested exhaustively with no cluster, no model
and no load. That is worth doing properly: every rule here exists to stop a specific
failure, and a rule that silently stops working looks exactly like a rule that was never
needed.

The last test is the Phase 5 Definition-of-Done item "safety layer demonstrably prevents
flapping in the quiet segment" - driven by the REAL frozen segment data, not a synthetic
sawtooth.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "controller")]

from config import Config  # noqa: E402
from planner import Action, Planner  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, PASS if cond else FAIL, detail))


def base_cfg(**kw) -> Config:
    return Config(**kw)


# ---------------------------------------------------------------------------------
# 1. The formula
# ---------------------------------------------------------------------------------
def test_formula() -> None:
    cfg = base_cfg()
    p = Planner(cfg)
    cap = cfg.per_pod_capacity_rpm_trace           # 19.47 req/trace-min

    # A fraction of a pod cannot exist: just over one pod's worth must ask for two.
    d = p.plan(cap + 0.01, current=1, now=1000.0)
    check("ceil: 1.0005 pods' load -> 2 replicas", d.desired_raw == 2,
          f"raw={d.desired_raw}")

    # Exactly one pod's worth must NOT round up.
    p2 = Planner(cfg)
    d = p2.plan(cap, current=1, now=1000.0)
    check("ceil: exactly 1 pod's load -> 1 replica", d.desired_raw == 1,
          f"raw={d.desired_raw}")

    # Clamping, both ends.
    p3 = Planner(cfg)
    d = p3.plan(1e9, current=1, now=1000.0)
    check("clamp: absurd forecast capped at max_replicas",
          d.desired_clamped == cfg.max_replicas and d.target == cfg.max_replicas,
          f"raw={d.desired_raw} clamped={d.desired_clamped}")

    p4 = Planner(cfg)
    d = p4.plan(0.0, current=1, now=1000.0)
    check("clamp: zero forecast never goes below min_replicas",
          d.desired_clamped >= cfg.min_replicas, f"clamped={d.desired_clamped}")


# ---------------------------------------------------------------------------------
# 2. The corrected Equations 8 / 9
# ---------------------------------------------------------------------------------
def test_equations() -> None:
    cfg = base_cfg()
    # As printed, both equations test `desired < current`, so the up branch can never
    # fire. If this test fails, the typo has been reintroduced.
    p = Planner(cfg)
    d = p.plan(200.0, current=2, now=1000.0)      # 200/19.47 = 11 -> clamped to 10
    check("Eq.8 corrected: rising demand scales UP",
          d.action is Action.SCALE_UP and d.target > 2,
          f"action={d.action.value} target={d.target}")

    p = Planner(cfg)
    d = p.plan(5.0, current=8, now=1000.0)
    check("Eq.9: falling demand moves toward scale-down",
          d.action in (Action.SCALE_DOWN, Action.HOLD_CONFIRMING),
          f"action={d.action.value}")


# ---------------------------------------------------------------------------------
# 3. Dead-band
# ---------------------------------------------------------------------------------
def test_deadband() -> None:
    cfg = base_cfg(deadband_replicas=2)
    cap = cfg.per_pod_capacity_rpm_trace

    # DOWN by one pod: absorbed.
    p = Planner(cfg)
    d = p.plan(cap * 2.5, current=4, now=1000.0)          # wants 3, has 4
    check("dead-band absorbs a 1-replica scale-DOWN",
          d.action is Action.HOLD_DEADBAND and d.target == 4,
          f"action={d.action.value} clamped={d.desired_clamped}")

    # DOWN by two pods: acted on (via the confirmation path).
    p = Planner(cfg)
    d = p.plan(cap * 1.5, current=4, now=1000.0)          # wants 2, has 4
    check("dead-band does NOT absorb a 2-replica scale-DOWN",
          d.action is Action.HOLD_CONFIRMING, f"action={d.action.value}")

    # UP by one pod: NOT absorbed. Dead-banding the up direction would delay the head
    # start the forecast exists to buy.
    p = Planner(cfg)
    d = p.plan(cap * 3.5, current=3, now=1000.0)          # wants 4, has 3
    check("dead-band never absorbs a scale-UP",
          d.action is Action.SCALE_UP and d.target == 4,
          f"action={d.action.value} target={d.target}")

    # A band of 1 filters nothing - the defect that shipped in the first config.
    one = base_cfg(deadband_replicas=1)
    p = Planner(one)
    d = p.plan(one.per_pod_capacity_rpm_trace * 2.5, current=4, now=1000.0)
    check("dead-band of 1 is a no-op (regression guard)",
          d.action is not Action.HOLD_DEADBAND, f"action={d.action.value}")


# ---------------------------------------------------------------------------------
# 4. Cooldowns, and the physical floor under the scale-up cooldown
# ---------------------------------------------------------------------------------
def test_cooldown() -> None:
    cfg = base_cfg()
    cap = cfg.per_pod_capacity_rpm_trace
    p = Planner(cfg)
    t = 1000.0

    # Deliberately NOT scaling to max_replicas here. An earlier version of this test
    # asked for 200 req/trace-min, which clamps to 10 = max; the follow-up calls then
    # had desired == current and returned HOLD_NO_CHANGE, so the cooldown was never
    # exercised and the test passed without testing anything.
    d1 = p.plan(cap * 2.5, current=1, now=t)              # wants 3
    check("first scale-up is allowed (no history)",
          d1.action is Action.SCALE_UP and d1.target == 3,
          f"action={d1.action.value} target={d1.target}")

    d2 = p.plan(cap * 4.5, current=d1.target, now=t + 1.0)   # wants 5, still cooling
    check("second scale-up inside cooldown is held",
          d2.action is Action.HOLD_COOLDOWN and d2.target == d1.target,
          f"action={d2.action.value} target={d2.target}")

    d3 = p.plan(cap * 4.5, current=d1.target,
                now=t + cfg.scale_up_cooldown_wall_s + 0.1)
    check("scale-up allowed once cooldown elapses",
          d3.action is Action.SCALE_UP and d3.target == 5,
          f"action={d3.action.value} target={d3.target}")

    # The cooldown must never be shorter than pod startup, or the controller decides
    # against a cluster that has not yet responded to its last decision.
    slow = base_cfg(pod_cold_start_wall_s=45.0)
    check("scale-up cooldown respects the physical pod-start floor",
          slow.scale_up_cooldown_wall_s >= 45.0,
          f"cooldown={slow.scale_up_cooldown_wall_s}s")
    fast = base_cfg(pod_cold_start_wall_s=0.5)
    check("scale-up cooldown is the compressed interval when startup is fast",
          math.isclose(fast.scale_up_cooldown_wall_s, 6.0),
          f"cooldown={fast.scale_up_cooldown_wall_s}s")


# ---------------------------------------------------------------------------------
# 5. Asymmetry: down is damped, up is not
# ---------------------------------------------------------------------------------
def test_asymmetry() -> None:
    cfg = base_cfg()
    p = Planner(cfg)
    t = 1000.0
    # Scale down requires N consecutive confirmations.
    seen = []
    for i in range(cfg.scale_down_confirmations):
        d = p.plan(5.0, current=8, now=t + i * cfg.control_period_wall_s)
        seen.append(d.action)
    check("scale-down needs N confirmations before acting",
          all(a is Action.HOLD_CONFIRMING for a in seen[:-1])
          and seen[-1] is Action.SCALE_DOWN,
          f"sequence={[a.value for a in seen]}")

    # Scale up acts on the first cycle, with no confirmation.
    p = Planner(cfg)
    d = p.plan(200.0, current=1, now=t)
    check("scale-up acts immediately (no confirmation)",
          d.action is Action.SCALE_UP, f"action={d.action.value}")


# ---------------------------------------------------------------------------------
# 6. Warm-up
# ---------------------------------------------------------------------------------
def test_warmup() -> None:
    cfg = base_cfg()
    p = Planner(cfg)
    d = p.plan(500.0, current=1, now=1000.0, warming_up=True)
    check("warm-up refuses to scale on a partial window",
          d.action is Action.HOLD_WARMUP and d.target == 1,
          f"action={d.action.value} target={d.target}")


# ---------------------------------------------------------------------------------
# 7. DoD: no flapping on the real quiet segment
# ---------------------------------------------------------------------------------
def test_no_flapping_quiet() -> None:
    """Drive the planner with the actual frozen quiet segment.

    The segment collapses from ~97 req/min to a 9-minute run of exact zeros and recovers
    to ~88 within 30 minutes. An undamped planner scales down into the dropout and back
    up on recovery; the damping is meant to make that a single, late, cheap movement
    rather than a rapid oscillation.

    "Flapping" is defined here as a DIRECTION REVERSAL: a scale-down followed by a
    scale-up, or vice versa, within a short window. That is the thing that costs
    something - each reversal pays a full pod cold start in wall time that does not
    compress. Counting raw scaling actions instead would penalise a smooth ramp, which
    is not a defect.
    """
    seg = json.loads((REPO / "load" / "segments.json").read_text())["segments"]["quiet"]
    y = seg["y"]

    def run(cfg: Config) -> tuple[list[int], int, int]:
        p = Planner(cfg)
        replicas = cfg.min_replicas
        series, actions = [], []
        for i, rate in enumerate(y):
            # Perfect-foresight forecast: feed the planner the true next-minute rate.
            # This isolates the SAFETY layer - any oscillation here is the planner's,
            # not the forecaster's, which is what this test is about.
            nxt = float(y[i + 1]) if i + 1 < len(y) else float(rate)
            d = p.plan(nxt, current=replicas, now=i * cfg.control_period_wall_s)
            replicas = d.target
            series.append(replicas)
            if d.action in (Action.SCALE_UP, Action.SCALE_DOWN):
                actions.append(d.action)
        reversals = sum(1 for a, b in zip(actions, actions[1:]) if a is not b)
        return series, len(actions), reversals

    damped = base_cfg()
    undamped = base_cfg(scale_down_confirmations=1, deadband_replicas=1,
                        scale_down_cooldown_trace_min=0.0)

    s_d, n_d, rev_d = run(damped)
    s_u, n_u, rev_u = run(undamped)

    check("quiet segment: damping reduces direction reversals",
          rev_d < rev_u, f"damped {rev_d} reversals vs undamped {rev_u}")
    check("quiet segment: damped controller never drops below min_replicas",
          min(s_d) >= damped.min_replicas, f"min={min(s_d)}")
    check("quiet segment: damped controller still tracks the recovery",
          max(s_d) >= math.ceil(max(y) / damped.per_pod_capacity_rpm_trace) - 1,
          f"peak replicas {max(s_d)} for peak load {max(y)} req/min")

    print(f"\n  quiet-segment flapping ({len(y)} trace-minutes, perfect foresight)")
    print(f"    {'configuration':<34} {'actions':>9} {'reversals':>10} "
          f"{'min':>5} {'max':>5}")
    print(f"    {'damped (shipping defaults)':<34} {n_d:>9} {rev_d:>10} "
          f"{min(s_d):>5} {max(s_d):>5}")
    print(f"    {'undamped (safety layer off)':<34} {n_u:>9} {rev_u:>10} "
          f"{min(s_u):>5} {max(s_u):>5}")
    return None


def main() -> int:
    for fn in (test_formula, test_equations, test_deadband, test_cooldown,
               test_asymmetry, test_warmup, test_no_flapping_quiet):
        fn()

    print("\nPhase 5 - planner tests")
    print("=" * 78)
    width = max(len(n) for n, _, _ in results)
    for name, verdict, detail in results:
        print(f"  {verdict:<5} {name:<{width}}  {detail}")
    n_fail = sum(1 for _, v, _ in results if v == FAIL)
    print("=" * 78)
    print(f"  {len(results) - n_fail}/{len(results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
