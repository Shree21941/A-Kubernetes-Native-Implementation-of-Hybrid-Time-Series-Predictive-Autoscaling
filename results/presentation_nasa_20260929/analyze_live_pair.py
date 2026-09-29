"""Score the paired cold-start pilot from preserved request/decision logs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "live_pair"
SLO_MS = float(json.loads((ROOT / "experiments/measured_constants.json").read_text())["slo_ms"])
ARMS = ("fixed_horizon", "readiness_aware")
BURST = json.loads((ROOT / "load" / "segments.json").read_text())["segments"]["burst"]["y"]


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(int(len(ordered) * percent / 100), len(ordered) - 1)]


def score_run(arm: str, rep: int) -> dict:
    path = OUT / f"{arm}_rep{rep}"
    with (path / "requests.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    latencies = [float(row["latency_ms"]) for row in requests
                 if row["status"] == "200"]
    errors = len(requests) - len(latencies)
    violations = sum(value > SLO_MS for value in latencies) + errors

    decisions = [json.loads(line) for line in
                 (path / "decisions.jsonl").read_text().splitlines() if line.strip()]
    decisions = [row for row in decisions if row.get("kind") == "decision"]
    replicas = [int(row["spec_replicas"]) for row in decisions
                if row.get("spec_replicas") is not None]
    ready = [int(row["ready_replicas"]) for row in decisions
             if row.get("ready_replicas") is not None]
    actions = [row.get("action") for row in decisions]
    scale_actions = [action for action in actions
                     if action in ("scale_up", "scale_down")]
    reversals = sum(a != b for a, b in zip(scale_actions, scale_actions[1:]))
    forecast_pairs = []
    inference_times = []
    for row in decisions:
        if row.get("mode") != "forecast" or row.get("predicted_rate") is None:
            continue
        horizon = int(row.get("forecast_horizon_steps") or 1)
        target_index = int(row["cycle"]) + horizon
        if target_index < len(BURST):
            forecast_pairs.append((float(row["predicted_rate"]),
                                   float(BURST[target_index])))
        if row.get("inference_ms") is not None:
            inference_times.append(float(row["inference_ms"]))
    forecast_errors = [predicted - actual for predicted, actual in forecast_pairs]
    horizon_values = [int(row["forecast_horizon_steps"]) for row in decisions
                      if row.get("mode") == "forecast"
                      and row.get("forecast_horizon_steps") is not None]

    return {
        "arm": arm,
        "rep": rep,
        "requests_total": len(requests),
        "requests_ok": len(latencies),
        "requests_error": errors,
        "slo_ms": SLO_MS,
        "slo_violations_pct": 100 * violations / max(len(requests), 1),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "mean_spec_replicas": statistics.mean(replicas) if replicas else 0,
        "mean_ready_replicas": statistics.mean(ready) if ready else 0,
        "replica_seconds_est": sum(replicas) * 6.0,
        "pending_samples": sum(r > q for r, q in zip(replicas, ready)),
        "pending_seconds_est": 6.0 * sum(r > q for r, q in zip(replicas, ready)),
        "scale_actions": len(scale_actions),
        "direction_reversals": reversals,
        "forecast_samples": len(forecast_pairs),
        "forecast_rmse": (math.sqrt(statistics.mean(error * error
                            for error in forecast_errors)) if forecast_errors else None),
        "forecast_bias": statistics.mean(forecast_errors) if forecast_errors else None,
        "underforecast_pct": (100 * sum(actual > predicted
                                  for predicted, actual in forecast_pairs)
                              / len(forecast_pairs) if forecast_pairs else None),
        "inference_p50_ms": percentile(inference_times, 50) if inference_times else None,
        "inference_p95_ms": percentile(inference_times, 95) if inference_times else None,
        "forecast_steps_min": min(horizon_values) if horizon_values else None,
        "forecast_steps_max": max(horizon_values) if horizon_values else None,
        "decision_rows": len(decisions),
    }


def main() -> int:
    params = {}
    for line in (OUT / "experiment_params.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            params[key] = value
    reps = int(params.get("replicates", "0"))
    runs = [score_run(arm, rep) for arm in ARMS for rep in range(1, reps + 1)]
    payload = {
        "status": "pilot; not publication-grade significance evidence",
        "params": params,
        "runs": runs,
    }
    (OUT / "scored_results.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Cold-start A/B pilot",
        "",
        "> Pilot evidence only (n=2 pairs); not a significance claim.",
        "",
        f"Condition: {params.get('trace_minutes')} burst minutes, startup delay "
        f"{params.get('startup_delay_s')} s, measured readiness p95 "
        f"{params.get('measured_ready_p95_s')} s, seed 1001+rep.",
        "",
        "| Arm | Rep | OK/total | Err | p95 | p99 | SLO% | Rep-s | Pend-s | "
        "Fcst RMSE | Under% | Inf p95 ms | Horizon |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {row['arm']} | {row['rep']} | {row['requests_ok']}/{row['requests_total']} | "
            f"{row['requests_error']} | "
            f"{row['latency_p95_ms']:.1f} | {row['latency_p99_ms']:.1f} | "
            f"{row['slo_violations_pct']:.2f} | {row['replica_seconds_est']:.0f} | "
            f"{row['pending_seconds_est']:.0f} | "
            f"{row['forecast_rmse']:.1f} | {row['underforecast_pct']:.1f} | "
            f"{row['inference_p95_ms']:.2f} | "
            f"{row['forecast_steps_min']}-{row['forecast_steps_max']}"
        )
    lines.extend(["", "## Paired deltas (readiness-aware minus fixed)", ""])
    for metric in ("latency_p95_ms", "latency_p99_ms", "slo_violations_pct",
                   "replica_seconds_est", "pending_seconds_est"):
        fixed = {r["rep"]: r[metric] for r in runs if r["arm"] == "fixed_horizon"}
        aware = {r["rep"]: r[metric] for r in runs if r["arm"] == "readiness_aware"}
        deltas = [aware[rep] - fixed[rep] for rep in sorted(fixed.keys() & aware.keys())]
        fixed_mean = statistics.mean(fixed.values())
        aware_mean = statistics.mean(aware.values())
        relative = 100 * (aware_mean - fixed_mean) / fixed_mean if fixed_mean else 0.0
        detail = f"- `{metric}`: mean {fixed_mean:.2f} -> {aware_mean:.2f} ({relative:+.1f}%); "
        if metric == "slo_violations_pct":
            detail += f"{aware_mean - fixed_mean:+.2f} percentage points; "
        detail += f"paired deltas {', '.join(f'{d:+.2f}' for d in deltas)}"
        lines.append(detail)
    for metric in ("forecast_rmse", "underforecast_pct", "inference_p95_ms"):
        fixed = [r[metric] for r in runs if r["arm"] == "fixed_horizon"
                 and r[metric] is not None]
        aware = [r[metric] for r in runs if r["arm"] == "readiness_aware"
                 and r[metric] is not None]
        if fixed and aware:
            lines.append(f"- `{metric}` mean: fixed {statistics.mean(fixed):.2f}; "
                         f"readiness-aware {statistics.mean(aware):.2f}.")
    fixed_errors = sum(r["requests_error"] for r in runs if r["arm"] == "fixed_horizon")
    aware_errors = sum(r["requests_error"] for r in runs if r["arm"] == "readiness_aware")
    lines.append(f"- Request errors: fixed {fixed_errors}; readiness-aware {aware_errors} "
                 "(one ClientPayloadError in readiness-aware rep 2).")
    lines.extend([
        "",
        "Interpret carefully: request arrivals use identical segment and seed within each pair. "
        "The policy is the only intended change. The slow start is deliberately injected via "
        "`STARTUP_DELAY_S=10`; report this as a controlled KIND pilot, not production evidence. "
        "Use more repeats and confidence intervals before making a journal claim.",
        "",
    ])
    (OUT / "scored_results.md").write_text("\n".join(lines))
    print(OUT / "scored_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
