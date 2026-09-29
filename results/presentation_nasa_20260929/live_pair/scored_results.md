# Cold-start A/B pilot

> Pilot evidence only (n=2 pairs); not a significance claim.

Condition: 180 burst minutes, startup delay 10 s, measured readiness p95 12.715 s, seed 1001+rep.

| Arm | Rep | OK/total | Err | p95 | p99 | SLO% | Rep-s | Pend-s | Fcst RMSE | Under% | Inf p95 ms | Horizon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_horizon | 1 | 6695/6695 | 0 | 2380.5 | 4426.7 | 25.93 | 3282 | 90 | 14.6 | 44.5 | 16.20 | 1-1
| fixed_horizon | 2 | 6541/6541 | 0 | 5248.5 | 6990.9 | 46.63 | 3270 | 114 | 14.4 | 50.4 | 17.46 | 1-1
| readiness_aware | 1 | 6695/6695 | 0 | 1045.2 | 1814.7 | 14.74 | 3246 | 60 | 14.3 | 38.5 | 93.06 | 3-3
| readiness_aware | 2 | 6540/6541 | 1 | 2895.2 | 4668.5 | 29.57 | 3246 | 90 | 14.0 | 45.3 | 51.64 | 3-3

## Paired deltas (readiness-aware minus fixed)

- `latency_p95_ms`: mean 3814.53 -> 1970.19 (-48.4%); paired deltas -1335.38, -2353.29
- `latency_p99_ms`: mean 5708.77 -> 3241.59 (-43.2%); paired deltas -2612.00, -2322.35
- `slo_violations_pct`: mean 36.28 -> 22.15 (-38.9%); -14.12 percentage points; paired deltas -11.19, -17.06
- `replica_seconds_est`: mean 3276.00 -> 3246.00 (-0.9%); paired deltas -36.00, -24.00
- `pending_seconds_est`: mean 102.00 -> 75.00 (-26.5%); paired deltas -30.00, -24.00
- `forecast_rmse` mean: fixed 14.49; readiness-aware 14.14.
- `underforecast_pct` mean: fixed 47.48; readiness-aware 41.88.
- `inference_p95_ms` mean: fixed 16.84; readiness-aware 72.35.
- Request errors: fixed 0; readiness-aware 1 (one ClientPayloadError in readiness-aware rep 2).

Interpret carefully: request arrivals use identical segment and seed within each pair. The policy is the only intended change. The slow start is deliberately injected via `STARTUP_DELAY_S=10`; report this as a controlled KIND pilot, not production evidence. Use more repeats and confidence intervals before making a journal claim.
