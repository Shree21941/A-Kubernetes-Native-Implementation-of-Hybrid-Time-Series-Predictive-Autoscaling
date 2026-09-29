# Evidence Pack: Paper Baseline and Cold-Start-Aware Autoscaling

**Evidence cutoff:** 29 September 2026  
**Reference:** Dang-Quang and Yoo, *Deep Learning-Based Autoscaling Using
Bidirectional Long Short-Term Memory for Kubernetes*, Applied Sciences 11 (2021), 3835.

## NASA input verification

The attached July/August 1995 archives match the copies in `data/raw/`. The project
parser counted 3,461,612 requests, skipped one malformed/truncated line, and verified
the `-0400` timezone. The canonical continuous series contains 89,280 one-minute bins,
including 7,884 zero-request minutes. Its sum matches the documented request count.

## Refreshed NASA Table 4 comparison

One full CPU run, seed 42, canonical zero-filled NASA series, 70/30 chronological split.
Values are RMSE in requests/minute. The reference values are from the paper's Table 4.

| Horizon | Model | Paper RMSE | Re-run RMSE | Relative difference |
|---|---|---:|---:|---:|
| 1-step | ARIMA | 14.010 | 14.063 | +0.4% |
| 1-step | Bi-LSTM | 13.551 | 13.788 | +1.7% |
| 5-step | ARIMA | 15.414 | 15.481 | +0.4% |
| 5-step | Bi-LSTM | 14.390 | 14.594 | +1.4% |

This is a close single-run accuracy reproduction. It is not bit-identical: neural
training is stochastic, the CPU/runtime differs from the paper's TPU environment, and
the canonical loader now restores the zero-request minutes omitted by the older sparse
CSV. Do not compare prediction-speed values across those environments.

Full machine-readable output and plots are in `tables/table_nasa_accuracy.csv` and
`figures/`. The 3-epoch smoke run is not included as evidence.

## Optimization implementation

The controller derives forecast steps from readiness p95 plus forecast overhead,
recursively applies the packaged one-step forecaster up to a configured cap, logs the
required/used steps and any uncovered readiness lead, and holds scale-down while
requested replicas are not all Ready. The measured readiness p95 is 2.8662 s from five
1-to-5-pod trials; at compression 10, one trace-minute is 6 s. Thus the configured
lead is currently one step, identical to the fixed-horizon arm.

A separate controlled slow-start condition now injects `STARTUP_DELAY_S=10` into the
app. Across 20 scale-outs from one to five replicas, all five became Ready in 12.64 s
on average; the interpolated p95 is 12.715 s. This crosses the 6-second control
interval and selects a 3-step lead. Raw trials are in `slow_start_coldstart.json`.

Validation completed: 24 planner assertions, recursive forecasting tests, 2 data-loader
tests, measured-configuration consistency, and Python compilation. The KIND control
plane later recovered, and the app, Prometheus ServiceMonitor, and predictive controller
were deployed. Two paired burst replays (seeds 1001 and 1002) completed. The fixed arm
used one step; the readiness-aware arm used three.

| Metric | Fixed one-step mean | Readiness-aware mean | Mean change |
|---|---:|---:|---:|
| p95 latency | 3,814.5 ms | 1,970.2 ms | -48.4% |
| p99 latency | 5,708.8 ms | 3,241.6 ms | -43.2% |
| SLO violations (>593.7 ms) | 36.28% | 22.15% | -14.12 percentage points (-38.9%) |
| Estimated replica-seconds | 3,276 | 3,246 | -0.9% |
| Estimated pending seconds | 102 | 75 | -26.5% |
| Scale actions / reversals per run | 8.5 / 4 | 7.5 / 3 | lower |
| Request errors | 0 | 1 total | one ClientPayloadError in rep 2 |

Both matched seeds reduced p95, p99, SLO violations, estimated replica-seconds, and
pending time. Paired p95 deltas were -1,335.4 ms and -2,353.3 ms; paired SLO deltas
were -11.19 and -17.06 percentage points.

**Claim boundary:** this is a consistent two-pair pilot win under a controlled
10-second startup delay. It is still only n=2, not statistical or publication-grade
evidence. One request failed in readiness-aware rep 2. This compares horizon policies
using the same live hybrid forecaster, not the paper's Bi-LSTM/HPA system, so it does
not establish superiority over the paper itself.

## Next experiment required for an optimization claim

1. Increase paired repetitions (ideally at least 10) and report confidence intervals.
2. Repeat under natural startup and additional controlled startup delays.
3. Run both arms on identical load segments, with
   the same forecast model, capacity, limits, and replica bounds.
4. Report SLO violations/p95 latency together with replica-seconds, pending-pod time,
   scale actions, and confidence intervals. Keep the paper's simulated Algorithm 1 as a
   separate reference; the live hybrid controller is not the paper's Bi-LSTM controller.
