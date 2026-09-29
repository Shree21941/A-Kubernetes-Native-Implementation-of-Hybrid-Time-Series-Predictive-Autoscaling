# Cold-Start-Aware Autoscaling

This extension keeps the live forecast model unchanged and makes the planner account
for the delay between requesting replicas and those replicas becoming Ready. The
comparison arm is the same controller with a fixed one-trace-minute forecast horizon.
The paper-faithful offline implementation remains in `run_experiment.py`.

## Policy

Let `period_wall_s` be the controller interval, `ready_p95_wall_s` the measured
scale-out-to-Ready p95, and `overhead_wall_s` a conservative allowance for monitoring,
inference, and applying the scale request. The required forecast lead is:

```text
required_steps = max(1, ceil((ready_p95_wall_s + overhead_wall_s) / period_wall_s))
```

The controller recursively applies its packaged one-step model for the selected lead,
up to `MAX_FORECAST_HORIZON_TRACE_MIN` (default 5). Recursive error compounds, so the
cap is explicit and every decision logs both required and used steps. If readiness takes
longer than the capped forecast covers, the controller uses the maximum of that forecast
and the recent observed demand as a conservative fallback, and logs the uncovered lead.
Scale-down is held while requested replicas are not all Ready. Scale-up remains
immediate; the startup-aware cooldown prevents duplicate requests while earlier pods are
still starting.

An optional additive forecast margin can be set with
`FORECAST_ERROR_MARGIN_RPM_TRACE`. Keep it at zero until its value is calibrated as a
one-sided under-forecast quantile on validation data. Do not calculate it from the test
segments used for final results.

## Current Measurements

The saved KIND measurement is 2.7 s mean and 2.8662 s interpolated p95, from five
1-to-5-replica trials with images already loaded. At compression 10, one trace-minute is
6 wall-seconds, so the required lead is currently one step. That means the optimized arm
does not need a longer horizon under this measured condition; test startup under load or
node contention to evaluate the multi-step behavior. Five trials are not enough for a
robust tail estimate; increase repetitions before presenting p95 as a strong result.

The capacity in this testbed is 3.135 requests/s per pod at the configured latency SLO.
Do not compare these live capacity units directly with the paper's simulated assumption
of 500 requests/minute per pod.

## Run and Compare

Reproduce the paper's offline tables first, then build and deploy the updated live
controller:

```sh
python3 run_experiment.py --dataset both --autoscaler
./k8s/deploy_predictive_autoscaler.sh
# For a cluster with a non-default name:
KIND_CLUSTER_NAME=aiops-cluster ./k8s/deploy_predictive_autoscaler.sh
```

The Kubernetes manifest enables readiness-aware mode. The fixed-horizon comparison uses
the same image and forecast bundle, changing only the policy flag:

```sh
kubectl -n aiops set env deployment/predictive-autoscaler READINESS_AWARE_SCALING=0
# Wait for rollout, run the same trace/load, and archive its decision log and metrics.
kubectl -n aiops rollout status deployment/predictive-autoscaler
kubectl -n aiops set env deployment/predictive-autoscaler READINESS_AWARE_SCALING=1
# Repeat the identical trace/load and archive the second run.
kubectl -n aiops rollout status deployment/predictive-autoscaler
```

Record the manifest/configuration and decision log for each arm. Use the same traces,
load, replica bounds, and capacity calibration. Report p95/p99 latency and SLO violations
alongside replica-seconds/CPU-seconds, scaling reversals, readiness delay, and the
forecast horizon used. Repeat runs and report uncertainty; do not claim a latency win
when intervals overlap. KIND nodes share a host, so describe results as a reproducible
local-cluster evaluation rather than independent physical nodes.

The current saved cold-start p95 comes from only five trials, and measured startup is
shorter than one control interval. These settings exercise the optimized policy but do
not demonstrate that a longer forecast lead improves outcomes. Before a performance
claim, collect more readiness trials and include a controlled condition where p95
readiness exceeds one interval, keeping workload and resource limits matched across arms.

## Validation

```sh
python3 src/controller/test_planner.py
python3 src/controller/test_forecaster_horizon.py
python3 -m py_compile src/controller/config.py src/controller/planner.py \
  src/controller/forecaster.py src/controller/main.py src/eval/coldstart.py \
  experiments/wire_measurements.py
```

The first two commands need only Python and NumPy; running the controller additionally
requires the project dependencies, a reachable Prometheus, the bundle, and KIND.