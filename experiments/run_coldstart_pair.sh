#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/presentation_nasa_20260929/live_pair"
NS=aiops
DEPLOY=target-app
TRACE_START="1995-08-24 06:52:00"
PROM="http://127.0.0.1:19090"
APP="http://172.18.0.2:30080/"
READINESS_MEAN="12.6403"
READINESS_P95="12.715"
REPS="${1:-2}"
MINUTES="${2:-180}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$OUT"

for bin in kubectl docker; do
  command -v "$bin" >/dev/null || { echo "missing: $bin" >&2; exit 1; }
done

kubectl get nodes >/dev/null
kubectl -n "$NS" get deployment "$DEPLOY" >/dev/null
kubectl -n "$NS" get deployment predictive-autoscaler >/dev/null
curl -fsS --max-time 5 "$APP/readyz" >/dev/null
curl -fsS --max-time 5 "$PROM/-/ready" >/dev/null

active_hpas=$(kubectl -n "$NS" get hpa -o jsonpath='{range .items[*]}{.spec.scaleTargetRef.name}{"\n"}{end}')
active_sos=$(kubectl -n "$NS" get scaledobject -o jsonpath='{range .items[*]}{.spec.scaleTargetRef.name}{"\n"}{end}' 2>/dev/null || true)
if printf '%s\n%s\n' "$active_hpas" "$active_sos" | grep -qx "$DEPLOY"; then
  echo "Refusing paired run: another autoscaler targets $NS/$DEPLOY" >&2
  exit 1
fi

cat > "$OUT/experiment_params.txt" <<EOF
cluster=$(kubectl config current-context)
segment=burst
trace_start=$TRACE_START
trace_minutes=$MINUTES
compression=10
replicates=$REPS
startup_delay_s=10
measured_ready_mean_s=$READINESS_MEAN
measured_ready_p95_s=$READINESS_P95
capacity_rps=3.135
slo_p95_ms=593.7
seed_per_rep=1001+rep
arms=fixed_horizon,readiness_aware
EOF

run_arm() {
  local arm="$1" rep="$2" aware="$3" tag="coldstart-$1-r$2"
  local run_dir pod run_path
  run_dir="$OUT/${arm}_rep${rep}"
  mkdir -p "$run_dir"

  echo "[$arm rep $rep] resetting app to one Ready replica"
  kubectl -n "$NS" scale "deployment/$DEPLOY" --replicas=1 >/dev/null
  kubectl -n "$NS" rollout status "deployment/$DEPLOY" --timeout=180s

  echo "[$arm rep $rep] setting policy and trace clock"
  kubectl -n "$NS" set env deployment/predictive-autoscaler \
    "READINESS_AWARE_SCALING=$aware" \
    "FORECASTER=hybrid" \
    "TRACE_START=$TRACE_START" \
    "POD_COLD_START_WALL_S=$READINESS_MEAN" \
    "POD_COLD_START_P95_WALL_S=$READINESS_P95" \
    POD_COLD_START_MEASURED=1 \
    MAX_FORECAST_HORIZON_TRACE_MIN=5 >/dev/null
  kubectl -n "$NS" rollout status deployment/predictive-autoscaler --timeout=180s

  kubectl -n "$NS" get deployment predictive-autoscaler -o yaml > "$run_dir/controller.yaml"
  pod=$(kubectl -n "$NS" get pods -l app=predictive-autoscaler -o jsonpath='{.items[0].metadata.name}')
  kubectl -n "$NS" logs "$pod" > "$run_dir/controller-startup.log"

  echo "[$arm rep $rep] replaying $MINUTES burst minutes (seed $((1000 + rep)))"
  docker run --rm --network host -v "$ROOT:/workspace" -w /workspace \
    python:3.11-slim sh -c \
    'pip install --quiet aiohttp && python load/trace_replay.py --segment burst --url "$4" --compression 10 --minutes "$1" --seed "$2" --tag "$3"' \
    sh "$MINUTES" "$((1000 + rep))" "$tag" "$APP" > "$run_dir/replay.log" 2>&1

  run_path=$(find "$ROOT/experiments" -maxdepth 1 -type d -name "*-burst-$tag" -print | sort | tail -n 1)
  if [[ -z "$run_path" || ! -f "$run_path/summary.json" ]]; then
    echo "Missing trace replay summary for $arm rep $rep" >&2
    tail -60 "$run_dir/replay.log" >&2
    return 1
  fi
  cp "$run_path/summary.json" "$run_dir/summary.json"
  cp "$run_path/raw/requests.csv" "$run_dir/requests.csv"
  pod=$(kubectl -n "$NS" get pods -l app=predictive-autoscaler -o jsonpath='{.items[0].metadata.name}')
  kubectl -n "$NS" exec "$pod" -- cat /var/log/controller/decisions.jsonl > "$run_dir/decisions.jsonl"
  kubectl -n "$NS" logs "$pod" > "$run_dir/controller.log"
  echo "[$arm rep $rep] saved evidence in $run_dir"
}

for rep in $(seq 1 "$REPS"); do
  run_arm fixed_horizon "$rep" 0
  run_arm readiness_aware "$rep" 1
done

echo "Paired run complete: $OUT"
