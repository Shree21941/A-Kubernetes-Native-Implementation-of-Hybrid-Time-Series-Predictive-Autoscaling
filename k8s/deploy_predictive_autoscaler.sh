#!/usr/bin/env bash
set -euo pipefail

CLUSTER="${KIND_CLUSTER_NAME:-aiops}"
CONTROL_PLANE="${CLUSTER}-control-plane"
IMAGE=aiops/predictive-autoscaler:dev
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

for bin in docker kind kubectl; do
  command -v "$bin" >/dev/null || { echo "missing: $bin" >&2; exit 1; }
done

kind get clusters | grep -qx "$CLUSTER" || {
  echo "KIND cluster '$CLUSTER' is not running; run ./k8s/setup.sh first." >&2
  exit 1
}
kubectl config use-context "kind-$CLUSTER" >/dev/null
kubectl -n aiops get deployment target-app >/dev/null

docker build -f "$ROOT/src/controller/Dockerfile" -t "$IMAGE" "$ROOT"
kind load docker-image "$IMAGE" --name "$CLUSTER" --nodes "$CONTROL_PLANE"
kubectl apply -f "$ROOT/k8s/controller.yaml"
kubectl -n aiops rollout restart deployment/predictive-autoscaler
kubectl -n aiops rollout status deployment/predictive-autoscaler --timeout=180s
kubectl -n aiops get deployment predictive-autoscaler