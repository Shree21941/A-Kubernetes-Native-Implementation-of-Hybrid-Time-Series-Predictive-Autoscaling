#!/usr/bin/env bash
# Phase 3 - build the whole testbed from nothing, idempotently.
#
#   ./k8s/setup.sh            create everything that is missing
#   ./k8s/setup.sh --rebuild  rebuild and reload the app image, then restart it
#   ./k8s/setup.sh --destroy  delete the cluster
#
# Safe to re-run: every stage checks whether its work is already done. That matters
# because the failure mode of a half-built cluster is not an error message, it is an
# experiment that produces plausible numbers from a broken environment.
set -euo pipefail

CLUSTER=aiops
NS=aiops
IMAGE=aiops/target-app:v1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
export PATH="$HOME/.local/bin:$PATH"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "${1:-}" == "--destroy" ]]; then
  say "deleting cluster $CLUSTER"
  kind delete cluster --name "$CLUSTER"
  exit 0
fi

# --- 0. preflight ----------------------------------------------------------
# Checked up front rather than allowed to surface three minutes into a Helm install.
for bin in docker kind kubectl helm; do
  command -v "$bin" >/dev/null || { echo "missing: $bin"; exit 1; }
done
docker info >/dev/null 2>&1 || {
  echo "ERROR: cannot talk to the Docker daemon as $(id -un)."
  echo "Fix:   sudo usermod -aG docker \$USER   then log out and back in."
  exit 1
}

# --- 1. cluster ------------------------------------------------------------
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  say "cluster '$CLUSTER' already exists"
else
  say "creating KIND cluster '$CLUSTER' (1 control-plane + 2 workers)"
  kind create cluster --name "$CLUSTER" --config "$HERE/kind-config.yaml" --wait 120s
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

say "waiting for all nodes Ready"
kubectl wait --for=condition=Ready nodes --all --timeout=180s
kubectl get nodes -o wide

# --- 2. app image ----------------------------------------------------------
# The image never goes to a registry; it is side-loaded straight into the node
# containers, which is why the Deployment must use imagePullPolicy: IfNotPresent.
if [[ "${1:-}" == "--rebuild" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "building $IMAGE"
  docker build -t "$IMAGE" "$ROOT/app"
fi
say "loading $IMAGE into cluster nodes"
kind load docker-image "$IMAGE" --name "$CLUSTER"

# --- 3. metrics-server -----------------------------------------------------
# Installed before Prometheus because it is the component most likely to be subtly
# broken on KIND, and a fast failure here is cheaper than one after a 2-minute install.
say "installing metrics-server (with the KIND TLS/address fixes)"
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/ >/dev/null 2>&1 || true
helm repo update metrics-server >/dev/null
helm upgrade --install metrics-server metrics-server/metrics-server \
  --namespace kube-system \
  --values "$HERE/metrics-server-values.yaml" \
  --wait --timeout 5m

# --- 4. kube-prometheus-stack ---------------------------------------------
say "installing kube-prometheus-stack (Prometheus + operator + kube-state-metrics)"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update prometheus-community >/dev/null
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values "$HERE/prometheus-values.yaml" \
  --wait --timeout 15m

# --- 5. the app ------------------------------------------------------------
# Applied last: the ServiceMonitor's CRD is created by the Prometheus operator above,
# so applying app.yaml first would fail on an unknown resource kind.
say "deploying the target app"
kubectl apply -f "$HERE/app.yaml"
kubectl -n "$NS" rollout status deploy/target-app --timeout=180s

if [[ "${1:-}" == "--rebuild" ]]; then
  say "restarting app to pick up the new image"
  kubectl -n "$NS" rollout restart deploy/target-app
  kubectl -n "$NS" rollout status deploy/target-app --timeout=180s
fi

say "setup complete - now run ./k8s/verify.sh"
