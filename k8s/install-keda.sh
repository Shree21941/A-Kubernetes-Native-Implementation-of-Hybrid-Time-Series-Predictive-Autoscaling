#!/usr/bin/env bash
# Phase 6 - install KEDA, for the request-rate reactive arm.
#
#   ./k8s/install-keda.sh
#   ./k8s/install-keda.sh --uninstall
#
# KEDA is installed as a separate step rather than folded into k8s/setup.sh because it is
# only needed by one arm, and because it adds a metrics APIService - the same class of
# component as metrics-server, which CLAUDE.md 3 records as a known source of silent
# failure on KIND. Keeping it separable means a broken KEDA install cannot take the rest
# of the testbed down with it.
#
# What KEDA actually is, since it is easy to over-imagine: it does not scale anything. It
# is a metrics adapter plus an operator. The operator watches ScaledObjects and creates an
# ordinary HorizontalPodAutoscaler for each one; the adapter serves the external metric
# that HPA then consumes. So an arm built on KEDA is still an HPA arm - which is exactly
# why it is the FAIR reactive comparison: it differs from arm-hpa-cpu only in the signal
# it reads, not in the machinery that acts on it.
set -euo pipefail

KEDA_VERSION="2.14.0"
NS=keda

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
info()  { printf '  %s\n' "$*"; }

if [[ "${1:-}" == "--uninstall" ]]; then
  helm uninstall keda -n "$NS" 2>/dev/null || true
  kubectl delete ns "$NS" --ignore-not-found
  green "KEDA removed"
  exit 0
fi

command -v helm >/dev/null || { red "helm not found"; exit 1; }
kubectl cluster-info >/dev/null 2>&1 || { red "no cluster reachable"; exit 1; }

info "installing KEDA ${KEDA_VERSION} into namespace ${NS}"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update kedacore >/dev/null

helm upgrade --install keda kedacore/keda \
  --namespace "$NS" --create-namespace \
  --version "$KEDA_VERSION" \
  --wait --timeout 5m \
  --set resources.operator.requests.cpu=50m \
  --set resources.metricServer.requests.cpu=50m \
  >/dev/null

# Verify, do not assume. A KEDA whose metrics APIService is not Available produces
# ScaledObjects that report READY=True while the HPA behind them reports <unknown> for
# its metric and never scales - the same silent-failure shape as a broken metrics-server,
# and it would show up in the results as "the reactive arm chose not to scale".
info "waiting for the external metrics APIService to become Available"
for i in $(seq 1 60); do
  status=$(kubectl get apiservice v1beta1.external.metrics.k8s.io \
    -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "")
  [[ "$status" == "True" ]] && break
  sleep 2
done

echo
if [[ "$status" == "True" ]]; then
  green "PASS  external.metrics.k8s.io APIService is Available"
else
  red   "FAIL  external.metrics.k8s.io APIService is not Available"
  red   "      ScaledObjects would report Ready while their HPAs see <unknown> and never scale."
  kubectl get apiservice v1beta1.external.metrics.k8s.io -o wide 2>/dev/null || true
  exit 1
fi

kubectl get pods -n "$NS" -o wide
echo
green "KEDA ready. Bring the arm up with: ./experiments/switch_arm.sh keda-rate"
