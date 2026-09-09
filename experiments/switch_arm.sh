#!/usr/bin/env bash
# Phase 6 - bring up exactly one autoscaler arm, and prove nothing else is running.
#
#   ./experiments/switch_arm.sh hpa-cpu
#   ./experiments/switch_arm.sh keda-rate
#   ./experiments/switch_arm.sh predictive-prophet    # ablation arm
#   ./experiments/switch_arm.sh predictive-hybrid     # the paper's method
#   ./experiments/switch_arm.sh none          # tear everything down, scale to baseline
#   ./experiments/switch_arm.sh --status      # what is active right now
#
# --------------------------------------------------------------------------------
# Why this script exists rather than three kubectl commands in the README
# --------------------------------------------------------------------------------
# Two autoscalers on one Deployment do not produce a blend of two behaviours; they
# produce a fight. Each sees the other's writes as cluster state to respond to, and the
# replica count oscillates on a period set by their relative sync intervals. The data is
# not merely noisy - it is data about an interaction that exists nowhere outside this
# mistake, and it looks superficially like an unstable controller.
#
# The failure is also SILENT. Nothing errors. `kubectl get hpa` shows a healthy HPA; the
# predictive controller logs confident decisions; the Deployment just behaves oddly. By
# the time it is noticed the experiment has been re-run several times.
#
# So teardown is unconditional and verified, not conditional and assumed: this script
# deletes every arm every time - including the one being brought up - waits for the API
# server to confirm they are gone, and refuses to start the new arm if anything remains.
#
# --------------------------------------------------------------------------------
# The reset protocol
# --------------------------------------------------------------------------------
# A run that starts from the previous run's inflated replica count is contaminated: the
# arm begins with capacity it never had to earn, and its first minutes of SLO data are
# free. Between arms we scale to BASELINE_REPLICAS, wait for the cluster to settle at
# exactly that number, and wait out any in-flight load.
set -euo pipefail

NS=aiops
DEPLOY=target-app
BASELINE_REPLICAS=1
SETTLE_TIMEOUT=120
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Every arm this project knows about. Adding an arm means adding it here, so that
# teardown can never miss one - a list that is only used for bring-up will eventually
# fail to tear something down.
ALL_ARMS=(hpa-cpu keda-rate predictive-prophet predictive-hybrid)

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
info()  { printf '  %s\n' "$*"; }

die() { red "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------------
# Detection: what is actually attached to the Deployment right now?
# ---------------------------------------------------------------------------------
# Deliberately queries the CLUSTER rather than reading a state file. A state file records
# what we believe we did; only the API server knows what is true, and the entire point of
# this check is to catch the case where those differ.
active_arms() {
  local found=()
  # Any HPA targeting our Deployment, EXCEPT one that a ScaledObject owns.
  #
  # KEDA is an operator: a ScaledObject makes it create an ordinary HPA (named
  # keda-hpa-<scaledobject>) which does the actual scaling. So one KEDA arm legitimately
  # produces TWO objects pointing at the Deployment, and counting both made this script
  # refuse to start the keda-rate arm with "expected exactly 1 active arm, found 2" -
  # a false positive that blocked a correctly-configured arm.
  #
  # Filtering on ownerReferences rather than the "keda-hpa-" name prefix, because the
  # prefix is a KEDA implementation detail and the ownership edge is the actual fact:
  # this HPA is not an independent autoscaler, it is how the ScaledObject acts.
  while read -r line; do
    [[ -z "$line" ]] && continue
    local name="${line%%|*}" owner="${line##*|}"
    [[ "$owner" == "ScaledObject" ]] && continue   # counted via its ScaledObject below
    found+=("hpa:$name")
  done < <(kubectl get hpa -n "$NS" -o \
    jsonpath="{range .items[?(@.spec.scaleTargetRef.name=='${DEPLOY}')]}{.metadata.name}{'|'}{.metadata.ownerReferences[0].kind}{'\n'}{end}" \
    2>/dev/null || true)

  while read -r name; do
    [[ -n "$name" ]] && found+=("scaledobject:$name")
  done < <(kubectl get scaledobject -n "$NS" -o \
    jsonpath="{range .items[?(@.spec.scaleTargetRef.name=='${DEPLOY}')]}{.metadata.name}{'\n'}{end}" \
    2>/dev/null || true)

  # The predictive controller holds no Kubernetes object pointing at the Deployment - it
  # just PATCHes it. So its presence has to be detected as a running Deployment of its
  # own. Missing this is the easiest way to end up with two active controllers.
  if kubectl get deploy predictive-autoscaler -n "$NS" >/dev/null 2>&1; then
    local r
    r=$(kubectl get deploy predictive-autoscaler -n "$NS" -o jsonpath='{.spec.replicas}')
    [[ "${r:-0}" -gt 0 ]] && found+=("controller:predictive-autoscaler(${r})")
  fi

  printf '%s\n' "${found[@]:-}"
}

teardown_all() {
  info "tearing down every arm (unconditionally)"
  kubectl delete -f "$REPO/k8s/arm-hpa-cpu.yaml"    --ignore-not-found --wait=true >/dev/null 2>&1 || true
  kubectl delete -f "$REPO/k8s/arm-keda-rate.yaml"  --ignore-not-found --wait=true >/dev/null 2>&1 || true
  # Scale to zero rather than delete: keeps the image and RBAC in place so the next
  # bring-up is fast, and makes "is it running" a single integer rather than a lookup.
  kubectl scale deploy/predictive-autoscaler -n "$NS" --replicas=0 >/dev/null 2>&1 || true

  # Also remove any HPA targeting our Deployment that we did NOT create - a leftover from
  # an interrupted run, or one created by hand during debugging.
  # Skip ScaledObject-owned HPAs here too. Deleting one directly is pointless - KEDA's
  # operator recreates it within seconds, so the confirmation loop below would spin until
  # it timed out. The ScaledObject deletion above is what removes them, by cascade.
  while read -r line; do
    [[ -z "$line" ]] && continue
    local name="${line%%|*}" owner="${line##*|}"
    [[ "$owner" == "ScaledObject" ]] && continue
    info "removing unrecognised HPA: $name"
    kubectl delete hpa "$name" -n "$NS" --ignore-not-found >/dev/null 2>&1 || true
  done < <(kubectl get hpa -n "$NS" -o \
    jsonpath="{range .items[?(@.spec.scaleTargetRef.name=='${DEPLOY}')]}{.metadata.name}{'|'}{.metadata.ownerReferences[0].kind}{'\n'}{end}" \
    2>/dev/null || true)

  # Confirm, do not assume. Deletion is asynchronous.
  local deadline=$((SECONDS + 60)) remaining
  while (( SECONDS < deadline )); do
    remaining=$(active_arms | grep -c . || true)
    [[ "$remaining" -eq 0 ]] && { info "confirmed: no arms attached"; return 0; }
    sleep 1
  done
  die "arms still attached after 60s: $(active_arms | tr '\n' ' ')"
}

reset_cluster() {
  info "reset: scaling $DEPLOY to $BASELINE_REPLICAS and waiting for it to settle"
  kubectl scale deploy/"$DEPLOY" -n "$NS" --replicas="$BASELINE_REPLICAS" >/dev/null

  # Wait for EXACTLY the baseline, both spec and ready, and for no extra pods to be
  # terminating. Terminating pods still hold CPU on the node, and a run that begins while
  # the previous run's pods are dying is measuring contention, not autoscaling.
  local deadline=$((SECONDS + SETTLE_TIMEOUT)) ready pods
  while (( SECONDS < deadline )); do
    ready=$(kubectl get deploy "$DEPLOY" -n "$NS" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    pods=$(kubectl get pods -n "$NS" -l app="$DEPLOY" --no-headers 2>/dev/null | wc -l)
    if [[ "${ready:-0}" -eq "$BASELINE_REPLICAS" && "$pods" -eq "$BASELINE_REPLICAS" ]]; then
      sleep 2   # brief quiet period so the node is genuinely idle
      ready=$(kubectl get deploy "$DEPLOY" -n "$NS" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
      pods=$(kubectl get pods -n "$NS" -l app="$DEPLOY" --no-headers 2>/dev/null | wc -l)
      if [[ "${ready:-0}" -eq "$BASELINE_REPLICAS" && "$pods" -eq "$BASELINE_REPLICAS" ]]; then
        green "  reset complete: $ready/$BASELINE_REPLICAS ready, $pods pod(s) total"
        return 0
      fi
    fi
    sleep 2
  done
  die "deployment did not settle at $BASELINE_REPLICAS within ${SETTLE_TIMEOUT}s"
}

check_load_quiet() {
  # A residual load generator from a previous run would drive the new arm before its
  # experiment has begun. We cannot see the host's processes from here, so we check the
  # symptom: is traffic still arriving?
  # Query Prometheus through the host NodePort (kind-config.yaml maps 30090) rather than
  # `kubectl exec` into its pod. The earlier version guessed the StatefulSet name and got
  # it wrong - the chart names it prometheus-prometheus-kube-prometheus-prometheus, not
  # prometheus-kube-prometheus-stack-prometheus - so this guard silently degraded to
  # "traffic check unavailable" on every switch and never actually guarded anything.
  # The NodePort has no name to guess and is the same path the load generator uses.
  local rate
  rate=$(curl -s -G --max-time 5 "http://localhost:30090/api/v1/query" \
    --data-urlencode "query=sum(rate(http_requests_total{namespace=\"$NS\"}[30s]))" \
    2>/dev/null | grep -oP '"value":\[[^,]+,"\K[^"]+' || echo "")
  if [[ -n "$rate" ]]; then
    if awk "BEGIN{exit !($rate > 0.5)}"; then
      red "  WARNING: traffic still arriving at ${rate} req/s - a load generator may still"
      red "           be running. Starting a run now would contaminate it."
      return 1
    fi
    info "traffic quiet (${rate} req/s)"
  else
    info "traffic check unavailable (Prometheus not reachable from here) - check manually"
  fi
  return 0
}

bring_up() {
  case "$1" in
    hpa-cpu)
      kubectl apply -f "$REPO/k8s/arm-hpa-cpu.yaml" >/dev/null
      info "applied k8s/arm-hpa-cpu.yaml" ;;
    keda-rate)
      kubectl get crd scaledobjects.keda.sh >/dev/null 2>&1 \
        || die "KEDA is not installed - run ./k8s/install-keda.sh first"
      kubectl apply -f "$REPO/k8s/arm-keda-rate.yaml" >/dev/null
      info "applied k8s/arm-keda-rate.yaml" ;;
    predictive-prophet|predictive-hybrid)
      kubectl get deploy predictive-autoscaler -n "$NS" >/dev/null 2>&1 \
        || die "predictive-autoscaler not deployed - apply k8s/controller.yaml first"
      # ONE Deployment, its FORECASTER env var patched - not two Deployments. The two
      # predictive arms must share every line of code except which forecaster they
      # consult, and two Deployments would eventually drift in some second detail,
      # at which point the ablation stops being an ablation. It also makes it
      # structurally impossible for both predictive arms to run at once.
      local which="${1#predictive-}"
      kubectl set env deploy/predictive-autoscaler -n "$NS" "FORECASTER=$which" >/dev/null
      kubectl scale deploy/predictive-autoscaler -n "$NS" --replicas=1 >/dev/null
      kubectl rollout status deploy/predictive-autoscaler -n "$NS" --timeout=180s >/dev/null
      # Confirm the running pod actually carries the env we asked for. `kubectl set env`
      # triggers a rollout; reading the Deployment spec would tell us what we requested,
      # not what is running.
      local live
      live=$(kubectl get pods -n "$NS" -l app=predictive-autoscaler \
        -o jsonpath='{.items[0].spec.containers[0].env[?(@.name=="FORECASTER")].value}')
      [[ "$live" == "$which" ]] || die "pod reports FORECASTER=$live, expected $which"
      info "predictive-autoscaler running with FORECASTER=$live" ;;
    none) info "no arm requested" ;;
    *) die "unknown arm '$1' (known: ${ALL_ARMS[*]} none)" ;;
  esac
}

# ---------------------------------------------------------------------------------
main() {
  local arm="${1:-}"
  [[ -z "$arm" ]] && { echo "usage: $0 {${ALL_ARMS[*]}|none|--status}"; exit 2; }

  kubectl get ns "$NS" >/dev/null 2>&1 || die "namespace $NS not found - run k8s/setup.sh"

  if [[ "$arm" == "--status" ]]; then
    echo
    echo "Active autoscalers on ${NS}/${DEPLOY}:"
    local found
    found=$(active_arms | grep -c . || true)
    if [[ "$found" -eq 0 ]]; then
      green "  none"
    else
      active_arms | sed 's/^/  /'
      [[ "$found" -gt 1 ]] && red "  ^ MORE THAN ONE ACTIVE - any data collected now is invalid"
    fi
    kubectl get deploy "$DEPLOY" -n "$NS" \
      -o custom-columns=DEPLOY:.metadata.name,SPEC:.spec.replicas,READY:.status.readyReplicas
    exit 0
  fi

  echo
  echo "Switching to arm: $arm"
  teardown_all
  check_load_quiet || true      # warn, do not block - the operator may know better
  reset_cluster
  bring_up "$arm"

  # Post-condition: exactly the arm we asked for, and nothing else.
  sleep 3
  local names count
  names=$(active_arms | grep -c . || true)
  count="${names:-0}"
  echo
  if [[ "$arm" == "none" ]]; then
    [[ "$count" -eq 0 ]] && green "OK - no arm active, deployment at baseline" \
                         || die "expected no arms, found: $(active_arms | tr '\n' ' ')"
  else
    [[ "$count" -eq 1 ]] || die "expected exactly 1 active arm, found $count: $(active_arms | tr '\n' ' ')"
    green "OK - exactly one arm active: $(active_arms)"
  fi
}

main "$@"
