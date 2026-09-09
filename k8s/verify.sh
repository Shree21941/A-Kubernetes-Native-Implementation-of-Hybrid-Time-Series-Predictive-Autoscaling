#!/usr/bin/env bash
# Phase 3 Definition-of-Done check.
#
#   ./k8s/verify.sh
#
# Every item in the phase's DoD, checked and reported as PASS/FAIL. The one that matters
# most is metrics-server: CLAUDE.md 3 is explicit that a silently broken metrics-server
# invalidates the HPA comparison arm, and the symptom is `<unknown>` rather than an
# error, so it has to be asserted rather than assumed.
set -uo pipefail

NS=aiops
export PATH="$HOME/.local/bin:$PATH"
PASS=0; FAIL=0
ok()   { printf '  [\033[32mPASS\033[0m]  %-46s %s\n' "$1" "$2"; PASS=$((PASS+1)); }
bad()  { printf '  [\033[31mFAIL\033[0m]  %-46s %s\n' "$1" "$2"; FAIL=$((FAIL+1)); }

echo
echo "Phase 3 - testbed verification"
echo "======================================================================"

# --- nodes -----------------------------------------------------------------
total=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
ready=$(kubectl get nodes --no-headers 2>/dev/null | awk '$2=="Ready"' | wc -l)
[[ "$total" -ge 3 && "$ready" == "$total" ]] \
  && ok "All KIND nodes Ready" "$ready/$total" \
  || bad "All KIND nodes Ready" "$ready/$total ready"

# --- app pods --------------------------------------------------------------
running=$(kubectl -n $NS get pods -l app=target-app --no-headers 2>/dev/null | awk '$3=="Running"' | wc -l)
[[ "$running" -ge 1 ]] \
  && ok "App pods Running" "$running running" \
  || bad "App pods Running" "none running"

# --- metrics-server --------------------------------------------------------
# The DoD gate. `kubectl top` must return a real number, not <unknown> and not an error.
top=$(kubectl top pods -n $NS --no-headers 2>&1)
if echo "$top" | grep -qE '^[a-z0-9-]+ +[0-9]+m? +[0-9]+Mi'; then
  ok "kubectl top pods returns real CPU/memory" "$(echo "$top" | head -1 | tr -s ' ')"
else
  bad "kubectl top pods returns real CPU/memory" "$(echo "$top" | head -1)"
fi

api=$(kubectl get apiservice v1beta1.metrics.k8s.io -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null)
[[ "$api" == "True" ]] \
  && ok "metrics.k8s.io APIService Available" "True" \
  || bad "metrics.k8s.io APIService Available" "${api:-missing}"

# --- prometheus target -----------------------------------------------------
PROM=http://localhost:30090
up=$(curl -s --max-time 10 "$PROM/api/v1/query?query=up%7Bjob%3D%22target-app%22%7D" \
     | grep -o '"value"' | wc -l)
[[ "$up" -ge 1 ]] \
  && ok "Prometheus is scraping the app" "$up target(s) up" \
  || bad "Prometheus is scraping the app" "no 'up{job=\"target-app\"}' series"

# --- request-rate query ----------------------------------------------------
# Generate traffic first: rate() over a counter that has not moved is legitimately 0,
# so querying an idle app proves nothing either way.
echo "  ... sending 40 requests to build a rate"
for _ in $(seq 40); do curl -s -o /dev/null --max-time 5 http://localhost:30080/ & done
wait
sleep 20   # at least one 15s scrape interval, plus slack

q='sum(rate(http_requests_total%5B1m%5D))'
val=$(curl -s --max-time 10 "$PROM/api/v1/query?query=$q" \
      | grep -oP '"value":\[[0-9.]+,"\K[0-9.e+-]+' | head -1)
if [[ -n "${val:-}" ]] && awk "BEGIN{exit !($val > 0)}"; then
  ok "sum(rate(http_requests_total[1m])) > 0" "$val req/s"
else
  bad "sum(rate(http_requests_total[1m])) > 0" "got '${val:-<empty>}'"
fi

echo "======================================================================"
echo "  $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]] || { echo; echo "Phase 3 DoD NOT met."; exit 1; }
echo
echo "Phase 3 infrastructure DoD met. Cold-start still to measure:"
echo "  ./.venv/bin/python src/eval/coldstart.py"
