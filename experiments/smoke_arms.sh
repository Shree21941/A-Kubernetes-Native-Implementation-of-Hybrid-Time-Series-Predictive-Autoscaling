#!/usr/bin/env bash
# Phase 6/7 - prove every arm actually scales, one at a time, against real load.
#
#   ./experiments/smoke_arms.sh
#
# The static parity checks in verify_arms.py prove the arms are CONFIGURED identically.
# They cannot prove an arm WORKS. The failure this catches is the one CLAUDE.md 3 warns
# about twice: a metric that never resolves produces an autoscaler that sits at
# minReplicas for the whole run, reports Healthy, and looks in the results like an arm
# that chose not to scale. Every such failure would be indistinguishable from a finding.
#
# For each arm: bring it up (which tears down every other arm and resets to 1 replica),
# assert exactly one autoscaler is attached, offer load well above one pod's capacity,
# and require the replica count to rise. Then tear down.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=aiops
DEPLOY=target-app
ARMS=(hpa-cpu keda-rate predictive-prophet predictive-hybrid)

# 30 trace-minutes of the quiet segment = 3 wall-minutes at C=10. Its opening runs at
# ~97 req/min = ~16 req/s, which needs ~6 pods at the measured 3.135 req/s - far enough
# above one pod that a working arm cannot fail to react, and short enough to run four
# times without spending an hour.
SEG=quiet
MINUTES=30
SETTLE=25          # wall-seconds to let the arm react after load starts

pass=0; fail=0
declare -a SUMMARY

for arm in "${ARMS[@]}"; do
  echo
  echo "================================================================"
  echo "SMOKE: $arm"
  echo "================================================================"

  if ! "$REPO/experiments/switch_arm.sh" "$arm" >/tmp/smoke_switch.log 2>&1; then
    echo "  FAIL: switch_arm.sh could not bring up $arm"
    tail -15 /tmp/smoke_switch.log | sed 's/^/    /'
    SUMMARY+=("FAIL $arm  (switch_arm failed)"); ((fail++)); continue
  fi
  echo "  arm active: $("$REPO/experiments/switch_arm.sh" --status 2>/dev/null | sed -n '3p')"

  # Exactly one autoscaler attached, checked against the API server.
  if ! "$REPO/.venv/bin/python" "$REPO/experiments/verify_arms.py" --cluster \
       >/tmp/smoke_verify.log 2>&1; then
    echo "  FAIL: verify_arms.py --cluster did not pass with $arm active"
    grep -E '^\s+FAIL' /tmp/smoke_verify.log | sed 's/^/    /'
    SUMMARY+=("FAIL $arm  (parity/exclusivity)"); ((fail++)); continue
  fi
  echo "  parity + exclusivity: PASS"

  before=$(kubectl get deploy "$DEPLOY" -n "$NS" -o jsonpath='{.spec.replicas}')

  # Offer load in the background; watch the replica count while it runs.
  "$REPO/.venv/bin/python" "$REPO/load/trace_replay.py" --segment "$SEG" \
      --minutes "$MINUTES" --url http://localhost:30080/ --tag "smoke-$arm" \
      >/tmp/smoke_load.log 2>&1 &
  load_pid=$!

  peak=$before
  for _ in $(seq 1 $SETTLE); do
    r=$(kubectl get deploy "$DEPLOY" -n "$NS" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
    [[ "${r:-0}" -gt "$peak" ]] && peak=$r
    sleep 1
  done
  # Keep watching until the replay finishes so the peak is over the whole window.
  while kill -0 "$load_pid" 2>/dev/null; do
    r=$(kubectl get deploy "$DEPLOY" -n "$NS" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
    [[ "${r:-0}" -gt "$peak" ]] && peak=$r
    sleep 2
  done
  wait "$load_pid"; load_rc=$?

  if [[ $load_rc -ne 0 ]]; then
    echo "  FAIL: load generator exited $load_rc"
    tail -8 /tmp/smoke_load.log | sed 's/^/    /'
    SUMMARY+=("FAIL $arm  (load generator)"); ((fail++)); continue
  fi

  lat=$(grep -E '^\s+latency ms' /tmp/smoke_load.log | head -1)
  echo "  replicas: started $before, peaked $peak"
  echo " ${lat:-  (no latency line)}"

  if [[ "$peak" -gt "$before" ]]; then
    echo "  PASS: $arm scaled up under load"
    SUMMARY+=("PASS $arm  $before -> $peak replicas"); ((pass++))
  else
    echo "  FAIL: $arm never scaled above $before - it did not react to the load."
    echo "        Check: kubectl describe hpa -n $NS ; kubectl top pods -n $NS"
    SUMMARY+=("FAIL $arm  stuck at $before replicas"); ((fail++))
  fi
done

"$REPO/experiments/switch_arm.sh" none >/dev/null 2>&1

echo
echo "================================================================"
echo "SMOKE TEST SUMMARY"
echo "================================================================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "  ${pass} passed, ${fail} failed"
exit $(( fail > 0 ? 1 : 0 ))
