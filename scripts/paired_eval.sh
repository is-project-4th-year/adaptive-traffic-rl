#!/usr/bin/env bash
set -euo pipefail

EPISODES="${1:-12}"             # number of baseline/RL pairs to run
PY=~/traffic_rl/.venv/bin/python3
SERV=~/traffic_rl/services
LOG=~/traffic_rl/logs

echo "== Preparing services =="
# Stop controllers so nothing else is writing to SUMO/ports
sudo systemctl stop controller_rl controller_baseline || true
# RL needs policy-service running to produce action.json
sudo systemctl restart policy-service

mkdir -p "$LOG"

for i in $(seq 1 "$EPISODES"); do
  echo "==================== Pair $i: BASELINE (10-min sim) ===================="
  ONE_SHOT=1 ARCHIVE_PREFIX=baseline "$PY" "$SERV/controller_baseline.py"

  echo "==================== Pair $i: RL (10-min sim) ========================="
  ONE_SHOT=1 ARCHIVE_PREFIX=rl "$PY" "$SERV/controller_bridge.py"

  # optional: compute running pair summary after each pair
  "$PY" ~/traffic_rl/scripts/paired_rollup.py || true
done

echo "All pairs finished."
