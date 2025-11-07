#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Runner — paired 10-minute WALL-CLOCK episodes, all day (06:00–23:00)
- Launches BASELINE (embedded TraCI) and RL (port 8812) simultaneously
- Each episode ends by wall clock (WALL_DURATION_S=600)
- Archives KPIs via controllers' ONE_SHOT archive logic
- Calls paired_rollup.py after each pair to keep dashboard fresh
"""

import os
import sys
import time
import signal
import subprocess as sp
from datetime import datetime, time as dtime
from pathlib import Path

HOME = str(Path.home())
ROOT = f"{HOME}/traffic_rl"
VENV_PY = f"{ROOT}/.venv/bin/python"
LOG_DIR  = f"{ROOT}/logs"
SERV_DIR = f"{ROOT}/services"
SCRIPTS  = f"{ROOT}/scripts"

BASELINE_PY = f"{SERV_DIR}/controller_baseline.py"
RL_PY       = f"{SERV_DIR}/controller_bridge.py"
ROLLUP_PY   = f"{SCRIPTS}/paired_rollup.py"

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ---- Config: service hours (local time) --------------------------------------
START_HHMM = (6, 0)    # 06:00
END_HHMM   = (23, 0)   # 23:00

# Grace/health knobs
GRACE_KILL_SEC = 15     # after wall cap, how long to wait before SIGTERM/SIGKILL
EPISODE_GAP_SEC = 5     # gap between consecutive pairs

def within_service_window() -> bool:
    """Return True if local time is within the configured daily window."""
    now = datetime.now().time()
    start = dtime(*START_HHMM)
    end   = dtime(*END_HHMM)
    if start <= end:
        return start <= now <= end
    # Overnight windows (not used here, but safe)
    return now >= start or now <= end

def start_pair(common_seed: str | None = None) -> tuple[sp.Popen, sp.Popen]:
    """Start baseline and RL controllers with 10min wall-clock caps."""
    env_base = os.environ.copy()
    env_rl   = os.environ.copy()

    if common_seed is None:
        # Stable seed per hour (so paired sims are comparable but change across day)
        common_seed = str(int(time.time()) // 3600)

    env_base.update({
        "ONE_SHOT": "1",
        "ARCHIVE_PREFIX": "baseline",
        "SIM_DURATION": "0",          # disable sim cap
        "WALL_DURATION_S": "600",     # 10-minute WALL CLOCK cap
        "LOG_INTERVAL": "5.0",        # RL uses LOG_INTERVAL; baseline uses LOG_INTERVAL_SIM, but harmless here
        "LOG_INTERVAL_SIM": "5.0",
        "SUMO_SEED": common_seed,
    })

    env_rl.update({
        "ONE_SHOT": "1",
        "ARCHIVE_PREFIX": "rl",
        "SIM_DURATION": "0",          # disable sim cap
        "WALL_DURATION_S": "600",     # 10-minute WALL CLOCK cap
        "LOG_INTERVAL": "5.0",
        "SUMO_SEED": common_seed,
    })

    # Log to files (handy if run under systemd)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = open(f"{LOG_DIR}/dayrunner_baseline_{ts}.out", "a", buffering=1)
    base_err = open(f"{LOG_DIR}/dayrunner_baseline_{ts}.err", "a", buffering=1)
    rl_out   = open(f"{LOG_DIR}/dayrunner_rl_{ts}.out", "a", buffering=1)
    rl_err   = open(f"{LOG_DIR}/dayrunner_rl_{ts}.err", "a", buffering=1)

    print(f"[day] launching pair @ {ts} seed={common_seed}", flush=True)

    # launch baseline first
    p_base = sp.Popen([VENV_PY, BASELINE_PY], env=env_base, stdout=base_out, stderr=base_err)
    print("[day] baseline started — giving it 5 s head-start", flush=True)
    time.sleep(5)

    # then RL
    p_rl = sp.Popen([VENV_PY, RL_PY], env=env_rl, stdout=rl_out, stderr=rl_err)
    print("[day] rl started", flush=True)

    return p_base, p_rl

def wait_and_harvest(p_base: sp.Popen, p_rl: sp.Popen, wall_seconds: int = 600):
    """Wait ~wall_seconds; then ensure both procs are down (SIGTERM→SIGKILL)."""
    # Wait the nominal wall duration
    t_end = time.time() + wall_seconds
    while time.time() < t_end:
        # if both have already finished early, break
        if (p_base.poll() is not None) and (p_rl.poll() is not None):
            break
        time.sleep(1.0)

    # Give grace time for natural shutdown
    end_deadline = time.time() + GRACE_KILL_SEC
    while time.time() < end_deadline:
        if (p_base.poll() is not None) and (p_rl.poll() is not None):
            break
        time.sleep(0.5)

    # For any still running, send TERM then KILL
    for name, proc in (("baseline", p_base), ("rl", p_rl)):
        if proc.poll() is None:
            print(f"[day] {name}: sending SIGTERM", flush=True)
            try: proc.terminate()
            except Exception: pass

    end_deadline = time.time() + GRACE_KILL_SEC
    while time.time() < end_deadline:
        if (p_base.poll() is not None) and (p_rl.poll() is not None):
            break
        time.sleep(0.5)

    for name, proc in (("baseline", p_base), ("rl", p_rl)):
        if proc.poll() is None:
            print(f"[day] {name}: sending SIGKILL", flush=True)
            try: proc.kill()
            except Exception: pass

def run_rollup():
    """Run paired_rollup.py to update paired_summary.csv & paired_day.json."""
    try:
        sp.run([VENV_PY, ROLLUP_PY], check=False)
    except Exception as e:
        print(f"[day] rollup failed: {e}", flush=True)

def main_loop():
    print("[day] runner started", flush=True)
    while True:
        if not within_service_window():
            # Sleep until we enter the window again
            time.sleep(15)
            continue

        # Start a simultaneous pair
        seed = str(int(time.time()) // 3600)
        p_base, p_rl = start_pair(common_seed=seed)

        # Wait ~10 mins wall-clock and ensure clean stop
        wait_and_harvest(p_base, p_rl, wall_seconds=600)

        # Archive happens in each controller (ONE_SHOT). Roll up now.
        run_rollup()

        # Short gap between pairs
        time.sleep(EPISODE_GAP_SEC)

def _sigterm_handler(signum, frame):
    print("[day] SIGTERM received — exiting", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    try:
        main_loop()
    except KeyboardInterrupt:
        print("[day] KeyboardInterrupt — exiting", flush=True)
