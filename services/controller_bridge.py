#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Controller — STRICT 600s Paired Episode Mode (v4 state)
----------------------------------------------------------
- Reads pair_sync.json
- Waits until start_at_epoch
- Runs SUMO at full speed (no sleeping)
- Writes v4 10-dim state to state.json (atomic)
- Reads action.json from policy-service
- KPI logging EXACTLY every STEP_S seconds (wall-clock)
- EXACT SAME tick count as baseline
"""

import os
import sys
import time
import json
import csv
import logging
import signal
from pathlib import Path

import traci

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [rl] %(message)s"
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
SYNC_PATH = "/home/azureuser/traffic_rl/shared/pair_sync.json"
ACTION_FILE = Path("/home/azureuser/traffic_rl/shared/action.json")
STATE_FILE  = Path("/home/azureuser/traffic_rl/shared/state.json")

LOG_FILE = Path("/home/azureuser/traffic_rl/logs/kpi_live.csv")
LOG_DIR  = LOG_FILE.parent

TLS_ID = "J0"
STEP_LENGTH = 1.0     # SIM TIME step
DEMAND_SCALE = os.environ.get("DEMAND_SCALE", "")
RUNNING = True

NEXT_TICK = None      # function(k)->wall_epoch
STEP_S    = 5         # KPI interval
K         = 0         # tick counter

# v4 detector IDs (match training script)
FLOW_DETECTORS  = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]


# ------------------------------------------------------------
# SIGNAL HANDLER
# ------------------------------------------------------------
def handler(sig, frame):
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


# ------------------------------------------------------------
# SYNC SETUP
# ------------------------------------------------------------
def sync_setup():
    global NEXT_TICK, STEP_S

    if not os.path.exists(SYNC_PATH):
        raise RuntimeError("pair_sync.json missing — cannot run paired episode")

    with open(SYNC_PATH) as f:
        sync = json.load(f)

    start_at = float(sync["start_at_epoch"])
    STEP_S   = int(sync["step_seconds"])

    # Hard wait until synced start
    while True:
        dt = start_at - time.time()
        if dt <= 0.02:
            break
        time.sleep(min(dt, 0.05))

    def _tick(k):
        return start_at + k * STEP_S

    NEXT_TICK = _tick
    logging.info(f"[rl-sync] start={start_at}, step={STEP_S}")

    return start_at


# ------------------------------------------------------------
# HEADER ENSURE
# ------------------------------------------------------------
def ensure_kpi_header():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size < 20:
        with LOG_FILE.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "avg_speed", "avg_wait", "queue_len", "action"]
            )


# ------------------------------------------------------------
# ACTION READ
# ------------------------------------------------------------
def read_action():
    """Read action_index from action.json (default 0 on any error)."""
    try:
        with ACTION_FILE.open() as f:
            idx = int(json.load(f).get("action_index", 0))
        return 1 if idx == 1 else 0
    except Exception as e:
        logging.warning(f"[rl] could not read action.json, defaulting to 0: {e}")
        return 0


# ------------------------------------------------------------
# EDGE DISCOVERY (for KPI only)
# ------------------------------------------------------------
def discover_edges():
    try:
        links = traci.trafficlight.getControlledLinks(TLS_ID)
        lanes = {l[0][0] for l in links if l and l[0]}
        return sorted({traci.lane.getEdgeID(l) for l in lanes})
    except Exception as e:
        logging.warning(f"[rl] discover_edges failed: {e}")
        return []


# ------------------------------------------------------------
# v4 STATE BUILD + WRITE
# ------------------------------------------------------------
# ------------------------------------------------------------
# v4 STATE BUILD + WRITE (fixed, clean, no syntax issues)
# ------------------------------------------------------------
def build_state_v4():
    """
    Build v4 10-dim state:

    [phase_binary, time_in_phase,
     N_queue, S_queue, E_queue, W_queue,
     N_speed, S_speed, E_speed, W_speed]
    """

    try:
        # Phase: 1 = EW green, 0 = NS green
        phase_idx = traci.trafficlight.getPhase(TLS_ID)
        phase_binary = 1.0 if phase_idx in [0, 1] else 0.0

        # Time in current phase
        t_now = traci.simulation.getTime()
        next_switch = traci.trafficlight.getNextSwitch(TLS_ID)
        duration = traci.trafficlight.getPhaseDuration(TLS_ID)
        time_in_phase = max(0.0, duration - (next_switch - t_now))

        # Queues
        qN = float(traci.inductionloop.getLastStepVehicleNumber("det_N_queue"))
        qS = float(traci.inductionloop.getLastStepVehicleNumber("det_S_queue"))
        qE = float(traci.inductionloop.getLastStepVehicleNumber("det_E_queue"))
        qW = float(traci.inductionloop.getLastStepVehicleNumber("det_W_queue"))

        # Speeds
        vN = max(0.0, float(traci.inductionloop.getLastStepMeanSpeed("det_N_in")))
        vS = max(0.0, float(traci.inductionloop.getLastStepMeanSpeed("det_S_in")))
        vE = max(0.0, float(traci.inductionloop.getLastStepMeanSpeed("det_E_in")))
        vW = max(0.0, float(traci.inductionloop.getLastStepMeanSpeed("det_W_in")))

        return {
            "schema": "v4",
            "phase_binary": phase_binary,
            "time_in_phase": time_in_phase,
            "N_queue": qN,
            "S_queue": qS,
            "E_queue": qE,
            "W_queue": qW,
            "N_speed": vN,
            "S_speed": vS,
            "E_speed": vE,
            "W_speed": vW,
            "timestamp": time.time(),
        }

    except Exception as e:
        logging.warning(f"[rl] build_state_v4 failed: {e}")

        return {
            "schema": "v4",
            "phase_binary": 0.0,
            "time_in_phase": 0.0,
            "N_queue": 0.0,
            "S_queue": 0.0,
            "E_queue": 0.0,
            "W_queue": 0.0,
            "N_speed": 0.0,
            "S_speed": 0.0,
            "E_speed": 0.0,
            "W_speed": 0.0,
            "timestamp": time.time(),
        }


def write_state_atomic(payload: dict):
    """Atomic write of state.json to avoid truncated/bad JSON."""
    try:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(STATE_FILE)
    except Exception as e:
        logging.warning(f"[rl] failed to write state.json atomically: {e}")


# ------------------------------------------------------------
# APPLY ACTION (still simple: 0=hold, 1=advance)
# ------------------------------------------------------------
MIN_GREEN = 8.0   # seconds

def apply_action(idx):
    try:
        # get time in phase
        t_now = traci.simulation.getTime()
        next_switch = traci.trafficlight.getNextSwitch(TLS_ID)
        duration = traci.trafficlight.getPhaseDuration(TLS_ID)
        time_in_phase = max(0.0, duration - (next_switch - t_now))

        # prevent switch spam
        if time_in_phase < MIN_GREEN:
            return 0  # force hold

        if idx == 1:
            cur = traci.trafficlight.getPhase(TLS_ID)
            defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(TLS_ID)
            phases = len(defs[0].phases)
            nxt = (cur + 1) % phases
            traci.trafficlight.setPhase(TLS_ID, nxt)
            return 1

        return 0
    except Exception:
        return 0

# ------------------------------------------------------------
# MAIN LOOP (STRICT 600s, v4 state writer)
# ------------------------------------------------------------
# ------------------------------------------------------------
def main():
    global K

    logging.info("🔥 RL Controller (strict 600s mode, v4 state) starting…")
    ensure_kpi_header()

    # Sync with baseline (defines NEXT_TICK + STEP_S)
    start_at = sync_setup()

    # Launch SUMO
    SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru_rl/live.sumocfg"
    sumo_cmd = [
        "sumo",
        "-c", SUMO_CFG,
        "--num-clients", "1",
        "--step-length", f"{STEP_LENGTH}",
        "--start", "true",
        "--quit-on-end", "true",
        "--no-step-log",
    ]

    # Optional demand scaler (0.1–1.0)
    if DEMAND_SCALE:
        sumo_cmd += ["--scale", str(DEMAND_SCALE)]

    traci.start(sumo_cmd, port=8812)

    # Initial step so detectors/TLS state are valid
    time.sleep(0.3)
    traci.simulationStep()

    edges = discover_edges()
    logging.info(f"[rl] inbound={edges}")

    # strict fixed tick count
    EPISODE_LEN_S = 600  # must match baseline/day_runner
    n_steps = EPISODE_LEN_S // STEP_S
    steps_per_tick = max(1, int(round(STEP_S / STEP_LENGTH)))
    K = 0

    logging.info(
        f"[rl] Running {n_steps} ticks @ {STEP_S}s "
        f"({steps_per_tick} sim-steps per tick, step_length={STEP_LENGTH})"
    )

    # ------- MAIN LOOP (FAST SUMO, KPI + STATE PER TICK) -------
    try:
        for K in range(n_steps):
            if not RUNNING:
                break

            # 🕒 Wait until this tick's wall-clock time (real 5s cadence)
            target_ts = NEXT_TICK(K)
            while True:
                now = time.time()
                dt = target_ts - now
                if dt <= 0:
                    break
                time.sleep(min(0.05, dt))

            # Advance SUMO by STEP_S sim seconds
            for _ in range(steps_per_tick):
                traci.simulationStep()

            sim_t    = traci.simulation.getTime()
            veh_left = traci.simulation.getMinExpectedNumber()
            # Optional debug:
            # print(f"[rl-debug] tick={K} sim_t={sim_t} veh_left={veh_left}")

            # 1) Build + write v4 state for policy-service
            state_payload = build_state_v4()
            write_state_atomic(state_payload)

            # 2) Read action decided by policy-service
            action   = read_action()
            switched = apply_action(action)

            # 3) KPIs (edge-level)
            total_speed = 0.0
            total_wait  = 0.0
            total_queue = 0.0
            n_edges = len(edges) or 1

            for e in edges:
                total_speed += traci.edge.getLastStepMeanSpeed(e)
                total_wait  += traci.edge.getWaitingTime(e)
                total_queue += traci.edge.getLastStepHaltingNumber(e)

            avg_speed = total_speed / n_edges
            avg_wait  = total_wait  / n_edges

            # Wall-clock timestamp aligned with baseline via NEXT_TICK
            ts = NEXT_TICK(K)

            with LOG_FILE.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [ts, avg_speed, avg_wait, total_queue, switched]
                )

            logging.info(
                f"📊 tick={K} | wall={ts:.0f} | "
                f"simt={sim_t:.1f} | speed={avg_speed:.2f} "
                f"wait={avg_wait:.1f} queue={total_queue} action={switched}"
            )

            # tiny sleep just to avoid pegging CPU
            time.sleep(0.001)

        logging.info(f"RL DONE (600s strict run) — ticks={K+1}/{n_steps}")
        traci.close(False)
        return 0

    except Exception as e:
        logging.error(f"💥 RL fatal error: {e}", exc_info=True)
        try:
            traci.close(False)
        except Exception:
            pass
        return 1


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
