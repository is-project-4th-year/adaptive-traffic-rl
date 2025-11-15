#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline SUMO Controller — STRICT 600s Paired Episode Mode
-----------------------------------------------------------
- Fixed-cycle benchmark controller
- Reads shared/pair_sync.json written by day_runner.py
- Waits until start_at_epoch (wall-clock)
- Runs SUMO at full speed (tiny sleep)
- Logs KPIs exactly every STEP_S seconds (wall-clock)
- Produces EXACTLY n_steps rows, aligned with RL controller

KPI schema:
    timestamp, avg_speed, avg_wait, queue_len, action
where:
    action = 1 if a phase switch happened since last tick, else 0
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

# ------------------------------------------------------------
# PATHS / CONSTANTS
# ------------------------------------------------------------
ROOT = str(Path.home()) + "/traffic_rl"

SYNC_PATH = os.environ.get(
    "PAIR_SYNC",
    f"{ROOT}/shared/pair_sync.json"
)

LOG_DIR = Path(os.environ.get("LOG_DIR", f"{ROOT}/logs"))
KPI_FILE = Path(os.environ.get("KPI_FILE", LOG_DIR / "kpi_baseline.csv"))

SUMO_BINARY = os.environ.get("SUMO_BINARY", "/usr/bin/sumo")
SUMO_CFG = os.environ.get(
    "SUMO_CFG",
    f"{ROOT}/junctions/uhuru_baseline/live.sumocfg",
)
SUMO_SEED = os.environ.get("SUMO_SEED", "123")
DEMAND_SCALE = os.environ.get("DEMAND_SCALE", "")
TLS_ID = os.environ.get("TLS_ID", "J0")
STEP_LENGTH = float(os.environ.get("STEP_LENGTH", "1.0"))
CYCLE_TIME = float(os.environ.get("CYCLE_TIME", "30.0"))  # fixed cycle length (sim seconds)

# Paired episode timing
EPISODE_LEN_S = 600      # must match day_runner
STEP_S        = 5        # will be overwritten by pair_sync
NEXT_TICK     = None     # function(k) -> wall_epoch
K             = 0        # tick index

RUNNING = True

# Fixed green phases (simple two-phase plan)
GREEN_PHASES = [0, 2]

# ------------------------------------------------------------
# SIGNAL HANDLER
# ------------------------------------------------------------
def _sig_handler(sig, frame):
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [baseline] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ------------------------------------------------------------
# SYNC SETUP
# ------------------------------------------------------------
def sync_setup():
    """
    Read pair_sync.json, block until start_at_epoch, and define NEXT_TICK(k).
    """
    global NEXT_TICK, STEP_S

    if not os.path.exists(SYNC_PATH):
        raise RuntimeError(f"pair_sync.json missing at {SYNC_PATH}")

    with open(SYNC_PATH) as f:
        sync = json.load(f)

    start_at = float(sync["start_at_epoch"])
    STEP_S   = int(sync["step_seconds"])

    # Hard wait until aligned start
    while True:
        dt = start_at - time.time()
        if dt <= 0.02:
            break
        time.sleep(min(dt, 0.05))

    def _tick(k: int) -> float:
        return start_at + k * STEP_S

    NEXT_TICK = _tick
    logging.info(
        f"[baseline-sync] start_at={start_at:.0f}, step_seconds={STEP_S}"
    )
    return start_at


# ------------------------------------------------------------
# KPI HEADER
# ------------------------------------------------------------
def ensure_kpi_header():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not KPI_FILE.exists() or KPI_FILE.stat().st_size < 20:
        with KPI_FILE.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "avg_speed", "avg_wait", "queue_len", "action"]
            )


# ------------------------------------------------------------
# EDGE DISCOVERY
# ------------------------------------------------------------
def discover_inbound_edges(tls_id: str):
    try:
        links = traci.trafficlight.getControlledLinks(tls_id)
        lanes = {l[0][0] for l in links if l and l[0]}
        edges = sorted({traci.lane.getEdgeID(l) for l in lanes})
        return edges
    except Exception as e:
        logging.warning(f"could not discover inbound edges: {e}")
        return []


# ------------------------------------------------------------
# MAIN EPISODE LOOP (STRICT 600s)
# ------------------------------------------------------------
# MAIN EPISODE LOOP (STRICT 600s)
# ------------------------------------------------------------
def run_episode() -> int:
    global K

    logging.info("=" * 72)
    logging.info("🚦 Baseline Controller — strict paired mode start")
    logging.info(f"CFG={SUMO_CFG}, SEED={SUMO_SEED}, TLS={TLS_ID}")
    logging.info("=" * 72)

    ensure_kpi_header()
    start_at = sync_setup()  # defines NEXT_TICK + STEP_S

    # Launch SUMO
    sumo_cmd = [
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--no-step-log",
        "--step-length", str(STEP_LENGTH),
        "--start", "true",
        "--quit-on-end", "true",
        "--seed", str(SUMO_SEED),
    ]
    if DEMAND_SCALE:
        sumo_cmd += ["--scale", str(DEMAND_SCALE)]

    logging.info("⚙️ launching SUMO for baseline…")
    traci.start(sumo_cmd)
    time.sleep(0.3)
    traci.simulationStep()
    logging.info("✅ connected to TraCI (baseline)")

    # Fixed-cycle init
    phase_idx = GREEN_PHASES[0]
    traci.trafficlight.setPhase(TLS_ID, phase_idx)
    next_switch_simt = CYCLE_TIME
    logging.info(f"🟢 initial phase={phase_idx}")

    # Controlled edges
    edges = discover_inbound_edges(TLS_ID)
    logging.info(f"inbound edges={edges}")

    # Strict tick plan
    n_steps = EPISODE_LEN_S // STEP_S
    steps_per_tick = max(1, int(round(STEP_S / STEP_LENGTH)))
    K = 0

    logging.info(
        f"[baseline] running {n_steps} ticks @ {STEP_S}s "
        f"({steps_per_tick} sim-steps per tick, step_length={STEP_LENGTH})"
    )

    # ----------------------------------------------------------------------
    # MAIN LOOP (STRICT WALL-CLOCK TIMING)
    # ----------------------------------------------------------------------
    try:
        for K in range(n_steps):
            if not RUNNING:
                break

            # 1. Advance SUMO simulation
            for _ in range(steps_per_tick):
                traci.simulationStep()

            simt = traci.simulation.getTime()

            # 2. Fixed-cycle switching logic
            if simt >= next_switch_simt:
                try:
                    idx = GREEN_PHASES.index(phase_idx)
                except ValueError:
                    idx = 0

                phase_idx = GREEN_PHASES[(idx + 1) % len(GREEN_PHASES)]
                traci.trafficlight.setPhase(TLS_ID, phase_idx)
                next_switch_simt += CYCLE_TIME
                switched_flag = 1
            else:
                switched_flag = 0

            # 3. KPI sampling
            total_speed = 0.0
            total_wait = 0.0
            total_queue = 0
            n = len(edges) or 1

            for e in edges:
                total_speed += traci.edge.getLastStepMeanSpeed(e)
                total_wait += traci.edge.getWaitingTime(e)
                total_queue += traci.edge.getLastStepHaltingNumber(e)

            avg_speed = total_speed / n
            avg_wait = total_wait / n

            # 4. WALL-CLOCK GATE (REAL 5-SECOND TICK)
            target = NEXT_TICK(K)
            while time.time() < target:
                time.sleep(0.05)

            # 5. Write KPI row
            ts = target
            with KPI_FILE.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [ts, avg_speed, avg_wait, total_queue, switched_flag]
                )

            logging.info(
                f"📊 tick={K} | wall={ts:.0f} | simt={simt:.1f} | "
                f"speed={avg_speed:.2f} | wait={avg_wait:.1f} | "
                f"queue={total_queue} | switched={switched_flag}"
            )

        logging.info(f"baseline episode done — ticks={K+1}/{n_steps}")
        traci.close(False)
        return 0

    except Exception as e:
        logging.error(f"💥 baseline fatal error: {e}", exc_info=True)
        try:
            traci.close(False)
        except Exception:
            pass
        return 1

# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    rc = run_episode()
    sys.exit(rc)
