#!/usr/bin/env python3
"""
collector.py (v8) — TraCI-driven state generator for DQN
Stable 2-client architecture (controller + collector)
------------------------------------------------------
- Controller: client order 1 (steps SUMO)
- Collector:  client order 2 (reads state only)
- Policy-service: reads state.json, writes action.json

This module attaches to SUMO (already started by controller_bridge.py)
and continuously queries simulation state without calling simulationStep().
"""

import os, sys, time, json, logging

# --- SUMO setup --------------------------------------------------------------
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    print("FATAL: Please declare environment variable 'SUMO_HOME'")
    sys.exit(1)
tools = os.path.join(SUMO_HOME, "tools")
if tools not in sys.path:
    sys.path.append(tools)

import traci

# --- CONFIG ------------------------------------------------------------------
TLS_ID = "J0"
SUMO_PORT = 8813
STATE_PATH = os.path.expanduser("~/traffic_rl/shared/state.json")

FLOW_DETECTORS  = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]

# --- LOGGING -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INFO] [collector] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- STATE GENERATION --------------------------------------------------------
def get_state_dict():
    """
    Builds a v3 state vector with 10 continuous features:
      - Per-direction queue lengths and average speeds
      - Phase binary (0 = EW green, 1 = NS green)
      - Time in current phase
    """
    try:
        phase = traci.trafficlight.getPhase(TLS_ID)
        phase_binary = 0.0 if phase in (2, 3) else 1.0

        now = traci.simulation.getTime()
        time_to_next = traci.trafficlight.getNextSwitch(TLS_ID) - now
        dur = traci.trafficlight.getPhaseDuration(TLS_ID)
        time_in_phase = float(dur - time_to_next)

        # Detector data
        qN, qS, qE, qW = [traci.inductionloop.getLastStepVehicleNumber(d) for d in QUEUE_DETECTORS]
        sN, sS, sE, sW = [max(0.0, traci.inductionloop.getLastStepMeanSpeed(d)) for d in FLOW_DETECTORS]

        return {
            "E_queue": float(qE), "E_speed": float(sE),
            "N_queue": float(qN), "N_speed": float(sN),
            "S_queue": float(qS), "S_speed": float(sS),
            "W_queue": float(qW), "W_speed": float(sW),
            "phase_binary": float(phase_binary),
            "time_in_phase": float(time_in_phase),
            "schema": "v3"
        }
    except Exception as e:
        logging.error(f"get_state_dict() error: {e}")
        return {}

# --- MAIN LOOP ---------------------------------------------------------------
def main():
    logging.info("Collector v8 — waiting for SUMO on port %d...", SUMO_PORT)

    while True:
        try:
            # Connect to the running SUMO instance started by controller
            traci.connect(port=SUMO_PORT)
            time.sleep(1.0)
            traci.setOrder(2)
            logging.info("✅ Connected to SUMO (order=2).")

            while True:
                # No simulationStep() here — controller drives the clock
                state = get_state_dict()
                if state:
                    with open(STATE_PATH, "w") as f:
                        json.dump(state, f)
                time.sleep(1.0)  # small delay to reduce CPU use

        except traci.FatalTraCIError as e:
            logging.error(f"TraCI connection lost: {e}. Reconnecting in 5s...")
            try: traci.close()
            except Exception: pass
            time.sleep(5)

        except Exception as e:
            logging.error(f"Unexpected error: {e}. Reconnecting in 5s...", exc_info=True)
            try: traci.close()
            except Exception: pass
            time.sleep(5)

# --- ENTRY POINT -------------------------------------------------------------
if __name__ == "__main__":
    main()
