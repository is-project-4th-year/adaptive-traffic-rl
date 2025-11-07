#!/usr/bin/env python3
"""
generate_scaler.py
Runs a random agent in SUMO for 5000 steps to collect state data
and computes the mean/std for the new 10-feature state.
Saves the result to 'scaler_enhanced.json'.
"""
import os, json, random, sys, numpy as np

# --- Add SUMO_HOME/tools to path ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci

# --- Config ---
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"
OUT_SCALER = "/home/azureuser/traffic_rl/models/scaler_enhanced.json"
TLS_ID = "J0"
STATE_DIM = 10
STEPS_TO_RUN = 5000

# --- Detector IDs (Must match .add.xml) ---
FLOW_DETECTORS = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]

def get_raw_state():
    """Gets the 10-feature raw state vector from SUMO."""
    try:
        current_phase = float(traci.trafficlight.getPhase(TLS_ID))
        time_in_phase = float(traci.trafficlight.getPhaseDuration(TLS_ID) - \
                        (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime()))

        occupancy = [traci.inductionloop.getLastStepOccupancy(det_id) for det_id in QUEUE_DETECTORS]
        avg_speeds = [traci.inductionloop.getLastStepMeanSpeed(det_id) for det_id in FLOW_DETECTORS]
        avg_speeds = [s if s >= 0 else 0.0 for s in avg_speeds]

        raw = np.array([
            current_phase, time_in_phase,
            occupancy[0], occupancy[1], occupancy[2], occupancy[3], # N, S, E, W Occupancy
            avg_speeds[0], avg_speeds[1], avg_speeds[2], avg_speeds[3] # N, S, E, W Speed
        ], dtype=np.float32)

        return np.nan_to_num(raw, nan=0.0)

    except traci.TraCIException as e:
        print(f"[ERROR] TraCI error in get_raw_state: {e}")
        return np.zeros(STATE_DIM, dtype=np.float32)

def apply_random_action():
    """Applies a random action (0 or 2) to the traffic light."""
    action = random.choice([0, 2]) # Corresponds to Phase 0 and Phase 2
    try:
        # Only switch if not already in the target phase or its yellow
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        if action == 0 and current_phase in [2, 3]:
            traci.trafficlight.setPhase(TLS_ID, 0)
        elif action == 2 and current_phase in [0, 1]:
            traci.trafficlight.setPhase(TLS_ID, 2)
    except traci.TraCIException as e:
        print(f"[ERROR] TraCI error in apply_random_action: {e}")

# --- Main ---
def generate_data():
    print(f"--- Starting SUMO for {STEPS_TO_RUN} steps to generate scaler data ---")
    traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])

    state_data = []
    for step in range(STEPS_TO_RUN):
        apply_random_action()
        traci.simulationStep()
        state = get_raw_state()
        state_data.append(state)

        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{STEPS_TO_RUN}...")

    traci.close()
    print("--- SUMO simulation finished ---")
    return np.array(state_data)

# --- Run and Calculate ---
try:
    data = generate_data()

    if len(data) > 0:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # Save the scaler
        scaler_data = {
            "mean": mean.tolist(),
            "scale": std.tolist()
        }

        with open(OUT_SCALER, "w") as f:
            json.dump(scaler_data, f, indent=4)

        print(f"\n✅ Successfully generated and saved new scaler to {OUT_SCALER}")
        print(f"Mean: {np.round(mean, 3).tolist()}")
        print(f"Std:  {np.round(std, 3).tolist()}")
    else:
        print("❌ No data was generated. Scaler file not created.")

except Exception as e:
    print(f"\n--- An unexpected error occurred ---")
    import traceback
    traceback.print_exc()
finally:
    try: traci.close()
    except: pass
