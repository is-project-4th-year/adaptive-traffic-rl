#!/usr/bin/env python3
"""
generate_scaler_v3.py
Generates a scaler file using VEHICLE COUNT instead of occupancy.
"""
import os, json, random, sys, numpy as np
import traci

# --- Add SUMO_HOME/tools to path ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# --- Config ---
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"
OUT_SCALER = "/home/azureuser/traffic_rl/models/scaler_v3.json" # New scaler name
TLS_ID = "J0"
STATE_DIM = 10
STEPS_TO_RUN = 5000

FLOW_DETECTORS = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]

def get_raw_state():
    try:
        phase_binary = 0.0 if traci.trafficlight.getPhase(TLS_ID) in [2, 3] else 1.0 # 0=NS, 1=EW
        time_in_phase = float(traci.trafficlight.getPhaseDuration(TLS_ID) - \
                        (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime()))

        # *** THE KEY CHANGE IS HERE ***
        # Use vehicle count, not occupancy
        queue_lengths = [traci.inductionloop.getLastStepVehicleNumber(det_id) for det_id in QUEUE_DETECTORS]

        avg_speeds = [traci.inductionloop.getLastStepMeanSpeed(det_id) for det_id in FLOW_DETECTORS]
        avg_speeds = [s if s >= 0 else 0.0 for s in avg_speeds]

        raw = np.array([
            phase_binary, time_in_phase,
            queue_lengths[0], queue_lengths[1], queue_lengths[2], queue_lengths[3], # N, S, E, W Queue
            avg_speeds[0], avg_speeds[1], avg_speeds[2], avg_speeds[3] # N, S, E, W Speed
        ], dtype=np.float32)

        return np.nan_to_num(raw, nan=0.0)

    except traci.TraCIException:
        return np.zeros(STATE_DIM, dtype=np.float32)

def apply_random_action():
    action = random.choice([0, 2]) # Phase 0 or 2
    try:
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        if action == 0 and current_phase in [2, 3]:
            traci.trafficlight.setPhase(TLS_ID, 0)
        elif action == 2 and current_phase in [0, 1]:
            traci.trafficlight.setPhase(TLS_ID, 2)
    except traci.TraCIException:
        pass

# --- Main ---
def generate_data():
    print(f"--- Starting SUMO for {STEPS_TO_RUN} steps (v3 scaler) ---")
    traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])

    state_data = []
    for step in range(STEPS_TO_RUN):
        apply_random_action()
        traci.simulationStep()
        state_data.append(get_raw_state())
        if (step + 1) % 500 == 0: print(f"Step {step+1}/{STEPS_TO_RUN}...")

    traci.close()
    print("--- SUMO simulation finished ---")
    return np.array(state_data)

# --- Run and Calculate ---
try:
    data = generate_data()
    if len(data) > 0:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        scaler_data = {"mean": mean.tolist(), "scale": std.tolist()}

        with open(OUT_SCALER, "w") as f:
            json.dump(scaler_data, f, indent=4)

        print(f"\n✅ Saved new v3 scaler to {OUT_SCALER}")
        print(f"Mean: {np.round(mean, 3).tolist()}")
        print(f"Std:  {np.round(std, 3).tolist()}")
    else:
        print("❌ No data was generated.")

except Exception as e:
    print(f"\n--- An unexpected error occurred: {e} ---")
finally:
    try: traci.close()
    except: pass

if __name__ == "__main__":
    generate_data()
