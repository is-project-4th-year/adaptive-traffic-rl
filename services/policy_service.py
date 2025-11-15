#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy Inference Service — DQN decides traffic light actions
Reads /shared/state.json (from controller)
→ runs DQN inference
→ writes /shared/action.json
Runs continuously under systemd.
"""

import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

STATE_FILE = Path("/home/azureuser/traffic_rl/shared/state.json")
ACTION_FILE = Path("/home/azureuser/traffic_rl/shared/action.json")
MODEL_PATH = Path("/home/azureuser/traffic_rl/models/dqn_model_v4_best.weights.h5")
POLL_INTERVAL = 3.0     # seconds between polls of state.json
INPUT_DIM = 10           # number of features in state.json
OUTPUT_DIM = 2           # number of discrete actions (hold/advance)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [policy-service] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# MODEL DEFINITION + LOADING
# ---------------------------------------------------------------------------

def build_dqn(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
    """Define the SAME architecture as training (v4 = 128-128-2)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    return model


def load_dqn():
    """Build the model and load weights only."""
    model = build_dqn()
    if MODEL_PATH.exists():
        model.load_weights(MODEL_PATH)
        logging.info(f"✅ Loaded DQN weights from {MODEL_PATH}")
    else:
        logging.error(f"❌ Weights file not found at {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# STATE → ACTION INFERENCE
# ---------------------------------------------------------------------------

def read_state():
    """Reads state.json and returns the v4 10-element raw state vector."""
    try:
        data = json.loads(STATE_FILE.read_text())

        # Phase normalization (v4 uses 0=NS,1=EW same as training)
        phase_raw = float(data.get("phase_binary", 0.0))
        if phase_raw in (2.0, 3.0):
            phase_binary = 0.0
        else:
            phase_binary = 1.0

        raw = np.array([
            phase_binary,                      # 0
            float(data.get("time_in_phase", 0)),
            float(data.get("N_queue", 0)),     # 2
            float(data.get("S_queue", 0)),     # 3
            float(data.get("E_queue", 0)),     # 4
            float(data.get("W_queue", 0)),     # 5
            float(data.get("N_speed", 0)),     # 6
            float(data.get("S_speed", 0)),     # 7
            float(data.get("E_speed", 0)),     # 8
            float(data.get("W_speed", 0)),     # 9
        ], dtype=np.float32)

        return raw

    except Exception as e:
        logging.warning(f"⚠️ Bad state.json: {e}")
        return None

def decide_action(model, state_vec):
    """Runs the model to choose the best action."""
    try:
        q_values = model.predict(state_vec[np.newaxis, :], verbose=0)[0]
        action_index = int(np.argmax(q_values))
        return action_index, q_values.tolist()
    except Exception as e:
        logging.error(f"❌ Inference error: {e}")
        return 0, []


def write_action(action_index, q_values):
    """Writes the chosen action and optional debug info to action.json."""
    try:
        with ACTION_FILE.open("w") as f:
            json.dump({
                "action_index": int(action_index),
                "q_values": q_values,
                "timestamp": time.time()
            }, f)
        logging.info(f"🟢 Action {action_index} written → {ACTION_FILE.name}")
    except Exception as e:
        logging.error(f"❌ Failed to write action.json: {e}")


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    model = load_dqn()
    logging.info("♻️  Policy-service online. Waiting for state updates...")

    last_action = None
    while True:
        state_vec = read_state()
        if state_vec is not None:
            action_idx, q_values = decide_action(model, state_vec)
        write_action(action_idx, q_values)
        last_action = action_idx
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

