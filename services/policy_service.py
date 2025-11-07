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
MODEL_PATH = Path("/home/azureuser/traffic_rl/models/dqn_model_v3_best.weights.h5")

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
    """Define the same architecture used during training."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
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
    """Reads the latest state.json file and returns feature vector."""
    try:
        with STATE_FILE.open("r") as f:
            data = json.load(f)

        # Maintain the same feature order used during training
        features = np.array([
            data.get("E_queue", 0),
            data.get("E_speed", 0),
            data.get("N_queue", 0),
            data.get("N_speed", 0),
            data.get("S_queue", 0),
            data.get("S_speed", 0),
            data.get("W_queue", 0),
            data.get("W_speed", 0),
            data.get("phase_binary", 0),
            data.get("time_in_phase", 0)
        ], dtype=float)

        return features
    except Exception as e:
        logging.warning(f"⚠️ Could not read state.json: {e}")
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
            if action_idx != last_action:
                write_action(action_idx, q_values)
                last_action = action_idx
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

