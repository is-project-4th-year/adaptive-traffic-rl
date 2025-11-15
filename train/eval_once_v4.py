#!/usr/bin/env python3
import os, sys, json, time
import numpy as np
import traci
from tensorflow.keras import Sequential
import tensorflow as tf

TLS_ID = "J0"
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"
MODEL_BEST = "/home/azureuser/traffic_rl/models/dqn_model_v4_best.weights.h5"
STATE_DIM = 10

from online_train_v4_fixed_final import get_state, apply_action

def main():
    print("🚦 Eval run (full 2000 steps)...")

    # Load model
    model = Sequential([
        tf.keras.layers.Input(shape=(STATE_DIM,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="linear"),
    ])
    model.load_weights(MODEL_BEST)
    print("Loaded best model.")

    traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])

    s, raw = get_state()

    sum_speed = 0.0
    sum_queue = 0.0
    switches = 0
    teleports = 0
    prev_a = 1 if raw[0] == 0.0 else 0
    n = 0

    for t in range(2000):
        # action
        q = model(s.reshape(1, -1), training=False).numpy()[0]
        a = int(np.argmax(q))

        tp_before = traci.simulation.getStartingTeleportNumber()
        apply_action(a)
        traci.simulationStep()
        tp_after = traci.simulation.getStartingTeleportNumber()

        teleports += max(0, tp_after - tp_before)

        s, raw = get_state()
        sum_speed += float(np.mean(raw[6:]))
        sum_queue += float(np.max(raw[2:6]))

        if a != prev_a:
            switches += 1
        prev_a = a
        n += 1

        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    traci.close()

    print("\n===== FULL EVAL RESULTS =====")
    print(f"Avg Speed   : {sum_speed/n:.4f}")
    print(f"Avg Queue   : {sum_queue/n:.4f}")
    print(f"Switches    : {switches}")
    print(f"Teleports   : {teleports}")
    print("==============================")

if __name__ == "__main__":
    main()
