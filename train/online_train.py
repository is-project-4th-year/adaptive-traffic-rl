#!/usr/bin/env python3
"""
online_train.py — Real-time DQN training for Uhuru Junction (J0)
Trains an agent online using induction loop detectors and aggressive reward shaping.
"""

import os, json, time, random, numpy as np, pandas as pd
import traci, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

TLS_ID = "J0"  # traffic light ID
MODEL_PATH = "/home/azureuser/traffic_rl/models/dqn_model_online_latest.weights.h5"
BEST_MODEL_PATH = "/home/azureuser/traffic_rl/models/dqn_model_online_best.weights.h5"
SCALER_PATH = "/home/azureuser/traffic_rl/models/scaler.json"
LOG_CSV = "/home/azureuser/traffic_rl/logs/online_training_log.csv"
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"

STATE_DIM = 8
N_ACTIONS = 2

# --- Hyperparameters ---
LR = 1e-4
GAMMA = 0.9
BATCH_SIZE = 128
BUFFER_CAP = 200_000
WARMUP = 4000
TARGET_SYNC = 2000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000
MAX_EPISODES = 300
MAX_STEPS = 2000
SEED = 1337

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ============================================================================ #
# NORMALIZATION UTILITIES
# ============================================================================ #

def load_scaler(path):
    if not os.path.exists(path):
        print(f"[WARN] No scaler.json found — using identity normalization.")
        return np.zeros(STATE_DIM), np.ones(STATE_DIM)
    with open(path, "r") as f:
        data = json.load(f)
    mean = np.array(data["mean"])
    scale = np.where(np.array(data["scale"]) == 0, 1e-6, data["scale"])
    print(f"[INFO] Scaler loaded from {path}")
    return mean, scale

mean, scale = load_scaler(SCALER_PATH)
def z_norm(x): return (x - mean) / scale

# ============================================================================ #
# REPLAY BUFFER
# ============================================================================ #

class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.s = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.a = np.zeros((cap,), dtype=np.int64)
        self.r = np.zeros((cap,), dtype=np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.d = np.zeros((cap,), dtype=np.float32)
        self.idx = 0; self.full = False
    def push(self, s,a,r,s2,d):
        i = self.idx
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.idx=(self.idx+1)%self.cap
        if self.idx==0: self.full=True
    def sample(self, bs):
        size=self.cap if self.full else self.idx
        idx=np.random.randint(0,size,bs)
        return (tf.constant(self.s[idx]), tf.constant(self.a[idx]),
                tf.constant(self.r[idx]), tf.constant(self.s2[idx]),
                tf.constant(self.d[idx]))
    def __len__(self): return self.cap if self.full else self.idx

# ============================================================================ #
# DQN AGENT
# ============================================================================ #

class DQNAgent:
    def __init__(self, state_dim, n_actions, lr):
        self.q_net = self._build_net(state_dim, n_actions)
        self.target_q_net = self._build_net(state_dim, n_actions)
        self.update_target_network()
        self.optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_fn = Huber()
    def _build_net(self, s_dim, n_act):
        return Sequential([
            Input(shape=(s_dim,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(n_act, activation='linear')
        ])
    def update_target_network(self):
        self.target_q_net.set_weights(self.q_net.get_weights())
    @tf.function
    def train_step(self, s,a,r,s2,d):
        q_next_main = self.q_net(s2, training=False)
        best_next = tf.argmax(q_next_main, axis=1, output_type=tf.int64)
        q_next_target = self.target_q_net(s2, training=False)
        idx = tf.stack([tf.range(tf.shape(best_next)[0], dtype=tf.int64), best_next], axis=1)
        q_target_val = tf.gather_nd(q_next_target, idx)
        y = r + (1.0 - d) * GAMMA * q_target_val
        with tf.GradientTape() as tape:
            q_all = self.q_net(s, training=True)
            idx2 = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int64), a], axis=1)
            q_pred = tf.gather_nd(q_all, idx2)
            loss = self.loss_fn(y, q_pred)
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

# ============================================================================ #
# ENVIRONMENT INTERACTION
# ============================================================================ #

def get_state():
    """Reads induction loop detector data and returns (norm_state, raw_state)."""
    det_ids = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]

    veh_counts = [traci.inductionloop.getLastStepVehicleNumber(i) for i in det_ids]
    speeds = [traci.inductionloop.getLastStepMeanSpeed(i) for i in det_ids]
    speeds = [s if s >= 0 else 0 for s in speeds]

    total_veh = sum(veh_counts)
    avg_speed = np.mean(speeds) if speeds else 0.0
    queue_proxy = total_veh * (1 - avg_speed / 13.9)
    delay = np.std(speeds)
    congestion_index = queue_proxy * 0.3 + delay * 0.2

    raw = np.array([
        total_veh, delay, avg_speed, queue_proxy,
        np.std(speeds), congestion_index,
        np.mean(speeds[:2]), np.mean(speeds[2:])
    ], dtype=np.float32)

    norm = z_norm(np.nan_to_num(raw, nan=0.0))
    return norm, raw

def apply_action(action):
    """Apply simplified 2-phase mapping for Uhuru Junction J0."""
    if action == 0:
        traci.trafficlight.setPhase(TLS_ID, 0)  # East–West green
    elif action == 1:
        traci.trafficlight.setPhase(TLS_ID, 2)  # North–South green

def compute_reward(raw_state):
    """Aggressive reward shaping: reward speed, penalize queue & delay."""
    speed = raw_state[2]
    wait = raw_state[1]
    q = raw_state[3]
    r = 0.5 * speed - 0.01 * wait - 0.01 * q
    return np.clip(r, -10.0, 10.0)

# ============================================================================ #
# MAIN TRAINING LOOP
# ============================================================================ #

def main():
    print("🚦 Starting online DQN training for Uhuru Junction...")
    agent = DQNAgent(STATE_DIM, N_ACTIONS, LR)
    buf = ReplayBuffer(BUFFER_CAP)

    eps = EPS_START
    best_reward = -np.inf
    global_step = 0

    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    with open(LOG_CSV, "w") as f:
        f.write("episode,total_reward,avg_loss,epsilon\n")

    for ep in range(MAX_EPISODES):
        traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log"])
        s_norm, s_raw = get_state()
        total_reward, losses = 0, []

        for step in range(MAX_STEPS):
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))
            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
            else:
                q = agent.q_net(np.expand_dims(s_norm, 0), training=False).numpy()[0]
                a = int(np.argmax(q))

            apply_action(a)
            traci.simulationStep()
            s2_norm, s2_raw = get_state()
            r = compute_reward(s2_raw)

            buf.push(s_norm, a, r, s2_norm, 0.0)
            s_norm = s2_norm
            total_reward += r

            if len(buf) > WARMUP:
                batch = buf.sample(BATCH_SIZE)
                loss = agent.train_step(*batch)
                losses.append(float(loss))
                if global_step % TARGET_SYNC == 0:
                    agent.update_target_network()
            global_step += 1

        traci.close()

        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Ep {ep:03d} | Reward={total_reward:.2f} | AvgLoss={avg_loss:.4f} | eps={eps:.3f}")
        with open(LOG_CSV, "a") as f:
            f.write(f"{ep},{total_reward:.3f},{avg_loss:.5f},{eps:.4f}\n")

        agent.q_net.save_weights(MODEL_PATH)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.q_net.save_weights(BEST_MODEL_PATH)
            print(f"💾 New best model saved (Reward={best_reward:.2f})")

    print("✅ Training complete. Models saved to /models/")

# ============================================================================ #
if __name__ == "__main__":
    main()
