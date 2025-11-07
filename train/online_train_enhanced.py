#!/usr/bin/env python3
"""
online_train_enhanced.py
Trains an agent with a 10-feature state (phase, time, occupancy, speed)
and a balanced reward function.
"""

import os, json, time, random, math, sys, csv, numpy as np
import traci, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from dataclasses import dataclass
from typing import List, Tuple

# --- Add SUMO_HOME/tools to path ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
         sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

TLS_ID = "J0"
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"
MODELS_DIR = "/home/azureuser/traffic_rl/models"
LOGS_DIR = "/home/azureuser/traffic_rl/logs"

MODEL_PATH = f"{MODELS_DIR}/dqn_model_enhanced_latest.weights.h5"
BEST_MODEL_PATH = f"{MODELS_DIR}/dqn_model_enhanced_best.weights.h5"
SCALER_PATH = f"{MODELS_DIR}/scaler_enhanced.json" # Use the new scaler
LOG_CSV = f"{LOGS_DIR}/online_training_log_enhanced.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- State & Action ---
STATE_DIM = 10 # 10 features
N_ACTIONS = 2  # Phase 0 (E/W) or Phase 2 (N/S)

# --- Hyperparameters ---
LR = 5e-5
GAMMA = 0.90
BATCH_SIZE = 128
BUFFER_CAP = 200_000
WARMUP = 4000
TARGET_SYNC = 2500
MAX_EPISODES = 300
MAX_STEPS = 2000
SEED = 1337

# --- Epsilon (Exploration) ---
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY_FACTOR = 0.995 # Multiplicative decay per episode

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# --- Detector IDs (Must match .add.xml) ---
FLOW_DETECTORS = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]

# ============================================================================ #
# NORMALIZATION UTILITIES
# ============================================================================ #

def load_scaler(path):
    if not os.path.exists(path):
        print(f"❌ FATAL: Scaler file not found at {path}. Run generate_scaler.py first.")
        sys.exit(1) # Exit if scaler is missing
    with open(path, "r") as f:
        data = json.load(f)
    mean = np.array(data["mean"], dtype=np.float32)
    scale = np.array(data["scale"], dtype=np.float32)

    if len(mean) != STATE_DIM or len(scale) != STATE_DIM:
        print(f"❌ FATAL: Scaler dimensions ({len(mean)}) don't match STATE_DIM ({STATE_DIM}).")
        sys.exit(1)

    scale = np.where(scale == 0, 1e-6, scale) # Avoid division by zero
    print(f"[INFO] Scaler loaded from {path}")
    return mean, scale

mean, scale = load_scaler(SCALER_PATH)
def z_norm(x): 
    return (np.nan_to_num(x, nan=0.0) - mean) / scale

# ============================================================================ #
# REPLAY BUFFER & AGENT (Updated for 10-dim state)
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
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=float(d)
        self.idx=(self.idx+1)%self.cap
        if self.idx==0: self.full=True
    def sample(self, bs):
        size=self.cap if self.full else self.idx
        bs = min(bs, size)
        if bs == 0: return None
        idx=np.random.randint(0,size,bs)
        return (tf.constant(self.s[idx]), tf.constant(self.a[idx]),
                tf.constant(self.r[idx]), tf.constant(self.s2[idx]),
                tf.constant(self.d[idx]))
    def __len__(self): return self.cap if self.full else self.idx

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
    def predict_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        q_values = self.q_net(state.reshape(1, -1), training=False)[0].numpy()
        action = int(np.argmax(q_values))
        return action, q_values

# ============================================================================ #
# ENVIRONMENT INTERACTION (Using NEW 10-feature state)
# ============================================================================ #

def get_state():
    """Reads detector data and phase info, returns (norm_state, raw_state)."""
    try:
        current_phase = float(traci.trafficlight.getPhase(TLS_ID))
        # Use % 4 to map phases [0,1,2,3] to [0,1,2,3]
        current_phase = current_phase % 4 
        # Represent as [0] for N/S (phase 2,3) or [1] for E/W (phase 0,1)
        # This is a simpler binary representation for the agent
        phase_binary = 0.0 if current_phase in [2, 3] else 1.0 # 0=NS, 1=EW

        time_in_phase = float(traci.trafficlight.getPhaseDuration(TLS_ID) - \
                        (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime()))

        occupancy = [traci.inductionloop.getLastStepOccupancy(det_id) for det_id in QUEUE_DETECTORS]
        avg_speeds = [traci.inductionloop.getLastStepMeanSpeed(det_id) for det_id in FLOW_DETECTORS]
        avg_speeds = [s if s >= 0 else 0.0 for s in avg_speeds] # Handle -1 for no cars

        raw = np.array([
            phase_binary,       # 0: Current green direction (0=NS, 1=EW)
            time_in_phase,      # 1: Time elapsed in current phase
            occupancy[0],       # 2: N Approach Occupancy (%)
            occupancy[1],       # 3: S Approach Occupancy (%)
            occupancy[2],       # 4: E Approach Occupancy (%)
            occupancy[3],       # 5: W Approach Occupancy (%)
            avg_speeds[0],      # 6: N Approach Speed (m/s)
            avg_speeds[1],      # 7: S Approach Speed (m/s)
            avg_speeds[2],      # 8: E Approach Speed (m/s)
            avg_speeds[3],      # 9: W Approach Speed (m/s)
        ], dtype=np.float32)

        return z_norm(raw), raw # Return both normalized and raw

    except traci.TraCIException as e:
        print(f"[ERROR] TraCI error in get_state: {e}")
        default_raw = np.zeros(STATE_DIM, dtype=np.float32)
        return z_norm(default_raw), default_raw

def apply_action(action):
    """Maps agent action (0=EW, 1=NS) to SUMO phases (0, 2)."""
    try:
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        # Agent wants EW Green (action 0)
        if action == 0 and current_phase in [2, 3]: # If currently NS
            traci.trafficlight.setPhase(TLS_ID, 0)  # Switch to EW green
        # Agent wants NS Green (action 1)
        elif action == 1 and current_phase in [0, 1]: # If currently EW
            traci.trafficlight.setPhase(TLS_ID, 2)  # Switch to NS green
    except traci.TraCIException as e:
         print(f"[ERROR] TraCI error in apply_action: {e}")

def compute_reward(raw_state):
    """Balanced reward: Avg Speed minus Max Approach Occupancy."""
    # Indices from new raw state:
    # 2:N Occ, 3:S Occ, 4:E Occ, 5:W Occ
    # 6:N Spd, 7:S Spd, 8:E Spd, 9:W Spd

    all_speeds = raw_state[6:]
    all_occupancies = raw_state[2:6]

    avg_speed = np.mean(all_speeds)
    max_occupancy = np.max(all_occupancies) # Get the worst approach

    # Reward speed, heavily penalize the *worst* queue
    # Tune these weights!
    speed_reward = 0.3 * avg_speed
    queue_penalty = 0.7 * (max_occupancy / 100.0) # Normalize occ % to 0-1

    r = speed_reward - queue_penalty
    return np.clip(r, -10.0, 10.0) # Clip reward

# ============================================================================ #
# MAIN TRAINING LOOP
# ============================================================================ #

def main():
    print("🚦 Starting online DQN training (Enhanced State)...")
    agent = DQNAgent(STATE_DIM, N_ACTIONS, LR)
    buf = ReplayBuffer(BUFFER_CAP)

    # Try to resume from existing model
    if os.path.exists(MODEL_PATH):
         print(f"Resuming from previous model: {MODEL_PATH}")
         agent.q_net.load_weights(MODEL_PATH)
         agent.update_target_network()

    eps = EPS_START
    best_reward = -np.inf
    global_step = 0

    log_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["episode", "total_reward", "avg_loss", "epsilon", "steps"])

        for ep in range(MAX_EPISODES):
            try:
                traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])
                s_norm, s_raw = get_state()
                total_reward, losses = 0.0, []
                steps_taken = 0

                for step in range(MAX_STEPS):
                    if random.random() < eps:
                        a = random.randint(0, N_ACTIONS - 1)
                    else:
                        a, _ = agent.predict_action(s_norm)

                    apply_action(a)
                    traci.simulationStep() # Run simulation for 1 step
                    s2_norm, s2_raw = get_state()
                    r = compute_reward(s2_raw)

                    buf.push(s_norm, a, r, s2_norm, 0.0) # 0.0 for 'done'
                    s_norm = s2_norm
                    total_reward += r
                    steps_taken = step + 1

                    if len(buf) > WARMUP:
                        batch = buf.sample(BATCH_SIZE)
                        if batch:
                            loss = agent.train_step(*batch)
                            losses.append(float(loss))
                            if global_step % TARGET_SYNC == 0:
                                agent.update_target_network()

                    global_step += 1

                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break # Simulation ended early

            except traci.TraCIException as e:
                 print(f"\n[ERROR] TraCI fatal error in episode {ep}: {e}. Skipping.")
                 try: traci.close()
                 except: pass
                 continue
            except Exception as e:
                 print(f"\n[ERROR] Unexpected error in episode {ep}: {e}")
                 import traceback
                 traceback.print_exc()
                 try: traci.close()
                 except: pass
                 continue
            finally:
                try: traci.close()
                except: pass

            # --- Episode End ---
            eps = max(EPS_END, eps * EPS_DECAY_FACTOR) # Decay epsilon

            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Ep {ep:03d}/{MAX_EPISODES} | Reward={total_reward:.2f} | AvgLoss={avg_loss:.4f} | Steps={steps_taken} | eps={eps:.3f}")

            writer.writerow([ep, total_reward, avg_loss, eps, steps_taken])
            f.flush()

            agent.q_net.save_weights(MODEL_PATH) # Save latest
            if total_reward > best_reward:
                best_reward = total_reward
                agent.q_net.save_weights(BEST_MODEL_PATH)
                print(f"💾 New best model saved (Reward={best_reward:.2f})")

    print("\n✅ Training complete. Models saved.")

# --- Run Main ---
if __name__ == "__main__":
    main()
