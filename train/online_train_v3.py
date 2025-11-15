#!/usr/bin/env python3
"""
online_train_v3.py
PRODUCTION-READY: Occupancy + Balanced Quadratic + Safe Phase Transitions

Key improvements:
- Normalized occupancy (0-1) with balanced weights
- Yellow phases + min green time to prevent oscillation
- Decision interval to reduce thrashing
- Proper target network sync
- Q-value monitoring
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

MODEL_PATH = f"{MODELS_DIR}/dqn_model_v3_latest.weights.h5"
BEST_MODEL_PATH = f"{MODELS_DIR}/dqn_model_v3_best.weights.h5"
SCALER_PATH = f"{MODELS_DIR}/scaler_enhanced.json"  # Occupancy-based scaler
LOG_CSV = f"{LOGS_DIR}/online_training_log_v3.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- State & Action ---
STATE_DIM = 10
N_ACTIONS = 2

# --- Hyperparameters ---
LR = 5e-5
GAMMA = 0.90
BATCH_SIZE = 128
BUFFER_CAP = 200_000
WARMUP = 4000
TARGET_SYNC = 1000  # Sync every 1000 steps (multiple times per episode)
MAX_EPISODES = 300
MAX_STEPS = 2000
SEED = 1337

# --- Epsilon (Exploration) ---
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY_FACTOR = 0.995

# --- Reward Tuning (Balanced on normalized occupancy) ---
SPEED_WEIGHT = 0.6
MAX_OCC_WEIGHT = 6.0      # Quadratic on normalized occupancy (0-1)
MEAN_OCC_WEIGHT = 1.0

# --- Traffic Control Parameters (Prevent oscillation) ---
DECISION_INTERVAL_STEPS = 5   # Make decisions every 5 simulation steps
MIN_GREEN_STEPS = 8           # Minimum green time (~8 seconds if step=1s)

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

FLOW_DETECTORS = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]

# --- Global state for phase management ---
_last_action_step = 0
_last_switch_step = -10**9

# ============================================================================ #
# NORMALIZATION UTILITIES
# ============================================================================ #

def load_scaler(path):
    """Load normalization parameters from JSON file."""
    if not os.path.exists(path):
        print(f"❌ FATAL: Scaler file not found at {path}")
        print("   This scaler must be built for occupancy-based state!")
        print("   Run generate_scaler.py with occupancy features first.")
        sys.exit(1)
    
    with open(path, "r") as f:
        data = json.load(f)
    
    mean = np.array(data["mean"], dtype=np.float32)
    scale = np.array(data["scale"], dtype=np.float32)
    
    if len(mean) != STATE_DIM or len(scale) != STATE_DIM:
        print(f"❌ FATAL: Scaler dimensions ({len(mean)}) don't match STATE_DIM ({STATE_DIM}).")
        sys.exit(1)
    
    scale = np.where(scale == 0, 1e-6, scale)
    print(f"[INFO] Scaler loaded from {path}")
    print(f"[INFO] State: [phase, time, 4×occupancy%, 4×speed]")
    return mean, scale

mean, scale = load_scaler(SCALER_PATH)

def z_norm(x):
    """Z-score normalization using pre-computed mean and scale."""
    return (np.nan_to_num(x, nan=0.0) - mean) / scale

# ============================================================================ #
# REPLAY BUFFER & AGENT
# ============================================================================ #

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: bool

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    def __init__(self, cap):
        self.cap = cap
        self.s = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.a = np.zeros((cap,), dtype=np.int64)
        self.r = np.zeros((cap,), dtype=np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.d = np.zeros((cap,), dtype=np.float32)
        self.idx = 0
        self.full = False
    
    def push(self, s, a, r, s2, d):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(d)
        self.idx = (self.idx + 1) % self.cap
        if self.idx == 0:
            self.full = True
    
    def sample(self, bs):
        size = self.cap if self.full else self.idx
        bs = min(bs, size)
        if bs == 0:
            return None
        idx = np.random.randint(0, size, bs)
        return (
            tf.constant(self.s[idx]),
            tf.constant(self.a[idx]),
            tf.constant(self.r[idx]),
            tf.constant(self.s2[idx]),
            tf.constant(self.d[idx])
        )
    
    def __len__(self):
        return self.cap if self.full else self.idx

class DQNAgent:
    """Deep Q-Network agent with Double DQN."""
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
    def train_step(self, s, a, r, s2, d):
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
# ENVIRONMENT INTERACTION - OCCUPANCY-BASED
# ============================================================================ #

def get_state():
    """
    Reads detector data and phase info, returns (norm_state, raw_state).
    
    State features (10 total):
    - Phase binary (0=NS, 1=EW)
    - Time in current phase
    - Occupancy % for N, S, E, W (0-100)
    - Average speeds for N, S, E, W (m/s)
    """
    try:
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        phase_binary = 0.0 if current_phase in [2, 3] else 1.0
        
        time_in_phase = float(
            traci.trafficlight.getPhaseDuration(TLS_ID) - 
            (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime())
        )
        
        # OCCUPANCY (%) - Superior metric
        occupancy = [
            traci.inductionloop.getLastStepOccupancy(det_id) 
            for det_id in QUEUE_DETECTORS
        ]
        
        # Average speeds (m/s)
        avg_speeds = [
            traci.inductionloop.getLastStepMeanSpeed(det_id) 
            for det_id in FLOW_DETECTORS
        ]
        avg_speeds = [s if s >= 0 else 0.0 for s in avg_speeds]
        
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
        
        return z_norm(raw), raw
    
    except traci.TraCIException as e:
        print(f"[ERROR] TraCI error in get_state: {e}")
        default_raw = np.zeros(STATE_DIM, dtype=np.float32)
        return z_norm(default_raw), default_raw

def apply_action(action):
    """
    Applies action with SAFE phase transitions:
    - Yellow phases (3 steps) before switching
    - Minimum green time enforcement
    - Prevents oscillation and reduces teleports
    
    Action 0 = E/W green (SUMO phase 0)
    Action 1 = N/S green (SUMO phase 2)
    """
    global _last_switch_step
    
    try:
        cur = traci.trafficlight.getPhase(TLS_ID)
        sim_t = int(traci.simulation.getTime())
        
        # Enforce minimum green time
        if sim_t - _last_switch_step < MIN_GREEN_STEPS:
            return
        
        want = 0 if action == 0 else 2  # 0=EW green, 2=NS green
        
        # EW -> NS with yellow transition
        if cur in [0, 1] and want == 2:
            traci.trafficlight.setPhase(TLS_ID, 1)  # Yellow for EW
            for _ in range(3):
                traci.simulationStep()
            traci.trafficlight.setPhase(TLS_ID, 2)  # Green for NS
            _last_switch_step = int(traci.simulation.getTime())
        
        # NS -> EW with yellow transition
        elif cur in [2, 3] and want == 0:
            traci.trafficlight.setPhase(TLS_ID, 3)  # Yellow for NS
            for _ in range(3):
                traci.simulationStep()
            traci.trafficlight.setPhase(TLS_ID, 0)  # Green for EW
            _last_switch_step = int(traci.simulation.getTime())
    
    except traci.TraCIException as e:
        print(f"[ERROR] TraCI error in apply_action: {e}")

def compute_reward(raw_state):
    """
    BALANCED REWARD on normalized occupancy (0-1).
    
    This fixes the weight imbalance from the previous version:
    - Occupancy normalized to 0-1 (divide by 100)
    - Weights tuned so good cases get +2 to +6
    - Bad cases get -2 to -8 (not -20!)
    
    Examples:
    - Excellent: speed=12, max_occ=25%, mean_occ=18%
      → 7.2 - 0.375 - 0.18 = +6.65 ✅
    
    - Good: speed=10, max_occ=40%, mean_occ=30%
      → 6.0 - 0.96 - 0.30 = +4.74 ✅
    
    - Mediocre: speed=6, max_occ=60%, mean_occ=50%
      → 3.6 - 2.16 - 0.50 = +0.94 ✅
    
    - Bad: speed=3, max_occ=80%, mean_occ=65%
      → 1.8 - 3.84 - 0.65 = -2.69 ❌
    
    - Severe: speed=2, max_occ=95%, mean_occ=80%
      → 1.2 - 5.42 - 0.80 = -5.02 ❌❌
    """
    # Normalize occupancy to 0-1
    occ = raw_state[2:6] / 100.0
    spd = raw_state[6:]
    
    avg_speed = float(np.mean(spd))
    max_occ = float(np.max(occ))
    mean_occ = float(np.mean(occ))
    
    # Balanced weights (readable on 0-1 scale)
    reward = (
        SPEED_WEIGHT * avg_speed - 
        MAX_OCC_WEIGHT * (max_occ ** 2) - 
        MEAN_OCC_WEIGHT * mean_occ
    )
    
    return float(np.clip(reward, -20.0, 10.0))

# ============================================================================ #
# MAIN TRAINING LOOP
# ============================================================================ #

def main():
    print("🚦 Starting PRODUCTION-READY DQN training...")
    print("   State: OCCUPANCY (%) - Accurate congestion measurement")
    print("   Reward: BALANCED QUADRATIC - Good cases +2..+6, bad cases -2..-8")
    print("   Safety: Yellow phases + Min green + Decision intervals")
    print(f"   Weights: Speed={SPEED_WEIGHT}, MaxOcc²={MAX_OCC_WEIGHT}, MeanOcc={MEAN_OCC_WEIGHT}")
    print(f"   Control: Decision every {DECISION_INTERVAL_STEPS} steps, Min green {MIN_GREEN_STEPS}s")
    
    agent = DQNAgent(STATE_DIM, N_ACTIONS, LR)
    buf = ReplayBuffer(BUFFER_CAP)
    
    # Handle existing weights
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[INFO] Resuming from existing model: {MODEL_PATH}")
            agent.q_net.load_weights(MODEL_PATH)
            agent.update_target_network()
            print("[INFO] Weights loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}")
            print("[INFO] Starting fresh training.")
    else:
        print("[INFO] No existing model found. Starting fresh.")
    
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
                a = 0  # Initialize action
                
                # Reset phase management
                global _last_switch_step
                _last_switch_step = -10**9
                
                for step in range(MAX_STEPS):
                    # Only make decisions at intervals
                    do_decide = (step % DECISION_INTERVAL_STEPS == 0)
                    
                    if do_decide:
                        if random.random() < eps:
                            a = random.randint(0, N_ACTIONS - 1)
                        else:
                            a, q_vals = agent.predict_action(s_norm)
                            
                            # Q-value sanity check (every 50 decisions)
                            if global_step % 50 == 0:
                                print(f"[Q] step={global_step} | Q0={q_vals[0]:.3f}, Q1={q_vals[1]:.3f}")
                        
                        apply_action(a)
                    
                    traci.simulationStep()
                    s2_norm, s2_raw = get_state()
                    r = compute_reward(s2_raw)
                    
                    buf.push(s_norm, a, r, s2_norm, 0.0)
                    s_norm = s2_norm
                    total_reward += r
                    steps_taken = step + 1
                    
                    # Training
                    if len(buf) > WARMUP:
                        batch = buf.sample(BATCH_SIZE)
                        if batch:
                            loss = agent.train_step(*batch)
                            losses.append(float(loss))
                            
                            # Sync target network (multiple times per episode)
                            if global_step % TARGET_SYNC == 0:
                                agent.update_target_network()
                                print(f"[SYNC] Target network updated at step {global_step}")
                    
                    global_step += 1
                    
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break
            
            except traci.TraCIException as e:
                print(f"\n[ERROR] TraCI error in episode {ep}: {e}. Skipping.")
                try:
                    traci.close()
                except:
                    pass
                continue
            
            except Exception as e:
                print(f"\n[ERROR] Unexpected error in episode {ep}: {e}")
                import traceback
                traceback.print_exc()
                try:
                    traci.close()
                except:
                    pass
                continue
            
            finally:
                try:
                    traci.close()
                except:
                    pass
            
            # Episode end
            eps = max(EPS_END, eps * EPS_DECAY_FACTOR)
            
            avg_loss = np.mean(losses) if losses else 0.0
            avg_reward_per_step = total_reward / steps_taken if steps_taken > 0 else 0.0
            print(f"Ep {ep:03d}/{MAX_EPISODES} | TotalReward={total_reward:.2f} | AvgReward/step={avg_reward_per_step:.3f} | AvgLoss={avg_loss:.4f} | Steps={steps_taken} | eps={eps:.3f}")
            
            writer.writerow([ep, total_reward, avg_loss, eps, steps_taken])
            f.flush()
            
            # Save models
            agent.q_net.save_weights(MODEL_PATH)
            if total_reward > best_reward:
                best_reward = total_reward
                agent.q_net.save_weights(BEST_MODEL_PATH)
                print(f"💾 New best model saved (TotalReward={best_reward:.2f})")
    
    print("\n✅ Training complete!")
    print(f"   Best total reward: {best_reward:.2f}")
    print(f"   Final epsilon: {eps:.3f}")
    print(f"   Total training steps: {global_step}")
    print(f"\n📊 Check your results:")
    print(f"   - Training log: {LOG_CSV}")
    print(f"   - Best model: {BEST_MODEL_PATH}")
    print(f"\n🎯 Success metrics to look for:")
    print(f"   - Rewards trending upward over 50 episodes")
    print(f"   - Fewer teleports/emergency stops in SUMO output")
    print(f"   - Q-values stable (not exploding to ±100)")

if __name__ == "__main__":
    main()
