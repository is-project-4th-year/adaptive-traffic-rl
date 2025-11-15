#!/usr/bin/env python3
"""
online_train_v4_fixed_final.py

FIXES:
1. Correct SUMO phase transitions (0→1→2→3→0, never skip yellow)
2. State consistency after phase switches
3. Single epsilon decay (inside loop only)

10-feature state: [phase, time, 4×queue_length, 4×speed]
Dual-threshold hysteresis + queue-pressure override for switching.
Stable reward: avg_speed - max_queue - teleport_penalty (no deltas).

Features:
- Prioritized Experience Replay (PER)
- Evaluation loop (every N episodes)
- Best-model tracking
- CSV logging (training + eval)
- Resume from checkpoint
- Safe signal handling
- Decision interval (agent acts every 6 steps)
- Min/max green constraints
- Queue-pressure jam protection
"""

import os, sys, json, csv, time, random, math, signal
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, Reduction

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Signal handler for safe interrupt ---
INTERRUPTED = False
def handle_sigint(signum, frame):
    global INTERRUPTED
    print("\n^C Interrupt received - will stop after current episode.")
    INTERRUPTED = True
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# --- SUMO setup ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

TLS_ID = "J0"
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"
MODELS_DIR = "/home/azureuser/traffic_rl/models"
LOGS_DIR = "/home/azureuser/traffic_rl/logs"

MODEL_LATEST = f"{MODELS_DIR}/dqn_model_v4_latest.weights.h5"
MODEL_BEST = f"{MODELS_DIR}/dqn_model_v4_best.weights.h5"
SCALER_PATH = f"{MODELS_DIR}/scaler_enhanced.json"
TRAIN_CSV = f"{LOGS_DIR}/online_training_v4.csv"
EVAL_CSV = f"{LOGS_DIR}/online_eval_v4.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

STATE_DIM = 10
N_ACTIONS = 2

# --- RL hyperparameters ---
LR = 5e-5
GAMMA = 0.90
BATCH_SIZE = 128
BUFFER_CAP = 200_000
WARMUP = 4_000
TARGET_SYNC_STEPS = 2_500
MAX_EPISODES = 500

EPS_START, EPS_END = 0.9, 0.05
EPS_DECAY_PER_EPISODE = 0.995

# --- PER parameters ---
PER_ALPHA = 0.6
PER_BETA_START, PER_BETA_END = 0.5, 1.0
PER_BETA_STEPS = 200_000
PRIORITY_EPS = 1e-3

# --- Traffic control ---
DECISION_INTERVAL_STEPS = 3
MIN_GREEN_SECS = 12.0
MAX_GREEN_SECS = 60.0
MAX_SWITCHES_PER_EP = 24

LOW_QUEUE_THRESH = 2.0
ABS_QUEUE_MARGIN = 1.0
RELATIVE_ALPHA = 0.15

QUEUE_PRESSURE_LEN = 8.0
QUEUE_PRESSURE_HOLD = 2

# --- Reward ---
SPEED_REWARD_WEIGHT = 0.3
MAX_QUEUE_PENALTY_WEIGHT = 0.1
TELEPORT_PENALTY = 0.5

# --- Eval ---
EVAL_EVERY_EP = 10
EVAL_STEPS = 200

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

FLOW_DETECTORS = ["det_N_in", "det_S_in", "det_E_in", "det_W_in"]
QUEUE_DETECTORS = ["det_N_queue", "det_S_queue", "det_E_queue", "det_W_queue"]

# Global control state
_last_switch_time = -1e9
_switch_count = 0
_tp_count = 0
_pressure_dwell = 0

# ============================================================================ #
# NORMALIZATION
# ============================================================================ #

def load_scaler(path):
    if not os.path.exists(path):
        print(f"FATAL: Scaler missing at {path}"); sys.exit(1)
    with open(path) as f:
        d = json.load(f)
    mean = np.array(d["mean"], np.float32)
    scale = np.array(d["scale"], np.float32)
    scale = np.where(scale == 0, 1e-6, scale)
    print(f"[INFO] Scaler loaded from {path}")
    return mean, scale

MEAN, SCALE = load_scaler(SCALER_PATH)
def z_norm(x): 
    return (np.nan_to_num(x, nan=0.0) - MEAN) / SCALE

# ============================================================================ #
# PER BUFFER
# ============================================================================ #

class PERBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.s = np.zeros((cap, STATE_DIM), np.float32)
        self.a = np.zeros((cap,), np.int32)
        self.r = np.zeros((cap,), np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), np.float32)
        self.d = np.zeros((cap,), np.float32)
        self.p = np.zeros((cap,), np.float32)
        self.idx = 0
        self.full = False
        self.n_seen = 0

    def __len__(self):
        return self.cap if self.full else self.idx

    def push(self, s, a, r, s2, d, init_p=1.0):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = d
        self.p[i] = max(init_p, PRIORITY_EPS)
        self.idx = (self.idx + 1) % self.cap
        if self.idx == 0:
            self.full = True

    def sample(self, bs, beta):
        n = len(self)
        if n == 0:
            return None
        bs = min(bs, n)
        p_norm = self.p[:n] ** PER_ALPHA
        p_norm /= p_norm.sum()
        idx = np.random.choice(n, bs, replace=False, p=p_norm)
        w = (n * p_norm[idx]) ** (-beta)
        w /= w.max()
        self.n_seen += bs
        return (tf.constant(self.s[idx]),
                tf.constant(self.a[idx]),
                tf.constant(self.r[idx]),
                tf.constant(self.s2[idx]),
                tf.constant(self.d[idx]),
                tf.constant(w, np.float32),
                idx)

    def update_priorities(self, idx, td):
        self.p[idx] = np.maximum(td.numpy() + PRIORITY_EPS, self.p[idx])

# ============================================================================ #
# DQN AGENT
# ============================================================================ #

class DQN:
    def __init__(self):
        self.q = self._build_net()
        self.t = self._build_net()
        self.t.set_weights(self.q.get_weights())
        self.opt = Adam(learning_rate=LR, clipnorm=10.0)
        self.loss_fn = Huber(reduction=Reduction.NONE)
        self.global_step = 0

    def _build_net(self):
        return Sequential([
            Input(shape=(STATE_DIM,)),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(N_ACTIONS, activation="linear"),
        ])

    def sync(self):
        self.t.set_weights(self.q.get_weights())

    @tf.function
    def train_step(self, s, a, r, s2, d, is_w):
        q_next_main = self.q(s2, training=False)
        best_next = tf.argmax(q_next_main, axis=1, output_type=tf.int32)
        q_next_t = self.t(s2, training=False)
        idx_next = tf.stack([tf.range(tf.shape(best_next)[0], dtype=tf.int32), best_next], axis=1)
        target_next = tf.gather_nd(q_next_t, idx_next)
        y = r + (1.0 - d) * GAMMA * target_next

        with tf.GradientTape() as tape:
            q_all = self.q(s, training=True)
            idx = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
            q_pred = tf.gather_nd(q_all, idx)
            per_sample_loss = self.loss_fn(y, q_pred)
            loss = tf.reduce_mean(is_w * per_sample_loss)
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss, tf.abs(y - q_pred)

# ============================================================================ #
# STATE / REWARD / ACTION
# ============================================================================ #

def get_state():
    """Returns (normalized_state, raw_state).
    
    For J0 lefthand network:
    - Phases 0,1 = EW green/yellow
    - Phases 2,3 = NS green/yellow
    
    phase_binary: 1.0=EW green, 0.0=NS green
    """
    try:
        phase_idx = traci.trafficlight.getPhase(TLS_ID)
        phase_binary = 1.0 if phase_idx in [0, 1] else 0.0  # 1=EW, 0=NS

        time_in_phase = float(
            traci.trafficlight.getPhaseDuration(TLS_ID)
            - (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime())
        )
        time_in_phase = max(0.0, time_in_phase)

        queue_lengths = [traci.inductionloop.getLastStepVehicleNumber(d) for d in QUEUE_DETECTORS]
        avg_speeds = [traci.inductionloop.getLastStepMeanSpeed(d) for d in FLOW_DETECTORS]
        avg_speeds = [max(s, 0.0) for s in avg_speeds]

        raw = np.array([phase_binary, time_in_phase, *queue_lengths, *avg_speeds], np.float32)
        return z_norm(raw), raw
    except Exception as e:
        print(f"[ERROR] get_state: {e}")
        return z_norm(np.zeros(STATE_DIM, np.float32)), np.zeros(STATE_DIM, np.float32)

def should_switch(action, raw):
    """Dual-threshold hysteresis on queues."""
    qN, qS, qE, qW = raw[2:6]
    qNS = max(qN, qS)
    qEW = max(qE, qW)

    if qNS < LOW_QUEUE_THRESH and qEW < LOW_QUEUE_THRESH:
        return False

    if action == 0:  # wants EW green
        abs_ok = (qEW - qNS) > ABS_QUEUE_MARGIN
        rel_ok = qEW > (1.0 + RELATIVE_ALPHA) * max(qNS, 1.0)
        return abs_ok or rel_ok
    else:  # wants NS green
        abs_ok = (qNS - qEW) > ABS_QUEUE_MARGIN
        rel_ok = qNS > (1.0 + RELATIVE_ALPHA) * max(qEW, 1.0)
        return abs_ok or rel_ok

def apply_action(action):
    """
    FIXED: Proper SUMO phase transitions with yellow states.
    
    Phase sequence: 0→1→2→3→0
    - Phase 0: EW green
    - Phase 1: EW yellow
    - Phase 2: NS green  
    - Phase 3: NS yellow
    
    Action 0 = request EW green (phases 0,1)
    Action 1 = request NS green (phases 2,3)
    """
    global _last_switch_time, _switch_count, _pressure_dwell

    try:
        if _switch_count >= MAX_SWITCHES_PER_EP:
            return

        # Get CURRENT phase directly from SUMO
        cur_phase = traci.trafficlight.getPhase(TLS_ID)
        sim_t = traci.simulation.getTime()
        
        # Get fresh state for decision logic
        _, raw = get_state()
        time_in_phase = raw[1]
        qN, qS, qE, qW = raw[2:6]
        qNS = max(qN, qS)
        qEW = max(qE, qW)

        if sim_t - _last_switch_time < MIN_GREEN_SECS:
            return

        if qNS < LOW_QUEUE_THRESH and qEW < LOW_QUEUE_THRESH:
            return

        want_EW = (action == 0)

        # Queue pressure override
        heavy_NS = qNS >= QUEUE_PRESSURE_LEN
        heavy_EW = qEW >= QUEUE_PRESSURE_LEN
        if heavy_NS or heavy_EW:
            _pressure_dwell += 1
        else:
            _pressure_dwell = 0
        pressure_override = (_pressure_dwell >= QUEUE_PRESSURE_HOLD)

        # Max green force
        allow_force = False
        if time_in_phase >= MAX_GREEN_SECS:
            if cur_phase in [2, 3] and qEW > LOW_QUEUE_THRESH:  # NS green, EW jammed
                allow_force = True
            if cur_phase in [0, 1] and qNS > LOW_QUEUE_THRESH:  # EW green, NS jammed
                allow_force = True

        hysteresis_ok = should_switch(action, raw)

        if not (pressure_override or allow_force or hysteresis_ok):
            return

        # ===== CORRECT PHASE TRANSITIONS =====
        # Never skip yellow phases!
        
        if want_EW:
            # Want EW green (phases 0,1)
            if cur_phase == 2:  # Currently NS green
                traci.trafficlight.setPhase(TLS_ID, 3)  # Go to NS yellow first
                return
            elif cur_phase == 3:  # Currently NS yellow
                traci.trafficlight.setPhase(TLS_ID, 0)  # Now go to EW green
                _last_switch_time = sim_t
                _switch_count += 1
                return
        else:
            # Want NS green (phases 2,3)
            if cur_phase == 0:  # Currently EW green
                traci.trafficlight.setPhase(TLS_ID, 1)  # Go to EW yellow first
                return
            elif cur_phase == 1:  # Currently EW yellow
                traci.trafficlight.setPhase(TLS_ID, 2)  # Now go to NS green
                _last_switch_time = sim_t
                _switch_count += 1
                return

    except Exception as e:
        print(f"[ERROR] apply_action: {e}")

def compute_reward(raw, tp_delta=0):
    """Stable reward: speed - queue - teleport penalty."""
    speeds = raw[6:]
    queues = raw[2:6]
    avg_speed = float(np.mean(speeds))
    max_queue = float(np.max(queues))
    speed_reward = SPEED_REWARD_WEIGHT * avg_speed
    queue_penalty = MAX_QUEUE_PENALTY_WEIGHT * max_queue
    teleport_penalty = TELEPORT_PENALTY * tp_delta
    r = speed_reward - queue_penalty - teleport_penalty
    return float(np.clip(r, -10.0, 10.0))

# ============================================================================ #
# EVALUATION
# ============================================================================ #

def evaluate(agent, steps=EVAL_STEPS):
    """Run eval episode, return metrics dict."""
    try:
        traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])
        
        global _last_switch_time, _switch_count, _tp_count, _pressure_dwell
        _last_switch_time = -1e9
        _switch_count = 0
        _tp_count = 0
        _pressure_dwell = 0
        
        s, raw = get_state()
        sum_spd = sum_q = 0.0
        sw = tp = n = 0
        prev_a = 0 if raw[0] == 1.0 else 1

        for _ in range(steps):
            q = agent.q(s.reshape(1, -1), training=False).numpy()[0]
            a = int(np.argmax(q))

            b_tp = traci.simulation.getStartingTeleportNumber()
            apply_action(a)
            traci.simulationStep()
            a_tp = traci.simulation.getStartingTeleportNumber()
            tp += max(0, a_tp - b_tp)

            s, raw = get_state()
            sum_spd += float(np.mean(raw[6:]))
            sum_q += float(np.max(raw[2:6]))
            if a != prev_a:
                sw += 1
            prev_a = a
            n += 1

            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        n = max(1, n)
        return dict(
            avg_speed=sum_spd / n,
            avg_queue=sum_q / n,
            switches=sw,
            teleports=tp
        )
    finally:
        try:
            traci.close()
        except:
            pass

# ============================================================================ #
# TRAINING
# ============================================================================ #

def main():
    print("🚦 Starting DQN training v4 FIXED (Correct Phase Transitions)...")
    agent = DQN()
    buf = PERBuffer(BUFFER_CAP)

    if os.path.exists(MODEL_LATEST):
        print(f"[INFO] Resuming from {MODEL_LATEST}")
        agent.q.load_weights(MODEL_LATEST)
        agent.sync()

    new_train = not os.path.exists(TRAIN_CSV)
    tf1 = open(TRAIN_CSV, "a", newline="")
    tw = csv.writer(tf1)
    if new_train:
        tw.writerow(["ep", "reward", "loss", "eps", "steps", "buf", "sw", "tp"])

    new_eval = not os.path.exists(EVAL_CSV)
    tf2 = open(EVAL_CSV, "a", newline="")
    ew = csv.writer(tf2)
    if new_eval:
        ew.writerow(["ep", "avg_speed", "avg_queue", "sw", "tp"])

    eps = EPS_START
    best_speed = -1.0
    global_step = 0

    try:
        for ep in range(MAX_EPISODES):
            try:
                traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])

                global _last_switch_time, _switch_count, _tp_count, _pressure_dwell
                _last_switch_time = -1e9
                _switch_count = 0
                _tp_count = 0
                _pressure_dwell = 0

                s, raw = get_state()
                total = 0.0
                losses = []
                steps = 0
                sw = 0
                prev_a = 0 if raw[0] == 1.0 else 1

                for t in range(2000):
                    decide = (t % DECISION_INTERVAL_STEPS == 0)

                    if decide:
                        if random.random() < eps:
                            a = random.randint(0, N_ACTIONS - 1)
                        else:
                            q = agent.q(s.reshape(1, -1), training=False).numpy()[0]
                            a = int(np.argmax(q))
                        
                        # FIXED: Single epsilon decay location
                        eps = max(EPS_END, eps * EPS_DECAY_PER_EPISODE)
                    else:
                        a = prev_a

                    b_tp = traci.simulation.getStartingTeleportNumber()
                    apply_action(a)  # Uses fresh state from SUMO
                    traci.simulationStep()
                    a_tp = traci.simulation.getStartingTeleportNumber()
                    tp_step = max(0, a_tp - b_tp)
                    _tp_count += tp_step

                    s2, raw2 = get_state()  # Get fresh state after step
                    r = compute_reward(raw2, tp_step)
                    done = traci.simulation.getMinExpectedNumber() <= 0

                    if decide:
                        buf.push(s, a, r, s2, float(done))

                        if len(buf) > WARMUP:
                            beta = PER_BETA_START + min(1.0, buf.n_seen / PER_BETA_STEPS) * (PER_BETA_END - PER_BETA_START)
                            batch = buf.sample(BATCH_SIZE, beta)
                            if batch:
                                s_b, a_b, r_b, s2_b, d_b, w_b, idx = batch
                                loss, td = agent.train_step(s_b, a_b, r_b, s2_b, d_b, w_b)
                                buf.update_priorities(idx, td)
                                losses.append(float(loss))
                                if global_step % TARGET_SYNC_STEPS == 0:
                                    agent.sync()
                                    print(f"[SYNC] Target updated at step {global_step}")
                                global_step += 1

                    if a != prev_a and decide:
                        sw += 1
                    prev_a = a
                    s, raw = s2, raw2
                    total += r
                    steps = t + 1

                    if done:
                        break

                try:
                    traci.close()
                except:
                    pass

                avg_loss = float(np.mean(losses)) if losses else 0.0
                tw.writerow([ep, f"{total:.3f}", f"{avg_loss:.5f}", f"{eps:.3f}", steps, len(buf), sw, _tp_count])
                tf1.flush()
                agent.q.save_weights(MODEL_LATEST)

                if ep % EVAL_EVERY_EP == 0:
                    print(f"--- Eval (Ep {ep}) ---")
                    m = evaluate(agent, EVAL_STEPS)
                    ew.writerow([ep, f"{m['avg_speed']:.4f}", f"{m['avg_queue']:.4f}", m["switches"], m["teleports"]])
                    tf2.flush()
                    if m["avg_speed"] > best_speed:
                        best_speed = m["avg_speed"]
                        agent.q.save_weights(MODEL_BEST)
                        print(f"[BEST] ep={ep} speed={best_speed:.3f}")

                print(f"Ep{ep:03d}|R={total:.3f}|L={avg_loss:.4f}|eps={eps:.3f}|buf={len(buf)}|sw={sw}|tp={_tp_count}")

                if INTERRUPTED:
                    print("Graceful stop - saving.")
                    agent.q.save_weights(MODEL_LATEST)
                    break

            except traci.TraCIException as e:
                print(f"[ERROR] TraCI in ep {ep}: {e}")
                try:
                    traci.close()
                except:
                    pass
                continue

    except KeyboardInterrupt:
        print("\nInterrupted - saving.")
        agent.q.save_weights(MODEL_LATEST)
    finally:
        tf1.close()
        tf2.close()
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    main()
