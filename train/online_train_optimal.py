#!/usr/bin/env python3
# train/online_train_optimal.py
# FINAL PATCHED VERSION
# Fixes Huber Loss, Yellow Lights, and Eval Averaging
# Includes PER, Action Masking, Delta-Reward, t_in clamp, Safe Interrupt

import os, sys, time, math, json, csv, random, signal
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, Reduction  # Reduction for PER

# ---------- SIGNAL HANDLER FOR SAFE INTERRUPT ----------
INTERRUPTED = False
def handle_sigint(signum, frame):
    global INTERRUPTED
    print("\n^C Interrupt received - will stop after current episode.")
    INTERRUPTED = True
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# ---------- ENV PATHS ----------
TLS_ID = "J0"
SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg"
MODELS_DIR = "/home/azureuser/traffic_rl/models"
LOGS_DIR = "/home/azureuser/traffic_rl/logs"
SCALER_PATH = f"{MODELS_DIR}/scaler_enhanced.json"

MODEL_LATEST = f"{MODELS_DIR}/dqn_model_enhanced_latest.weights.h5"
MODEL_BEST   = f"{MODELS_DIR}/dqn_model_enhanced_best.weights.h5"
TRAIN_CSV    = f"{LOGS_DIR}/online_training_optimal.csv"
EVAL_CSV     = f"{LOGS_DIR}/online_eval_optimal.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------- SUMO ----------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci

# ---------- PARAMETERS ----------
STATE_DIM = 10
N_ACTIONS = 2
MIN_GREEN_STEPS = 24
DECISION_INTERVAL = 5
PRESSURE_MASK = True
PRESSURE_THRESH = 0.55
YELLOW_SECS = 5
LR = 2e-4
GAMMA = 0.90
BATCH_SIZE = 128
BUFFER_CAP = 200_000
WARMUP = 4_000
TARGET_SYNC_STEPS = 5_000
EPS_START, EPS_END = 0.9, 0.05
EPS_DECAY_PER_STEP = math.exp(math.log(EPS_END / EPS_START) / 50_000)
PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_BETA_STEPS = 0.6, 0.5, 1.0, 200_000
PRIORITY_EPS = 1e-3
EVAL_EVERY_EP, EVAL_STEPS = 10, 200
SEED = 1337
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------- SCALER ----------
def load_scaler(path):
    if not os.path.exists(path):
        print(f"FATAL: missing scaler at {path}"); sys.exit(1)
    with open(path) as f:
        d = json.load(f)
    mean, scale = np.array(d["mean"], np.float32), np.array(d["scale"], np.float32)
    scale = np.where(scale == 0, 1e-6, scale)
    return mean, scale

MEAN, SCALE = load_scaler(SCALER_PATH)
def z_norm(x): return (np.nan_to_num(x, nan=0.0) - MEAN) / SCALE

# ---------- MODEL ----------
def build_net():
    return Sequential([
        Input(shape=(STATE_DIM,)),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(N_ACTIONS, activation="linear"),
    ])

class DQN:
    def __init__(self):
        self.q, self.t = build_net(), build_net()
        self.t.set_weights(self.q.get_weights())
        self.opt = Adam(learning_rate=LR, clipnorm=10.0)
        # Use per-sample loss for PER
        self.loss_fn = Huber(reduction=Reduction.NONE)

    def sync(self): self.t.set_weights(self.q.get_weights())

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
        return loss, tf.abs(y - q_pred)  # TD-errors

# ---------- PER BUFFER ----------
class PERBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.s  = np.zeros((cap, STATE_DIM), np.float32)
        self.a  = np.zeros((cap,), np.int32)
        self.r  = np.zeros((cap,), np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), np.float32)
        self.d  = np.zeros((cap,), np.float32)
        self.p  = np.zeros((cap,), np.float32)
        self.idx = 0; self.full = False; self.n_seen = 0

    def __len__(self): return self.cap if self.full else self.idx

    def push(self, s, a, r, s2, d, init_p=1.0):
        i = self.idx
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.d[i] = d
        self.p[i] = max(init_p, PRIORITY_EPS)
        self.idx = (self.idx + 1) % self.cap
        if self.idx == 0: self.full = True

    def sample(self, bs, beta):
        n = len(self)
        if n == 0: return None
        bs = min(bs, n)
        p = self.p[:n] ** PER_ALPHA
        p /= p.sum()
        idx = np.random.choice(n, bs, replace=False, p=p)
        w = (n * p[idx]) ** (-beta)
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

# ---------- STATE / ACTION / REWARD ----------
FLOW_DETS  = ["det_N_in","det_S_in","det_E_in","det_W_in"]
QUEUE_DETS = ["det_N_queue","det_S_queue","det_E_queue","det_W_queue"]

def read_state():
    phase = traci.trafficlight.getPhase(TLS_ID)
    phase_bin = 1.0 if phase in [0, 1] else 0.0
    time_raw = float(
        traci.trafficlight.getPhaseDuration(TLS_ID)
        - (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime())
    )
    time_in_phase = max(0.0, time_raw)  # clamp
    occ = [traci.inductionloop.getLastStepOccupancy(i) for i in QUEUE_DETS]
    spd = [max(traci.inductionloop.getLastStepMeanSpeed(i), 0.0) for i in FLOW_DETS]
    raw = np.array([phase_bin, time_in_phase, *occ, *spd], np.float32)
    return z_norm(raw), raw, phase

PRESSURE_MARGIN = 0.10   # require 10% advantage to justify a switch
COOLDOWN_STEPS = 8       # extra guard after dwell (prevents immediate ping-pong)

def valid_actions_mask(phase, time_in_phase, raw):
    """
    Returns [can_EW, can_NS] where 1 means allowed, 0 means masked out.
    raw[2:6] = [N_occ, S_occ, E_occ, W_occ] (%); we take max per axis.
    """
    holding_EW = phase in [0, 1]  # 0/1 = EW, 2/3 = NS
    # Enforce minimum green
    if time_in_phase < MIN_GREEN_STEPS:
        return np.array([1, 0], np.float32) if holding_EW else np.array([0, 1], np.float32)

    # Short cooldown after dwell completes (extra hysteresis)
    if time_in_phase < (MIN_GREEN_STEPS + COOLDOWN_STEPS):
        return np.array([1, 0], np.float32) if holding_EW else np.array([0, 1], np.float32)

    # Pressure gating with margin
    if PRESSURE_MASK:
        ns = max(raw[2], raw[3]) / 100.0
        ew = max(raw[4], raw[5]) / 100.0
        if holding_EW:
            # Only allow NS if NS is clearly busier
            if not (ns >= PRESSURE_THRESH and (ns - ew) >= PRESSURE_MARGIN):
                return np.array([1, 0], np.float32)
        else:
            # Only allow EW if EW is clearly busier
            if not (ew >= PRESSURE_THRESH and (ew - ns) >= PRESSURE_MARGIN):
                return np.array([0, 1], np.float32)

    # Otherwise both allowed (policy will choose)
    return np.array([1, 1], np.float32)

def apply_action(a, phase):
    # 0 -> want EW (phase 0), 1 -> want NS (phase 2)
    if a == 0 and phase in [2, 3]:      # currently NS; switch to NS yellow then EW green
        traci.trafficlight.setPhase(TLS_ID, 3)  # NS yellow
        traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_SECS)
    elif a == 1 and phase in [0, 1]:    # currently EW; switch to EW yellow then NS green
        traci.trafficlight.setPhase(TLS_ID, 1)  # EW yellow
        traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_SECS)

class RewardTracker:
    def __init__(self):
        self.prev_s = self.prev_w = self.prev_q = 0.0
    def step(self, raw, a, prev, tp):
        spd = float(np.mean(raw[6:10]))
        occ = float(np.max(raw[2:6])) / 100.0
        wait = np.clip((1 - np.clip(spd / 12.0, 0, 1)) * 0.7 + occ * 0.3, 0, 1)
        sn, qn = np.clip(spd / 12.0, 0, 1), np.clip(occ, 0, 1)
        ds, dw, dq = sn - self.prev_s, wait - self.prev_w, qn - self.prev_q
        sw, tp_ = (a != prev), tp > 0
        r = 0.6 * ds - 0.3 * dw - 0.2 * dq - 0.1 * sw - 0.6 * tp_
        self.prev_s, self.prev_w, self.prev_q = sn, wait, qn
        return float(np.clip(r, -1, 1))

# ---------- EVAL ----------
def evaluate(agent, steps=EVAL_STEPS):
    traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])
    try:
        s, raw, phase = read_state()
        rt = RewardTracker()
        prev = 0 if raw[0] == 1 else 1
        sw = tp = n = 0
        sum_spd = sum_w = sum_q = 0.0
        for _ in range(steps):
            time_raw = float(
                traci.trafficlight.getPhaseDuration(TLS_ID)
                - (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime())
            )
            time_in_phase = max(0.0, time_raw)
            mask = valid_actions_mask(phase, time_in_phase, raw)
            q = agent.q(s.reshape(1, -1), training=False).numpy()[0]
            q[mask == 0] = -1e9
            a = int(np.argmax(q))

            b_tp = traci.simulation.getStartingTeleportNumber()
            apply_action(a, phase)
            traci.simulationStep()
            a_tp = traci.simulation.getStartingTeleportNumber()
            tp += max(0, a_tp - b_tp)

            s2, raw2, phase2 = read_state()

            sp = float(np.mean(raw2[6:10]))
            oq = float(np.max(raw2[2:6])) / 100.0
            w = np.clip((1 - np.clip(sp / 12.0, 0, 1)) * 0.7 + oq * 0.3, 0, 1)
            sum_spd += sp; sum_w += w; sum_q += oq
            if a != prev: sw += 1
            s, raw, phase = s2, raw2, phase2; prev = a
            n += 1

            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        n = max(1, n)
        return dict(
            avg_speed=sum_spd / n,
            avg_wait_proxy=sum_w / n,
            avg_queue=sum_q / n,
            switches=sw,
            teleports=tp
        )
    finally:
        try: traci.close()
        except: pass

# ---------- TRAIN ----------
def main():
    print("Starting DQN training (Optimal, Safe Interrupt, all fixes)...")
    agent = DQN()
    buf = PERBuffer(BUFFER_CAP)

    if os.path.exists(MODEL_LATEST):
        agent.q.load_weights(MODEL_LATEST)
        agent.sync()
        print(f"Resumed weights from {MODEL_LATEST}")

    new_train = not os.path.exists(TRAIN_CSV)
    tf1 = open(TRAIN_CSV, "a", newline="")
    tw = csv.writer(tf1)
    if new_train:
        tw.writerow(["ep","reward","loss","eps","steps","buf","sw","tp"])

    new_eval = not os.path.exists(EVAL_CSV)
    tf2 = open(EVAL_CSV, "a", newline="")
    ew = csv.writer(tf2)
    if new_eval:
        ew.writerow(["ep","avg_speed","avg_wait_proxy","avg_queue","sw","tp"])

    eps = EPS_START; best_speed = -1.0; global_step = 0
    try:
        for ep in range(10_000):
            traci.start(["sumo", "-c", SUMO_CFG, "--no-step-log", "--no-warnings"])
            s, raw, phase = read_state()
            rt = RewardTracker()
            total = 0.0; losses = []; steps = 0; sw = 0; tp_ep = 0
            prev = 0 if raw[0] == 1 else 1

            for t in range(800):
                time_raw = float(
                    traci.trafficlight.getPhaseDuration(TLS_ID)
                    - (traci.trafficlight.getNextSwitch(TLS_ID) - traci.simulation.getTime())
                )
                time_in_phase = max(0.0, time_raw)
                mask = valid_actions_mask(phase, time_in_phase, raw)
                decide = (t % DECISION_INTERVAL == 0)

                if decide:
                    if random.random() < eps:
                        a = random.choice([i for i in range(N_ACTIONS) if mask[i] > 0])
                    else:
                        q = agent.q(s.reshape(1, -1), training=False).numpy()[0]
                        q[mask == 0] = -1e9
                        a = int(np.argmax(q))
                else:
                    a = prev

                b_tp = traci.simulation.getStartingTeleportNumber()
                apply_action(a, phase)
                traci.simulationStep()
                a_tp = traci.simulation.getStartingTeleportNumber()  # <-- fixed typo
                tp_step = max(0, a_tp - b_tp)
                tp_ep += tp_step

                s2, raw2, phase2 = read_state()
                r = rt.step(raw2, a, prev, tp_step)
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
                                print(f"[SYNC] Target updated at training step {global_step}")
                            global_step += 1

                if a != prev and decide:
                    sw += 1
                prev = a
                s, raw, phase = s2, raw2, phase2
                total += r
                steps = t + 1
                eps = max(EPS_END, eps * EPS_DECAY_PER_STEP)

                if done:
                    break

            try:
                traci.close()
            except:
                pass

            avg_loss = float(np.mean(losses)) if losses else 0.0
            tw.writerow([ep, f"{total:.3f}", f"{avg_loss:.5f}", f"{eps:.3f}", steps, len(buf), sw, tp_ep])
            tf1.flush()
            agent.q.save_weights(MODEL_LATEST)

            if ep % EVAL_EVERY_EP == 0:
                print(f"--- Evaluating (Ep {ep}) ---")
                m = evaluate(agent, EVAL_STEPS)
                ew.writerow([ep, f"{m['avg_speed']:.4f}", f"{m['avg_wait_proxy']:.4f}",
                             f"{m['avg_queue']:.4f}", m["switches"], m["teleports"]])
                tf2.flush()
                if m["avg_speed"] > best_speed:
                    best_speed = m["avg_speed"]
                    agent.q.save_weights(MODEL_BEST)
                    print(f"[BEST] ep={ep} avg_speed={best_speed:.3f} -> saved {MODEL_BEST}")

            print(f"Ep{ep:03d}|R={total:.3f}|L={avg_loss:.4f}|eps={eps:.3f}|buf={len(buf)}|sw={sw}|tp={tp_ep}")

            if INTERRUPTED:
                print("Graceful stop requested - saving latest weights and exiting...")
                agent.q.save_weights(MODEL_LATEST)
                break

    except KeyboardInterrupt:
        print("Interrupted - saving latest.")
        agent.q.save_weights(MODEL_LATEST)
    finally:
        tf1.close(); tf2.close()
        try: traci.close()
        except: pass

if __name__ == "__main__":
    main()
