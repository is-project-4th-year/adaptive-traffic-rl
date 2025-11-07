#!/usr/bin/env python3
"""
run_live.py
-----------
Live inference loop that:
- loads trained DQN (8-dim input)
- runs SUMO via SumoEnvironment
- replaces detector speeds with Google live speeds (same 4 features, same shape)
- takes actions until the sim ends

Assumptions:
- det_ids.txt lists 4 detectors in N,S,E,W approach order
- weights file: ~/traffic_rl/junctions/uhuru/rl/uhuru_dqn_model.weights.h5
- live speeds JSON written by gmaps_sensor: ~/traffic_rl/shared/live_speeds.json
- SUMO net/rou/add files live in ../ (uhuru folder)
"""

import os
import json
import numpy as np
import sys
sys.path.append(os.path.expanduser('~/traffic_rl/rl_common'))

from env import SumoEnvironment
from agent import DQNAgent

# ---- paths & config ----
BASE_DIR   = os.path.expanduser('~/traffic_rl/junctions/uhuru')
RL_DIR     = os.path.join(BASE_DIR, 'rl')
STATE_JSON = os.path.expanduser('~/traffic_rl/shared/live_speeds.json')

NET_FILE   = os.path.join(BASE_DIR, 'cross.net.xml')
ROU_FILE   = os.path.join(BASE_DIR, 'stress.rou.xml')      # or typical.rou.xml if you prefer
ADD_FILES  = [os.path.join(BASE_DIR, 'detectors.add.xml')] # TL logic is embedded in cross.net.xml

DET_LIST   = os.path.join(BASE_DIR, 'det_ids.txt')
WEIGHTS    = os.path.join(RL_DIR, 'uhuru_dqn_model.weights.h5')

GUI = False             # set True if you want to watch it
STEP_LENGTH = 1.0       # SUMO step length (s)

# ---- load detector IDs (expect 4) ----
DETS = [line.strip() for line in open(DET_LIST) if line.strip()]
if len(DETS) != 4:
    raise RuntimeError(f"det_ids.txt must list 4 detectors (N,S,E,W). Got {len(DETS)}: {DETS}")

# model input size: 4 counts + 4 speeds = 8
STATE_DIM = len(DETS) * 2  # 8
N_ACTIONS = 2              # 0=extend, 1=switch
EPS_START = 0.0            # pure exploitation in live run
EPS_END   = 0.0

# ---- helper: read Google speeds in N,S,E,W order ----
def read_gmaps_speeds():
    """
    Returns [N,S,E,W] speeds (m/s) from live_speeds.json.
    Missing file/keys => 0.0 fallback.
    """
    try:
        with open(STATE_JSON, 'r') as f:
            data = json.load(f)
        segs = {s.get("id"): s for s in data.get("segments", [])}
        N = float(segs.get("N_approach", {}).get("speed_mps", 0.0))
        S = float(segs.get("S_approach", {}).get("speed_mps", 0.0))
        E = float(segs.get("E_approach", {}).get("speed_mps", 0.0))
        W = float(segs.get("W_approach", {}).get("speed_mps", 0.0))
        return [N, S, E, W]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

# ---- helper: inject gmaps speeds into the state vector ----
def augment_state(state_vec: np.ndarray) -> np.ndarray:
    """
    Incoming s from env.get_state():
      s = [N_cnt,S_cnt,E_cnt,W_cnt, N_spd,S_spd,E_spd,W_spd]  (len=8)
    We overwrite the last 4 entries with Google speeds to keep dim=8.
    """
    s = np.array(state_vec, dtype=np.float32).copy()
    g = read_gmaps_speeds()
    s[-4:] = np.array(g, dtype=np.float32)
    return s

def main():
    # build env
    env = SumoEnvironment(
        net_file=NET_FILE,
        rou_file=ROU_FILE,
        add_files=ADD_FILES,
        det_ids=DETS,
        tls_id=None,
        gui=GUI,
        step_length=STEP_LENGTH
    )

    # build agent & load weights
    agent = DQNAgent(
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        eps_start=EPS_START,
        eps_end=EPS_END
    )
    agent.load(WEIGHTS)  # uses model.load_weights under the hood

    # optional: write outputs if you want files for analysis
    os.environ["SUMO_TRIPINFO"]  = "live_trips.xml"
    os.environ["SUMO_EMISSIONS"] = "live_emissions.xml"

    # run episode
    s = env.reset()
    s = augment_state(s)
    total_r = 0.0
    steps = 0

    while True:
        a = agent.act(s)              # epsilon=0 => greedy
        s2, r, done, _ = env.step(a)
        s2 = augment_state(s2)        # inject fresh speeds
        total_r += r
        steps += 1
        s = s2
        if done:
            break

    print(f"[live] done | steps={steps} | total_reward={total_r:.2f}")

if __name__ == "__main__":
    main()
