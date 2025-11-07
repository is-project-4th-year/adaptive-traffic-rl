import os, sys

# --- PATH FIX START ---
# Get the absolute path of the directory this script is in
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (which is 'uhuru')
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Make all paths absolute
COMMON_PATH = os.path.expanduser('~/traffic_rl/rl_common')
DETS_FILE = os.path.join(PARENT_DIR, "det_ids.txt")
MODEL_FILE = os.path.join(SCRIPT_DIR, "uhuru_dqn_model.weights.h5")
ADDS_FILE = os.path.join(PARENT_DIR, "detectors.add.xml")
NET_FILE = os.path.join(PARENT_DIR, "cross.net.xml")
ROU_FILE = os.path.join(PARENT_DIR, "stress.rou.xml") # Use the correct file name
# --- PATH FIX END ---

sys.path.append(COMMON_PATH)
from env import SumoEnvironment
from agent import DQNAgent

DETS = [l.strip() for l in open(DETS_FILE) if l.strip()]
STATE_DIM = len(DETS)*2

agent = DQNAgent(state_dim=STATE_DIM, n_actions=2, eps_start=0.0, eps_end=0.0)
# Use the absolute path variable here
agent.load(MODEL_FILE)

adds = [ADDS_FILE]
env  = SumoEnvironment(NET_FILE, ROU_FILE, adds, DETS, gui=False)

s = env.reset(); total=0; steps=0
while True:
    a = agent.act(s)
    s, r, done, _ = env.step(a)
    total += r; steps += 1
    if done: break

print(f"infer done | steps: {steps} | total_reward: {total}")
