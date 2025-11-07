import sys, os, pickle, random, traceback
sys.path.append(os.path.expanduser('~/traffic_rl/rl_common'))
from env import SumoEnvironment
from agent import DQNAgent

DETS = [l.strip() for l in open("../det_ids.txt") if l.strip()]
STATE_DIM = len(DETS)*2
ACTIONS = 2

def run_episode(env, agent, explore_prob=0.95, max_steps=600):
    s = env.reset(); steps = 0
    while True:
        a = random.randrange(ACTIONS) if random.random() < explore_prob else agent.act(s)
        s2, r, done, _ = env.step(a)
        agent.remember(s,a,r,s2,float(done))
        s = s2; steps += 1
        if done or steps >= max_steps: break

if __name__ == "__main__":
    # The TL logic is in cross.net.xml, so we ONLY load detectors.
    adds = ["../detectors.add.xml"]
    env_typ = SumoEnvironment("../cross.net.xml","../typical.rou.xml",adds,DETS,tls_id=None,gui=False)
    env_str = SumoEnvironment("../cross.net.xml","../stress.rou.xml", adds,DETS,tls_id=None,gui=False)

    agent = DQNAgent(state_dim=STATE_DIM, n_actions=ACTIONS, eps_start=1.0, eps_end=0.05)

    try:
        print("Episode 1/2 (typical)...", flush=True)
        run_episode(env_typ, agent, 0.95, 1200)
        print("Episode 2/2 (stress)...", flush=True)
        run_episode(env_str, agent, 0.95, 1200)
    except Exception as e:
        print("ERROR during logging:", e)
        traceback.print_exc()
    finally:
        out = "experience_dataset.pkl"
        with open(out,"wb") as f:
            pickle.dump(list(agent.memory), f)
        print(f"Saved {out} with {len(agent.memory)} samples", flush=True)
