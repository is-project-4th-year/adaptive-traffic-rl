#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controller.py — DQN Adaptive Controller (real-time, 1 Hz)
- Network: ~/traffic_rl/junctions/uhuru/live.sumocfg
- TraCI port: 8812  (set via traci.start(..., port=8812))
- Policy endpoint: http://127.0.0.1:8000/action  (expects {"state":[...]} → {"action": int})
- Logs: ~/traffic_rl/logs/kpi_live.csv
"""

import os, sys, time, csv, logging, requests

# --- SUMO tools path ---
CANDIDATES = ["/usr/share/sumo", "/usr/local/share/sumo", "/opt/sumo"]
sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    for c in CANDIDATES:
        if os.path.isdir(c):
            sumo_home = c; break
if not sumo_home:
    print("FATAL: SUMO_HOME not found"); sys.exit(1)
tools = os.path.join(sumo_home, "tools")
if tools not in sys.path:
    sys.path.append(tools)

import traci  # noqa

# --- Config ---
STEP_LENGTH = 1.0
TLS_ID = "J0"
POLICY_URL = "http://127.0.0.1:8000/action"
LOG_FILE = "/home/azureuser/traffic_rl/logs/kpi_live.csv"
EDGES_IN = []

SUMO_CMD = [
    "sumo",  # use "sumo-gui" if you want to watch it
    "-c", "/home/azureuser/traffic_rl/junctions/uhuru/live.sumocfg",
    "--step-length", str(STEP_LENGTH),
    "--no-step-log",
    "--start", "true",
    "--quit-on-end", "false",
    "--seed", "42",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [controller] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def setup_kpi_log():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow(["timestamp","avg_speed","avg_wait","queue_len","action"])

def discover_inbound_edges():
    global EDGES_IN
    links = traci.trafficlight.getControlledLinks(TLS_ID)
    lanes = sorted({ l[0][0] for l in links if l and l[0] })
    EDGES_IN = sorted({ traci.lane.getEdgeID(l) for l in lanes })
    logging.info(f"discovered inbound edges: {EDGES_IN}")

def build_state():
    """Example 8-dim state: [avg_queue, avg_wait, hour, min10, ... optional extras]."""
    queues = [traci.edge.getLastStepHaltingNumber(e) for e in EDGES_IN]
    waits  = [traci.edge.getWaitingTime(e) for e in EDGES_IN]
    avg_q = (sum(queues)/len(queues)) if EDGES_IN else 0.0
    avg_w = (sum(waits)/len(waits)) if EDGES_IN else 0.0
    lt = time.localtime()
    hour = lt.tm_hour
    min10 = lt.tm_min // 10
    return [avg_q, avg_w, hour, min10]

def query_policy(state):
    try:
        r = requests.post(POLICY_URL, json={"state": state}, timeout=2)
        if r.ok:
            a = r.json().get("action", 0)
            if isinstance(a, int): return a
    except Exception as e:
        logging.warning(f"policy error: {e}")
    return 0  # safe default

def log_kpis(action:int):
    total_speed = total_wait = 0.0
    total_queue = 0
    for e in EDGES_IN:
        total_speed += traci.edge.getLastStepMeanSpeed(e)
        total_wait  += traci.edge.getWaitingTime(e)
        total_queue += traci.edge.getLastStepHaltingNumber(e)
    n = len(EDGES_IN) or 1
    avg_speed = total_speed / n
    avg_wait  = total_wait  / n
    ts = time.time_ns() / 1e9
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, avg_speed, avg_wait, total_queue, action])

def main():
    logging.info("Starting DQN controller (real-time 1 Hz) on port 8812…")
    setup_kpi_log()
    traci.start(SUMO_CMD, port=8812)  # << set port here ONLY
    discover_inbound_edges()

    try:
        while True:
            traci.simulationStep()

            state = build_state()
            action = query_policy(state)      # expect phase index (e.g., 0..3)
            traci.trafficlight.setPhase(TLS_ID, int(action))

            log_kpis(action=int(action))

            # 1 Hz pacing
            time.sleep(max(0.0, STEP_LENGTH))
    except KeyboardInterrupt:
        logging.info("Stopped manually.")
    finally:
        try: traci.close(False)
        except: pass
        logging.info("SUMO closed.")

if __name__ == "__main__":
    main()
