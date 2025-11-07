#!/usr/bin/env python3
import os, json, time, logging
from statistics import mean

LIVE_PATH = "/home/azureuser/traffic_rl/shared/live_speeds.json"
STATE_PATH = "/home/azureuser/traffic_rl/shared/state.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] [collector] %(message)s")

def compute_state(data):
    segs = data.get("segments", [])
    if not segs:
        return None

    avg_delay = mean(s.get("delay_s", 0) for s in segs)
    avg_speed = mean(s.get("speed_kmh", 0) for s in segs)
    veh_count = min(int(avg_delay * 4), 800)
    congestion_index = round(0.7 * avg_delay + 0.3 * (100 - avg_speed), 2)

    # time features
    now = time.localtime()
    hour = now.tm_hour
    min10 = now.tm_min // 10

    return {
        "veh_count": veh_count,
        "avg_wait": round(avg_delay, 2),
        "avg_speed": round(avg_speed, 2),
        "stopped": veh_count,
        "delay": round(avg_delay, 2),
        "congestion_index": congestion_index,
        "north_flow": round(avg_speed, 2),
        "south_flow": round(avg_speed, 2),
        "hour": hour,
        "min10": min10
    }

def main():
    logging.info("collector v4 — watching live_speeds.json")
    while True:
        try:
            if os.path.exists(LIVE_PATH):
                with open(LIVE_PATH) as f:
                    data = json.load(f)
                state = compute_state(data)
                if state:
                    with open(STATE_PATH, "w") as f:
                        json.dump(state, f, indent=2)
                    logging.info(
                        f"state updated: veh={state['veh_count']} "
                        f"speed={state['avg_speed']} delay={state['delay']} "
                        f"index={state['congestion_index']} hour={state['hour']} min10={state['min10']}"
                    )
            else:
                logging.warning("live_speeds.json missing")
        except Exception as e:
            logging.error(f"update failed: {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()
