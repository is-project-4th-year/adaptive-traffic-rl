#=!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
import glob
from pathlib import Path

app = FastAPI()

# -----------------------------------------------------------
# CORS
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://40.120.26.11:8585",
        "http://localhost:8585",
        "http://127.0.0.1:8585",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# ROOTS
# -----------------------------------------------------------
ROOT = "/home/azureuser/traffic_rl/dash_public"
HIST = f"{ROOT}/episodes_history.csv"
EP_ROOT = "/home/azureuser/traffic_rl/episodes"


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def load_kpi(path: str):
    """Load KPI CSV or return None."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except:
        return None


def pct_gain(rlv, bas):
    """Percentage improvement where positive = RL better."""
    if bas <= 0:
        return 0.0
    return ((bas - rlv) / bas) * 100


def compute_episode_summary(ep_id: str):
    """Compute averaged KPIs + improvements for RL vs Baseline."""
    ep_clean = ep_id.strip().rstrip("/")
    ep_path = f"{EP_ROOT}/{ep_clean}"
    base = load_kpi(f"{ep_path}/kpi_baseline.csv")
    rl = load_kpi(f"{ep_path}/kpi_rl.csv")

    if base is None or rl is None:
        return None

    # Means
    base_speed = base["avg_speed"].mean()
    rl_speed = rl["avg_speed"].mean()

    base_wait = base["avg_wait"].mean()
    rl_wait = rl["avg_wait"].mean()

    base_queue = base["queue_len"].mean()
    rl_queue = rl["queue_len"].mean()

    return {
        "episode": ep_id,
        "timestamp": "",
        "seed": None,
        "profile": "Dynamic",
        "duration_min": 10,

        "summary": {
            "speed_rl": round(rl_speed, 3),
            "speed_base": round(base_speed, 3),
            "speed_improvement_pct":
                round(((rl_speed - base_speed) / base_speed) * 100, 2)
                if base_speed > 0 else 0,

            "delay_rl": round(rl_wait, 3),
            "delay_base": round(base_wait, 3),
            "delay_improvement_pct": -pct_gain(rl_wait, base_wait),

            "queue_rl": round(rl_queue, 3),
            "queue_base": round(base_queue, 3),
            "queue_improvement_pct": -pct_gain(rl_queue, base_queue),
        },

        # Placeholder until true per-approach JSON added
        "per_approach": [
            {"approach": "North", "rlDelay": rl_wait, "baselineDelay": base_wait,
             "rlQueue": rl_queue, "baselineQueue": base_queue},
            {"approach": "South", "rlDelay": rl_wait, "baselineDelay": base_wait,
             "rlQueue": rl_queue, "baselineQueue": base_queue},
            {"approach": "East",  "rlDelay": rl_wait, "baselineDelay": base_wait,
             "rlQueue": rl_queue, "baselineQueue": base_queue},
            {"approach": "West",  "rlDelay": rl_wait, "baselineDelay": base_wait,
             "rlQueue": rl_queue, "baselineQueue": base_queue},
        ],
    }


# -----------------------------------------------------------
# Paired history endpoint
# -----------------------------------------------------------
@app.get("/api/paired/summary")
def paired_summary():
    if not Path(HIST).exists():
        return []

    df = pd.read_csv(HIST)
    out = []

    for _, row in df.iterrows():
        out.append({
            "pair": row["episode"],
            "rl_speed": row["rl_speed"],
            "baseline_speed": row["baseline_speed"],
            "speed_impr_%": row["delta_speed"],
            "rl_wait": row["rl_wait"],
            "baseline_wait": row["baseline_wait"],
            "wait_red_%": row["delta_wait"],
            "rl_queue": row["rl_queue"],
            "baseline_queue": row["baseline_queue"],
            "queue_red_%": row["delta_queue"],
        })

    return out


# -----------------------------------------------------------
# LIVE STATUS ENDPOINT
# -----------------------------------------------------------
@app.get("/api/live")
def live_status():
    state_path = "/home/azureuser/traffic_rl/shared/state.json"
    action_log_path = "/home/azureuser/traffic_rl/shared/action_log.json"
    series_path = "/home/azureuser/traffic_rl/shared/live_series.json"

    try:
        with open(state_path) as f:
            s = json.load(f)
    except:
        s = {}

    try:
        series = json.load(open(series_path)) if os.path.exists(series_path) else []
    except:
        series = []

    try:
        logs = json.load(open(action_log_path)) if os.path.exists(action_log_path) else []
    except:
        logs = []

    return {
        "episode": s.get("episode"),
        "window": s.get("window", ""),
        "series": series,
        "queues": {
            "N": s.get("N_queue", 0),
            "S": s.get("S_queue", 0),
            "E": s.get("E_queue", 0),
            "W": s.get("W_queue", 0),
        },
        "phase": {
            "name": "NS-Green" if s.get("phase_binary", 0) == 1 else "EW-Green",
            "elapsed": s.get("time_in_phase", 0),
        },
        "actions": logs,
    }


# -----------------------------------------------------------
# EPISODE ENDPOINTS
# -----------------------------------------------------------
@app.get("/api/episodes/all")
def get_all_episodes():
    eps = sorted(os.listdir(EP_ROOT))
    out = []

    for ep in eps:
        if not ep.startswith("EP_"):
            continue

        result = compute_episode_summary(ep)
        if not result:
            continue

        # The frontend expects nested "summary"
        out.append({
            "episode": result["episode"],
            "timestamp": result.get("timestamp", ""),
            "summary": {
                "speed_rl": result["summary"]["speed_rl"],
                "speed_base": result["summary"]["speed_base"],
                "speed_improvement_pct": result["summary"]["speed_improvement_pct"],

                "delay_rl": result["summary"]["delay_rl"],
                "delay_base": result["summary"]["delay_base"],
                "delay_improvement_pct": result["summary"]["delay_improvement_pct"],

                "queue_rl": result["summary"]["queue_rl"],
                "queue_base": result["summary"]["queue_base"],
                "queue_improvement_pct": result["summary"]["queue_improvement_pct"]
            }
        })

    return out
@app.get("/api/episodes/{episode_id}")
def get_single_episode(episode_id: str):
    result = compute_episode_summary(episode_id)
    if not result:
        raise HTTPException(status_code=404, detail="Episode not found")

    return result
