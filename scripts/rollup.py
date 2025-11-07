#!/usr/bin/env python3
import os, json, pandas as pd

LOG_DIR = os.path.expanduser("~/traffic_rl/logs")
DASH_FILE = os.path.expanduser("~/traffic_rl/dash_public/daily.json")

LOG_BASELINE = os.path.join(LOG_DIR, "kpi_baseline.csv")
LOG_RL       = os.path.join(LOG_DIR, "kpi_live.csv")

def summarize(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"avg_speed": 0, "avg_wait": 0, "queue_len": 0}

    # skip header if exists
    first = open(path).readline().lower()
    skip_header = 1 if "timestamp" in first else 0

    df = pd.read_csv(
        path,
        skiprows=skip_header,
        names=["timestamp","avg_speed","avg_wait","queue_len","action"],
        on_bad_lines="skip"
    )

    # sanitize
    df = df[pd.to_numeric(df["avg_speed"], errors="coerce").notna()]
    df = df[(df["avg_speed"] >= 0) & (df["avg_speed"] <= 15)]
    df = df[(df["queue_len"] >= 0) & (df["queue_len"] <= 500)]
    df = df[(df["avg_wait"] >= 0) & (df["avg_wait"] <= 10000)]

    return {
        "avg_speed": round(df["avg_speed"].mean(), 2),
        "avg_wait":  round(df["avg_wait"].mean(), 2),
        "queue_len": round(df["queue_len"].mean(), 2),
    }

base = summarize(LOG_BASELINE)
rl   = summarize(LOG_RL)

data = {
    "avg_speed_baseline": base["avg_speed"],
    "avg_speed_rl": rl["avg_speed"],
    "avg_wait_baseline": base["avg_wait"],
    "avg_wait_rl": rl["avg_wait"],
    "queue_baseline": base["queue_len"],
    "queue_rl": rl["queue_len"],
    "speed_improvement": round((rl["avg_speed"] - base["avg_speed"]) / base["avg_speed"] * 100, 2) if base["avg_speed"] > 0 else 0,
    "wait_reduction": round((base["avg_wait"] - rl["avg_wait"]) / base["avg_wait"] * 100, 2) if base["avg_wait"] > 0 else 0,
    "queue_reduction": round((base["queue_len"] - rl["queue_len"]) / base["queue_len"] * 100, 2) if base["queue_len"] > 0 else 0,
}

with open(DASH_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(json.dumps(data, indent=2))
