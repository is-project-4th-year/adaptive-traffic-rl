#!/usr/bin/env python3
"""
daily_rollup.py — build a clean daily.json from episodes_history.csv

Input:
  ~/traffic_rl/dash_public/episodes_history.csv

Columns:
  episode,timestamp,baseline_speed,rl_speed,delta_speed,
  baseline_wait,rl_wait,delta_wait,
  baseline_queue,rl_queue,delta_queue

Output:
  ~/traffic_rl/dash_public/daily.json

Schema:
{
  "date": "YYYY-MM-DD",
  "episodes": [
    {
      "episode_id": "...",
      "timestamp": "...",
      "baseline": { "avg_speed": .., "avg_wait": .., "avg_queue": .. },
      "rl":       { ... },
      "delta_pct": { "speed": .., "wait": .., "queue": .. }
    },
    ...
  ],
  "summary": {
    "episodes": N,
    "avg_speed_base": ...,
    "avg_speed_rl": ...,
    "avg_wait_base": ...,
    "avg_wait_rl": ...,
    "avg_queue_base": ...,
    "avg_queue_rl": ...
  }
}

Invalid episodes NEVER enter episodes_history.csv, so daily.json
stays clean automatically.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = "/home/azureuser/traffic_rl"
DASH_DIR = f"{ROOT}/dash_public"
HIST_PATH = f"{DASH_DIR}/episodes_history.csv"
OUT_PATH = f"{DASH_DIR}/daily.json"


def main():
    Path(DASH_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(HIST_PATH) or os.path.getsize(HIST_PATH) == 0:
        # No history yet → minimal daily.json
        payload = {
            "date": datetime.now().date().isoformat(),
            "episodes": [],
            "summary": {
                "episodes": 0,
                "avg_speed_base": 0.0,
                "avg_speed_rl": 0.0,
                "avg_wait_base": 0.0,
                "avg_wait_rl": 0.0,
                "avg_queue_base": 0.0,
                "avg_queue_rl": 0.0,
            },
        }
        with open(OUT_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"daily.json written (no episodes) → {OUT_PATH}")
        return

    df = pd.read_csv(HIST_PATH)

    if df.empty:
        payload = {
            "date": datetime.now().date().isoformat(),
            "episodes": [],
            "summary": {
                "episodes": 0,
                "avg_speed_base": 0.0,
                "avg_speed_rl": 0.0,
                "avg_wait_base": 0.0,
                "avg_wait_rl": 0.0,
                "avg_queue_base": 0.0,
                "avg_queue_rl": 0.0,
            },
        }
        with open(OUT_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"daily.json written (empty history) → {OUT_PATH}")
        return

    # Normalize numeric columns
    num_cols = [
        "baseline_speed",
        "rl_speed",
        "delta_speed",
        "baseline_wait",
        "rl_wait",
        "delta_wait",
        "baseline_queue",
        "rl_queue",
        "delta_queue",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Episodes list
    episodes = []
    for _, row in df.iterrows():
        ep_id = str(row["episode"])
        ts = str(row["timestamp"])

        ep = {
            "episode_id": ep_id,
            "timestamp": ts,
            "baseline": {
                "avg_speed": float(row["baseline_speed"]),
                "avg_wait": float(row["baseline_wait"]),
                "avg_queue": float(row["baseline_queue"]),
            },
            "rl": {
                "avg_speed": float(row["rl_speed"]),
                "avg_wait": float(row["rl_wait"]),
                "avg_queue": float(row["rl_queue"]),
            },
            "delta_pct": {
                "speed": float(row["delta_speed"]),
                "wait": float(row["delta_wait"]),
                "queue": float(row["delta_queue"]),
            },
        }
        episodes.append(ep)

    # Summary (simple average over episodes)
    n = len(df)
    summary = {
        "episodes": int(n),
        "avg_speed_base": float(df["baseline_speed"].mean()),
        "avg_speed_rl": float(df["rl_speed"].mean()),
        "avg_wait_base": float(df["baseline_wait"].mean()),
        "avg_wait_rl": float(df["rl_wait"].mean()),
        "avg_queue_base": float(df["baseline_queue"].mean()),
        "avg_queue_rl": float(df["rl_queue"].mean()),
    }

    # Date: from first timestamp if parseable, else today
    try:
        first_ts = df["timestamp"].iloc[0]
        date = datetime.fromisoformat(str(first_ts)).date().isoformat()
    except Exception:
        date = datetime.now().date().isoformat()

    payload = {
        "date": date,
        "episodes": episodes,
        "summary": summary,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"daily.json written with {n} episodes → {OUT_PATH}")


if __name__ == "__main__":
    main()
