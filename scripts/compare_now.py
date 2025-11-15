#!/usr/bin/env python3
"""
Quick RL vs Baseline KPI comparison for current live CSVs.

This script assumes CSV rows have *no headers* and follow:
timestamp, speed, wait, queue, action
"""

import pandas as pd
import os

BASE_PATH = "/home/azureuser/traffic_rl/logs/kpi_baseline.csv"
RL_PATH   = "/home/azureuser/traffic_rl/logs/kpi_live.csv"

def load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[WARN] Missing or empty file: {path}")
        return pd.DataFrame(columns=["timestamp","speed","wait","queue","action"])

    cols = ["timestamp","speed","wait","queue","action"]
    return pd.read_csv(path, header=None, names=cols)

def compare(b, r):
    print("\n=== ROW COUNTS ===")
    print(f"Baseline: {len(b)}")
    print(f"RL:       {len(r)}")

    if len(b) == 0 or len(r) == 0:
        print("\n[ERROR] Not enough data to compare.")
        return

    print("\n=== AVERAGES ===")
    metrics = ["speed", "wait", "queue"]

    for m in metrics:
        b_val = b[m].mean()
        r_val = r[m].mean()

        if b_val == 0:
            pct = float("inf")
        else:
            pct = ((r_val - b_val) / b_val) * 100

        print(f"{m}:")
        print(f"  baseline = {b_val:.3f}")
        print(f"  RL       = {r_val:.3f}")
        print(f"  change   = {pct:+.2f}%\n")

if __name__ == "__main__":
    baseline_df = load_csv(BASE_PATH)
    rl_df       = load_csv(RL_PATH)
    compare(baseline_df, rl_df)
