#!/usr/bin/env python3
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--rl", required=True)
parser.add_argument("--window", default="0:15")
args = parser.parse_args()

# Load
b = pd.read_csv(args.baseline)
r = pd.read_csv(args.rl)

# Convert timestamps → numeric
for col in ["timestamp"]:
    b[col] = pd.to_numeric(b[col], errors="coerce")
    r[col] = pd.to_numeric(r[col], errors="coerce")

# Round timestamps to nearest 5s
b["t_round"] = (b["timestamp"] / 5).round() * 5
r["t_round"] = (r["timestamp"] / 5).round() * 5

# Merge on rounded timestamp
m = pd.merge(b, r, on="t_round", suffixes=("_b", "_r"))

# Window slicing: "0:15" means rows 0–15 of merged table
start, end = map(int, args.window.split(":"))
m = m.iloc[start:end]

# Summary metrics
summary = pd.DataFrame({
    "avg_speed_b": [m["avg_speed_b"].mean()],
    "avg_speed_r": [m["avg_speed_r"].mean()],
    "avg_wait_b":  [m["avg_wait_b"].mean()],
    "avg_wait_r":  [m["avg_wait_r"].mean()],
    "queue_b":     [m["queue_len_b"].mean()],
    "queue_r":     [m["queue_len_r"].mean()],
})

print("\n===== BUSY WINDOW COMPARISON =====")
print(summary)

speed_gain = (summary.avg_speed_r.iloc[0] - summary.avg_speed_b.iloc[0]) / summary.avg_speed_b.iloc[0] * 100
wait_red   = (summary.avg_wait_b.iloc[0] - summary.avg_wait_r.iloc[0]) / summary.avg_wait_b.iloc[0] * 100
queue_red  = (summary.queue_b.iloc[0] - summary.queue_r.iloc[0]) / summary.queue_b.iloc[0] * 100

print(f"\n🚦 RL speed improvement: {speed_gain:.2f}%")
print(f"⏳ RL waiting-time reduction: {wait_red:.2f}%")
print(f"🧵 RL queue reduction: {queue_red:.2f}%\n")
