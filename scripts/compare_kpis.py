#!/usr/bin/env python3
import pandas as pd, os, argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--rl", required=True)
args = parser.parse_args()

def load(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["timestamp","avg_speed","avg_wait","queue_len","action"])
    with open(path) as f:
        first = f.readline().lower()
    skip = 1 if "timestamp" in first else 0
    df = pd.read_csv(path, skiprows=skip, names=["timestamp","avg_speed","avg_wait","queue_len","action"])
    for c in ["avg_speed","avg_wait","queue_len"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["avg_speed","avg_wait","queue_len"])
    return df

base = load(args.baseline)
dqn  = load(args.rl)

def summary(df, label):
    return {
        "label": label,
        "rows": len(df),
        "avg_speed": df["avg_speed"].mean() if len(df) else float("nan"),
        "avg_wait":  df["avg_wait"].mean()  if len(df) else float("nan"),
        "avg_queue": df["queue_len"].mean() if len(df) else float("nan")
    }

s = pd.DataFrame([summary(base,"Baseline"), summary(dqn,"DQN")])
print("\n=== KPI Summary Comparison ===")
print(s.round(3).to_string(index=False))

if s.loc[0,"rows"]>0 and s.loc[1,"rows"]>0 and s.loc[0,"avg_speed"]>0:
    speed_gain = ((s.loc[1,"avg_speed"]/s.loc[0,"avg_speed"])-1)*100
    wait_drop  = ((s.loc[0,"avg_wait"]-s.loc[1,"avg_wait"])/s.loc[0,"avg_wait"]*100) if s.loc[0,"avg_wait"]>0 else float("nan")
    queue_drop = ((s.loc[0,"avg_queue"]-s.loc[1,"avg_queue"])/s.loc[0,"avg_queue"]*100) if s.loc[0,"avg_queue"]>0 else float("nan")
    print(f"\n🚦 DQN vs Baseline:")
    print(f"• Speed improvement: {speed_gain:.2f}%")
    print(f"• Waiting-time reduction: {wait_drop:.2f}%")
    print(f"• Queue reduction: {queue_drop:.2f}%")
else:
    print("\n(Insufficient rows or zeros to compute percentage deltas.)")
