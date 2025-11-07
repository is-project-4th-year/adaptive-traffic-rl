import pandas as pd

# --- Load KPI logs, skipping headers if they exist ---
base = pd.read_csv(
    "/home/azureuser/traffic_rl/logs/kpi_baseline.csv",
    skiprows=1,
    names=["timestamp", "avg_speed", "avg_wait", "queue_len", "action"]
)
dqn = pd.read_csv(
    "/home/azureuser/traffic_rl/logs/kpi_live.csv",
    skiprows=1,
    names=["timestamp", "avg_speed", "avg_wait", "queue_len", "action"]
)

# --- Convert all numeric columns safely ---
for col in ["avg_speed", "avg_wait", "queue_len"]:
    base[col] = pd.to_numeric(base[col], errors="coerce")
    dqn[col] = pd.to_numeric(dqn[col], errors="coerce")

# --- Summaries ---
def summary(df, label):
    return {
        "label": label,
        "avg_speed": df["avg_speed"].mean(),
        "avg_wait": df["avg_wait"].mean(),
        "avg_queue": df["queue_len"].mean()
    }

s = pd.DataFrame([summary(base, "Baseline"), summary(dqn, "DQN")])

# --- Metrics ---
speed_gain = ((s.loc[1, "avg_speed"] / s.loc[0, "avg_speed"]) - 1) * 100
wait_drop = ((s.loc[0, "avg_wait"] - s.loc[1, "avg_wait"]) / s.loc[0, "avg_wait"]) * 100 if s.loc[0, "avg_wait"] > 0 else 0
queue_drop = ((s.loc[0, "avg_queue"] - s.loc[1, "avg_queue"]) / s.loc[0, "avg_queue"]) * 100 if s.loc[0, "avg_queue"] > 0 else 0

# --- Print results ---
print("\n=== KPI Summary Comparison ===")
print(s.round(3))
print(f"\n🚦 DQN vs Baseline Performance:")
print(f"• Speed improvement: {speed_gain:.2f}%")
print(f"• Waiting-time reduction: {wait_drop:.2f}%")
print(f"• Queue reduction: {queue_drop:.2f}%")
