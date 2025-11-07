import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

LOG_DIR = Path.home() / "traffic_rl" / "logs"

st.set_page_config(page_title="Model Evaluation Metrics", layout="wide")
st.title("🚦 Traffic Controller Evaluation Dashboard")

baseline_files = sorted(LOG_DIR.glob("kpi_baseline_*.csv"))
rl_files = sorted(LOG_DIR.glob("kpi_rl_*.csv"))

if not baseline_files or not rl_files:
    st.warning("No paired KPI files found. Run baseline and RL episodes first.")
    st.stop()

baseline = pd.read_csv(baseline_files[-1])
rl = pd.read_csv(rl_files[-1])

def compute_metrics(df):
    return {
        "Average Wait Time (s)": df["avg_wait"].mean(),
        "Average Queue Length": df["queue_len"].mean(),
        "Average Speed (m/s)": df["avg_speed"].mean(),
        "Cumulative Reward": df["action"].sum()
    }

baseline_m = compute_metrics(baseline)
rl_m = compute_metrics(rl)

summary = pd.DataFrame([baseline_m, rl_m], index=["Baseline", "RL"])

st.subheader("📊 Summary Metrics")
st.dataframe(summary.style.format("{:.2f}"))

fig = px.bar(
    summary.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value"),
    x="Metric", y="Value", color="index", barmode="group",
    title="Comparison of Metrics: Baseline vs RL"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 KPI Trends (Time Series)")
cols = ["avg_speed", "avg_wait", "queue_len"]
for col in cols:
    fig = px.line(
        pd.DataFrame({
            "timestamp": baseline["timestamp"],
            "Baseline": baseline[col],
            "RL": rl[col]
        }),
        x="timestamp", y=["Baseline", "RL"],
        title=f"{col.replace('_', ' ').title()} Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
