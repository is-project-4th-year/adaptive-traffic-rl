#!/usr/bin/env python3
"""
app.py — Traffic RL vs Baseline Dashboard

Data sources:
- paired_metrics.json: latest VALID episode metrics
- episodes_history.csv: episode-by-episode history (only valid ones)
- last_status.json: status of the most recent run (ok/invalid)

Behavior:
- If last_status = invalid → show a clear red warning:
    "❌ Episode invalid — insufficient paired ticks"
- Cards show RL values with % delta vs baseline
- Trend charts show baseline vs RL over episodes
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = "/home/azureuser/traffic_rl"
DASH_DIR = f"{ROOT}/dash_public"
PAIR_PATH = f"{DASH_DIR}/paired_metrics.json"
HIST_PATH = f"{DASH_DIR}/episodes_history.csv"
STATUS_PATH = f"{DASH_DIR}/last_status.json"

st.set_page_config(page_title="Traffic RL Dashboard", layout="wide")

st.title("🚦 Traffic Signal Optimization — RL vs Baseline Dashboard")

Path(DASH_DIR).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# LAST STATUS (invalid episodes)
# -------------------------------------------------------------------
status_data = None
if os.path.exists(STATUS_PATH):
    try:
        with open(STATUS_PATH, "r") as f:
            status_data = json.load(f)
    except Exception:
        status_data = None

if status_data and status_data.get("status") == "invalid":
    ep = status_data.get("episode_id", "unknown")
    msg = status_data.get("message", "")
    st.error(f"❌ Last episode `{ep}` invalid — {msg}")
elif status_data and status_data.get("status") == "ok":
    ep = status_data.get("episode_id", "unknown")
    ts = status_data.get("timestamp", "")
    st.caption(f"Last processed episode: `{ep}` at {ts}")

st.divider()

# -------------------------------------------------------------------
# LATEST EPISODE SUMMARY (cards)
# -------------------------------------------------------------------
st.subheader("Latest Valid Episode Summary")

if os.path.exists(PAIR_PATH):
    try:
        with open(PAIR_PATH, "r") as f:
            m = json.load(f)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Avg Speed (m/s)",
            f"{m['avg_speed_rl']:.3f}",
            f"{m['avg_speed_delta']:.2f}%",
            help="RL vs Baseline — positive is better",
        )
        c2.metric(
            "Avg Wait (s)",
            f"{m['avg_wait_rl']:.1f}",
            f"{m['avg_wait_delta']:.2f}%",
            help="RL vs Baseline — negative is better",
        )
        c3.metric(
            "Avg Queue (veh)",
            f"{m['avg_queue_rl']:.1f}",
            f"{m['avg_queue_delta']:.2f}%",
            help="RL vs Baseline — negative is better",
        )
    except Exception as e:
        st.warning(f"Could not read paired_metrics.json: {e}")
else:
    st.info("No paired_metrics.json found yet — run a valid episode first.")

st.divider()

# -------------------------------------------------------------------
# EPISODE TRENDS
# -------------------------------------------------------------------
st.subheader("Performance Trends Over Episodes")

if not os.path.exists(HIST_PATH) or os.path.getsize(HIST_PATH) == 0:
    st.info("No episode history found in episodes_history.csv.")
else:
    try:
        df = pd.read_csv(HIST_PATH)
    except Exception as e:
        st.warning(f"Could not read episodes_history.csv: {e}")
        df = None

    if df is not None and not df.empty:
        # Ensure numeric
        num_cols = [
            "baseline_speed",
            "rl_speed",
            "baseline_wait",
            "rl_wait",
            "baseline_queue",
            "rl_queue",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Index episodes sequentially for charts
        df = df.reset_index(drop=True)
        df["ep_idx"] = df.index + 1

        tab1, tab2, tab3 = st.tabs(["Speed", "Wait", "Queue"])

        with tab1:
            st.markdown("**Speed Trend (higher = better)**")
            speed_df = df[["ep_idx", "baseline_speed", "rl_speed"]].set_index("ep_idx")
            st.line_chart(speed_df)

        with tab2:
            st.markdown("**Wait Time Trend (lower = better)**")
            wait_df = df[["ep_idx", "baseline_wait", "rl_wait"]].set_index("ep_idx")
            st.line_chart(wait_df)

        with tab3:
            st.markdown("**Queue Length Trend (lower = better)**")
            queue_df = df[["ep_idx", "baseline_queue", "rl_queue"]].set_index("ep_idx")
            st.line_chart(queue_df)

        with st.expander("Raw Episode History"):
            st.dataframe(df[[
                "episode",
                "timestamp",
                "baseline_speed",
                "rl_speed",
                "delta_speed",
                "baseline_wait",
                "rl_wait",
                "delta_wait",
                "baseline_queue",
                "rl_queue",
                "delta_queue",
            ]])
    else:
        st.info("episodes_history.csv is empty or unreadable.")
