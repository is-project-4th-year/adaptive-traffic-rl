#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import subprocess

# =============================================================================
# AUTHENTICATION
# =============================================================================
with open("/home/azureuser/traffic_rl/dash_public/auth_config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Optional signup ---
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

if st.button("🆕 New user? Create account"):
    st.session_state.show_signup = True

if st.session_state.show_signup:
    st.subheader("Create a new account")
    new_user = st.text_input("Username")
    new_name = st.text_input("Full Name")
    new_email = st.text_input("Email")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if new_user in config["credentials"]["usernames"]:
            st.error("Username already exists.")
        else:
            hashed_pw = stauth.Hasher([new_pass]).generate()[0]
            config["credentials"]["usernames"][new_user] = {
                "email": new_email,
                "name": new_name,
                "password": hashed_pw,
                "role": "viewer"
            }
            with open("/home/azureuser/traffic_rl/dash_public/auth_config.yaml", "w") as f:
                yaml.dump(config, f)
            st.success("✅ Account created! Please log in.")
            st.session_state.show_signup = False

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login(location="main")
authentication_status = st.session_state["authentication_status"]
username = st.session_state.get("username")
name = st.session_state.get("name")


if authentication_status is False:
    st.error("❌ Incorrect username or password.")
    st.stop()
if authentication_status is None:
    st.warning("Please log in to continue.")
    st.stop()

# Authenticated
authenticator.logout("Logout", "sidebar")
role = config["credentials"]["usernames"][username].get("role", "viewer")
st.sidebar.success(f"✅ Logged in as {name} ({role})")

# =============================================================================
# CONFIG
# =============================================================================
LOG_DIR = os.path.expanduser("~/traffic_rl/logs")
PAIR_CSV = os.path.join(LOG_DIR, "paired_summary.csv")
DAY_JSON = os.path.join(LOG_DIR, "paired_day.json")

st.set_page_config(
    page_title="Traffic Signal — RL vs Baseline",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# ROLE CONTROL
# =============================================================================
if role == "admin":
    st.subheader("🧠 Admin Panel")
    st.write("You have full control — clear logs, trigger rollups, etc.")

    def audit_log(action):
        with open("/home/azureuser/traffic_rl/logs/audit.log", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {username} | {role} | {action}\n")

    if st.button("🧹 Clear KPI logs"):
        os.system("rm -f ~/traffic_rl/logs/kpi_*.csv")
        audit_log("Cleared KPI logs")
        st.success("✅ KPI logs cleared.")

    if st.button("🔁 Force paired rollup"):
        subprocess.run(
            ["python3", "/home/azureuser/traffic_rl/scripts/paired_rollup.py"],
            check=False,
        )
        audit_log("Triggered paired rollup")
        st.success("✅ Paired rollup completed successfully!")

elif role == "viewer":
    st.subheader("📊 Dashboard Viewer Mode")
    st.info("Read-only access — you can view metrics and charts but not modify data.")

# =============================================================================
# HEADER + REFRESH
# =============================================================================
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

st.markdown(
    """
    <div style="text-align:center;">
      <h1 style="margin-bottom:0.2rem;">🚦 Traffic Signal Optimization</h1>
      <p style="color:#666;margin-top:0;">Paired 10-minute simulations — Baseline vs RL</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("🔄 Refresh"):
            safe_rerun()
    with c2:
        st.caption(f"Data folder: `{LOG_DIR}`")
    with c3:
        st.caption("This page reads `paired_summary.csv` and optional `paired_day.json`.")

# =============================================================================
# LOAD DATA
# =============================================================================
def load_pairs(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return pd.DataFrame()
    df = pd.read_csv(p)
    num_cols = [
        "pair",
        "baseline_speed", "rl_speed",
        "baseline_wait", "rl_wait",
        "baseline_queue", "rl_queue",
        "speed_impr_%", "wait_red_%", "queue_red_%"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["pair"])
    df["pair"] = df["pair"].astype(int)
    return df.sort_values("pair").reset_index(drop=True)

def load_day(j):
    if not os.path.exists(j) or os.path.getsize(j) == 0:
        return None
    try:
        return pd.read_json(j, typ="series")
    except Exception:
        return None

pairs = load_pairs(PAIR_CSV)
day   = load_day(DAY_JSON)

if pairs.empty:
    st.warning("No paired episodes found yet. Run paired evaluations to populate `paired_summary.csv`.")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("Display Options")
metric_choices = ["Speed (m/s)", "Average Wait (s)", "Queue (veh)"]
selected_metrics = st.sidebar.multiselect(
    "Metrics to show",
    options=metric_choices,
    default=["Speed (m/s)", "Average Wait (s)", "Queue (veh)"]
)
show_distributions = st.sidebar.checkbox("Show distribution box plots", value=True)
show_table = st.sidebar.checkbox("Show per-episode table", value=False)

# =============================================================================
# KPI CARDS
# =============================================================================
def compute_day_fallback(df_pairs: pd.DataFrame) -> dict:
    d = {}
    d["pairs"] = int(df_pairs["pair"].nunique())
    d["avg_speed_baseline"] = float(df_pairs["baseline_speed"].mean())
    d["avg_speed_rl"]       = float(df_pairs["rl_speed"].mean())
    d["avg_wait_baseline"]  = float(df_pairs["baseline_wait"].mean())
    d["avg_wait_rl"]        = float(df_pairs["rl_wait"].mean())
    d["avg_queue_baseline"] = float(df_pairs["baseline_queue"].mean())
    d["avg_queue_rl"]       = float(df_pairs["rl_queue"].mean())
    d["speed_impr_%"] = float((df_pairs["rl_speed"].mean() / max(1e-9, df_pairs["baseline_speed"].mean()) - 1) * 100)
    d["wait_red_%"]   = float((1 - (df_pairs["rl_wait"].mean() / max(1e-9, df_pairs["baseline_wait"].mean()))) * 100)
    d["queue_red_%"]  = float((1 - (df_pairs["rl_queue"].mean() / max(1e-9, df_pairs["baseline_queue"].mean()))) * 100)
    return d

if day is None or day.empty:
    kpis = compute_day_fallback(pairs)
else:
    kpis = day.to_dict()

st.markdown("### 🧭 Daily Summary (paired episodes)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pairs", f"{int(kpis.get('pairs', pairs.shape[0]))}")
k2.metric("Speed ↑", f"{kpis.get('speed_impr_%', 0):.1f}%")
k3.metric("Wait ↓", f"{kpis.get('wait_red_%', 0):.1f}%")
k4.metric("Queue ↓", f"{kpis.get('queue_red_%', 0):.1f}%")

# =============================================================================
# PLOTS, DISTRIBUTIONS, TABLE
# =============================================================================
st.markdown("### 🎯 Per-Episode (10-min) Averages — RL vs Baseline")

def bar_pairs(df, metric_baseline, metric_rl, title, ylab):
    plot_df = df[["pair", metric_baseline, metric_rl]].copy()
    plot_df = plot_df.rename(columns={
        metric_baseline: "Baseline",
        metric_rl: "RL"
    })
    long_df = plot_df.melt(id_vars="pair", var_name="Controller", value_name=ylab)

    fig = px.bar(
        long_df,
        x="pair",
        y=ylab,
        color="Controller",
        barmode="group",
        title=title,
        category_orders={"Controller": ["Baseline", "RL"]},
        color_discrete_map={"Baseline": "#95a5a6", "RL": "#2ecc71"},
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

if "Speed (m/s)" in selected_metrics:
    bar_pairs(pairs, "baseline_speed", "rl_speed", "Average Speed per Episode", "Speed (m/s)")
if "Average Wait (s)" in selected_metrics:
    bar_pairs(pairs, "baseline_wait", "rl_wait", "Average Wait per Episode", "Average Wait (s)")
if "Queue (veh)" in selected_metrics:
    bar_pairs(pairs, "baseline_queue", "rl_queue", "Average Queue per Episode", "Queue (vehicles)")

# =============================================================================
# Improvement chart
# =============================================================================
st.markdown("### 📈 Improvement by Episode (positive = RL better)")
imp_cols = ["speed_impr_%", "wait_red_%", "queue_red_%"]
imp_names = {
    "speed_impr_%": "Speed Improvement (%)",
    "wait_red_%": "Wait Reduction (%)",
    "queue_red_%": "Queue Reduction (%)",
}
imp_long = pairs[["pair"] + imp_cols].melt(id_vars="pair", var_name="Metric", value_name="Percent")
imp_long["Metric"] = imp_long["Metric"].map(imp_names)

fig_imp = px.bar(
    imp_long,
    x="pair", y="Percent", color="Metric", barmode="group",
    title="RL vs Baseline — Per-Episode Improvement",
    color_discrete_map={
        "Speed Improvement (%)": "#2ecc71",
        "Wait Reduction (%)": "#3498db",
        "Queue Reduction (%)": "#f39c12",
    },
)
fig_imp.update_layout(template="plotly_white", height=380, margin=dict(l=40, r=20, t=60, b=40))
st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# DISTRIBUTIONS
# =============================================================================
if show_distributions:
    st.markdown("### 🔍 Distribution Across Episodes")
    c1, c2, c3 = st.columns(3)

    def box_two_series(a, b, title, ylab, ax):
        dd = pd.DataFrame({
            "value": np.r_[a.values, b.values],
            "Controller": ["Baseline"] * len(a) + ["RL"] * len(b),
        })
        fig = px.box(dd, x="Controller", y="value", points="outliers",
                     title=title, color="Controller",
                     color_discrete_map={"Baseline": "#95a5a6", "RL": "#2ecc71"})
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=10, t=60, b=20), yaxis_title=ylab)
        ax.plotly_chart(fig, use_container_width=True)

    with c1:
        box_two_series(pairs["baseline_speed"], pairs["rl_speed"], "Speed (m/s) — Distribution", "Speed (m/s)", st)
    with c2:
        box_two_series(pairs["baseline_wait"], pairs["rl_wait"], "Average Wait (s) — Distribution", "Avg. Wait (s)", st)
    with c3:
        box_two_series(pairs["baseline_queue"], pairs["rl_queue"], "Queue (veh) — Distribution", "Queue (veh)", st)

# =============================================================================
# TABLE
# =============================================================================
if show_table:
    st.markdown("### 📋 Per-Episode Table")
    pretty = pairs.copy()
    pretty = pretty[[
        "pair",
        "baseline_file", "rl_file",
        "baseline_speed", "rl_speed",
        "baseline_wait", "rl_wait",
        "baseline_queue", "rl_queue",
        "speed_impr_%", "wait_red_%", "queue_red_%"
    ]]
    st.dataframe(pretty, use_container_width=True, hide_index=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
f1, f2 = st.columns([3, 2])
with f1:
    st.caption("Data source: `paired_summary.csv` & `paired_day.json` (10-minute paired simulations).")
with f2:
    ts = time.ctime(os.path.getmtime(PAIR_CSV)) if os.path.exists(PAIR_CSV) else "—"
    st.caption(f"Last updated: {ts}")
