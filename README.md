

# 🚦 Adaptive Traffic Signal Control using Reinforcement Learning

An adaptive traffic light optimization system for Nairobi, Kenya — powered by **SUMO**, **Google Maps live data**, and a **Deep Q-Network (DQN)** agent.

---

## 🧠 Overview

This project dynamically adjusts traffic-light timing at major intersections using **real-time traffic data** and **reinforcement learning**.  
It runs two controllers in parallel:

* **Baseline Controller** — Fixed 30-second cycle benchmark
* **RL Controller** — Learns optimal light timing using a DQN policy

All performance metrics (speed, wait time, queue length, and actions) are logged and visualized in a **Streamlit dashboard**.

---

## 🏗️ System Architecture

**Components**

| Service                      | Description                                   | Path                                                                  |
| ---------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| gmaps-sensor.service         | Polls Google Routes API for live travel times | \~/traffic\_rl/services/                                              |
| policy-service.service       | TensorFlow/Keras API returning RL actions     | \~/traffic\_rl/services/policy\_[service.py](http://service.py)       |
| controller\_baseline.service | SUMO fixed-cycle baseline                     | \~/traffic\_rl/services/controller\_[baseline.py](http://baseline.py) |
| controller\_rl.service       | Reinforcement Learning controller (DQN)       | \~/traffic\_rl/services/controller\_[bridge.py](http://bridge.py)     |
| rl-day-runner.service        | Automates daily paired runs (baseline + RL)   | \~/traffic\_rl/scripts/day\_[runner.py](http://runner.py)             |
| dashboard                    | Streamlit app for visualization               | \~/traffic\_rl/dash\_public/[app.py](http://app.py)                   |

---

## 📊 Dashboard Access

Once deployed, view metrics and comparisons at: http://:8501

**Roles:**

| Role   | Username | Password   | Access                                   |
| ------ | -------- | ---------- | ---------------------------------------- |
| Admin  | admin    | admin123   | Full control (clear logs, force rollups) |
| Viewer | viewer   | traffic123 | Read-only dashboard access               |

The dashboard uses `streamlit-authenticator` with YAML-based credentials (`auth_config.yaml`).

---

## 🧮 Model Evaluation Metrics

| Metric                          | Description                          | Objective          |
| ------------------------------- | ------------------------------------ | ------------------ |
| **Average Wait Time (s)**       | Mean time vehicles remain stationary | ↓ Lower is better  |
| **Average Queue Length (veh)**  | Avg. vehicles waiting per lane       | ↓ Lower is better  |
| **Average Speed (m/s)**         | Overall vehicle speed in network     | ↑ Higher is better |
| **Network Throughput (veh/hr)** | Vehicles completing trips            | ↑ Higher is better |
| **Cumulative Reward**           | RL reward total per episode          | ↑ Higher is better |

These are logged automatically to `~/traffic_rl/logs/` and aggregated into `paired_summary.csv`.

---
