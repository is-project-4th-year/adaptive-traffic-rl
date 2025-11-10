

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

| Service                      | Description                                   |                                                                   
| ---------------------------- | --------------------------------------------- | 
| gmaps-sensor.service         | Polls Google Routes API for live travel times |                                              
| policy-service.service       | TensorFlow/Keras API returning RL actions     |
| controller\_baseline.service | SUMO fixed-cycle baseline                     | 
| controller\_rl.service       | Reinforcement Learning controller (DQN)       | 
| rl-day-runner.service        | Automates daily paired runs (baseline + RL)   | 
| dashboard                    | Streamlit app for visualization               |

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
# 🚦 Adaptive Traffic RL — Setup & Deployment

## 1) Clone the repo

```plaintext

git clone https://github.com/is-project-4th-year/adaptive-traffic-rl.git
cd adaptive-traffic-rl
```

## 2) Create virtual environment

```plaintext
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

## 3) Environment variables

```plaintext
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=/usr/share/sumo/tools
export GOOGLE_MAPS_API_KEY="your_api_key"

```

> Tip: put these in `~/.bashrc` or a `.env` and source it.

## 4) Start services (systemd)

```plaintext
sudo systemctl start policy-service
sudo systemctl start controller_rl
sudo systemctl start controller_baseline
sudo systemctl start rl-day-runner

```

Check status:

```plaintext
systemctl --no-pager --type=service | grep -E "policy|controller|runner"

```

## 5) Launch the dashboard

```plaintext
streamlit run dash_public/app.py --server.port 8501 --server.address 0.0.0.0

```

Open: `http://<vm-public-ip>:8501`

---

## 📁 Project layout

```plaintext
traffic_rl/
├─ dash_public/      # Streamlit dashboard (+ auth_config.yaml)
├─ services/         # Controllers, policy service, sensors
├─ scripts/          # Paired rollup, day runner, helpers
├─ junctions/        # SUMO networks & routes
├─ logs/             # KPI + paired summaries (runtime)
└─ models/           # Trained DQN models

```

## 🔒 Security

* Keep API keys and credentials out of git (`.gitignore` already covers `auth_config.yaml`, logs, models).
* Don’t commit massive SUMO detector outputs or CSV logs.

## 👥 Contributors

* Jeremy Wanjohi — System design, RL integration, dashboard  
##

