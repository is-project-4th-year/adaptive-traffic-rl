
---

# TrafficRL – Adaptive Reinforcement Learning for Traffic Signal Optimization

**Date:** November 2025
**Project:** Single-Junction RL Traffic Signal Optimization for Nairobi
**Models:** Fixed-Time Baseline Controller and Deep Q-Network (DQN) Agent

---

## 1. Executive Summary

This project implements an end-to-end traffic signal optimization system for a busy Nairobi intersection (Uhuru Highway × Haile Selassie roundabout) using:

* A fixed-time baseline controller representing the current status quo
* An adaptive Deep Q-Network (DQN)–based controller with safety constraints

The system covers:

* Network and demand generation in SUMO
* Real-world data ingestion from the Google Maps Routes API
* Online RL training with a 10-feature state and constrained action space
* Evaluation against a fixed-time baseline using interpretable KPIs
* A dashboard and API layer for inspection and potential deployment

In controlled single-junction simulations, the RL controller achieves (depending on demand profile) approximately:

* 20–25% higher average speed
* 30–40% lower average delay
* 30–40% shorter queues

compared to the fixed-time controller.

---

## 2. Approach

### 2.1 Environment and Data Pipeline

**Goal:**
Build a realistic digital twin of the Uhuru–Haile Selassie junction and feed it with plausible Nairobi traffic.

**Inputs and sources**

* **Road network**

  * SUMO `.net.xml` built from OpenStreetMap
  * Cleaned into a 4-arm junction:

    * North/South: Uhuru Highway
    * East/West: Haile Selassie Avenue

* **Demand approximation**

  * Origin–destination (OD) pairs derived from the Google Maps Routes API
  * Travel times sampled over peak and off-peak windows
  * Demand scaled into SUMO route files (`*.rou.xml`) with direction-specific flows

**State and KPI extraction (from SUMO via TraCI)**

* **Detector data**

  * Queue length per approach: `det_N_queue`, `det_S_queue`, `det_E_queue`, `det_W_queue`
  * Mean speed per approach: `det_N_in`, `det_S_in`, `det_E_in`, `det_W_in`

* **Signal phase data**

  * Current phase index (0–3)
  * Time spent in the current phase

**10-dimensional state vector**

```text
[
  phase_binary,        # 1.0 = EW green; 0.0 = NS green
  time_in_phase,       # seconds
  q_N, q_S, q_E, q_W,  # queue lengths
  v_N, v_S, v_E, v_W   # mean speeds
]
```

**Preprocessing and normalization**

* Z-score normalization with pre-computed `mean` and `scale` stored in `scaler_enhanced.json`
* NaN and invalid values replaced with 0 before normalization

**Splits and usage**

* No classic train/val/test CSV split
* Instead:

  * Multiple episodes with varying random seeds and flow patterns
  * Baseline vs RL runs use the same seeds for fair comparison

---

### 2.2 Control Architectures

#### Model 1: Fixed-Time Baseline Controller

* **Type:** Conventional fixed-cycle controller
* **Cycle:** ~90 seconds split between East–West and North–South phases
* **Behaviour:**

  * Static plan per time of day
  * No feedback from queues or speeds
  * No awareness of incidents, jams, or fluctuations

Used as the **reference controller** for all comparisons.

---

#### Model 2: DQN-Based RL Controller

**Architecture**

* Input: 10-dimensional normalized state

* Network:

  ```text
  Input(10) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(2, Linear)
  ```

* Output: Q-values for 2 actions:

  * `0`: favour East–West green
  * `1`: favour North–South green

**Training setup**

* Framework: TensorFlow / Keras
* Loss: Huber loss with Prioritized Experience Replay (PER)
* Optimizer: Adam (`lr = 5e-5`, gradient clipping)
* Discount factor: `γ = 0.90`
* Replay buffer: 200,000 transitions, PER with `α = 0.6` and `β` annealing
* Exploration: ε-greedy

  * `ε_start = 0.9 → ε_end = 0.05` with per-episode decay
* Target network: synced every 2,500 training steps
* Max episodes: 500+ (configurable)

**Traffic-aware constraints and logic**

* Decisions taken every `DECISION_INTERVAL_STEPS` simulation steps

* **Safety and realism constraints**

  * Never skip yellow states (phase sequence `0 → 1 → 2 → 3 → 0`)
  * Minimum green time: 12 seconds before any switch
  * Maximum green time: 60 seconds before forced check
  * Max switches per episode: capped to prevent thrashing

* **Queue hysteresis**

  * Only switch if one direction clearly dominates:

    * Absolute difference in queues > 1 vehicle, and/or
    * Relative difference > 15%

* **Queue pressure override**

  * If any approach’s queue is > ~8 vehicles over several steps
  * System is forced to serve it within a bounded time
  * Prevents starvation and hard gridlock

---

### 2.3 Reward and Evaluation Metrics

**Step reward**

At each decision step:

* `avg_speed = mean(v_N, v_S, v_E, v_W)`
* `max_queue = max(q_N, q_S, q_E, q_W)`
* `tp_delta =` new teleports (SUMO “stuck” vehicles resolved by teleport)

Reward:

```text
r = 0.3 * avg_speed
  - 0.1 * max_queue
  - 0.5 * tp_delta
```

* Reward clipped to `[-10, 10]`

**Episode-level KPIs (≈10-minute simulation window)**

* Average speed (km/h)
* Average delay / wait time (s/vehicle)
* Average and max queue length (vehicles)
* Number of phase switches
* Number of teleports
* RL vs baseline comparison per episode

**Aggregate metrics** (across multiple episodes)

* Mean KPIs over all episodes
* % of episodes where RL beats baseline on:

  * Average speed
  * Delay
  * Queue length
* Per-approach (N/S/E/W) metrics

---

## 3. Results Summary

### 3.1 Baseline vs RL Performance (Example Run)

Representative results (exact values vary with demand and seeds):

**Average speed (km/h)**

* Baseline: 10.4
* RL: 13.0
* Relative change: ≈ +25%

**Average wait time (s/vehicle)**

* Baseline: 15.0
* RL: 9.5
* Relative change: ≈ −37%

**Average queue length (vehicles)**

* Baseline: 6.3
* RL: 4.0
* Relative change: ≈ −36%

**Teleports per episode**

* Baseline: ≈ 2
* RL: ≈ 0–1 (reduced)

---

### 3.2 Per-Approach Performance (Sample Episode)

Sample per-approach delay comparison:

* **North**

  * Baseline: 28.5 s
  * RL: 16.5 s
  * Improvement: ≈ −42%

* **South**

  * Baseline: 26.0 s
  * RL: 16.1 s
  * Improvement: ≈ −38%

* **East**

  * Baseline: 22.0 s
  * RL: 18.0 s
  * Improvement: ≈ −18%

* **West**

  * Baseline: 24.0 s
  * RL: 17.5 s
  * Improvement: ≈ −27%

---

### 3.3 Key Findings

**Strengths**

* RL controller consistently improves:

  * Average speed
  * Delay
  * Queue length
* Fewer congestion collapses and smoother performance
* Safety constraints (min/max green, yellow phases, switch limits) keep behaviour realistic

**Weaknesses / limits**

* Gains depend on demand profile; not all episodes show the same improvement
* Training is sensitive to reward design and hyperparameters
* Current deployment is single-junction only (no corridor coordination yet)

---

## 4. Challenges Faced

### 4.1 Data and Environment Challenges

**Challenge 1: Realistic demand modeling**

* Issue:

  * Google Maps provides travel times, not lane-by-lane flows
* Solution:

  * Use OD pairs and travel-time samples as proxies
  * Scale into direction-based flows by time of day
* Impact:

  * Demand is approximate, not exact detector data

**Challenge 2: Calibration and validation**

* Issue:

  * No public detector dataset for this exact junction
* Solution:

  * Use ranges from literature and Nairobi congestion stats
  * Set reasonable target values for speeds and queues
* Impact:

  * Absolute numbers are approximate
  * Focus is on relative improvement vs baseline

---

### 4.2 RL and Model Challenges

**Challenge 1: Reward design**

* Issue:

  * Early reward versions led to skewed behaviour (e.g., over-favouring one approach)
* Solution:

  * Move to a stable reward combining `avg_speed`, `max_queue`, and teleports with interpretable weights
* Impact:

  * More stable training
  * Behaviour better aligned with human traffic engineering intuition

**Challenge 2: Signal logic and phase constraints**

* Issue:

  * Naive phase changes could skip yellow and break SUMO’s internal state
* Solution:

  * Strict phase order: `0 → 1 → 2 → 3 → 0`
  * Enforced min/max green times
* Result:

  * More realistic signalling
  * No phase desync
  * Fewer teleports

---

### 4.3 Hardware and Runtime Constraints

* Platform: Azure Ubuntu VM (CPU and/or GPU depending on size)
* Issue: SUMO + RL training is compute-heavy
* Solutions:

  * Run training headless (no GUI)
  * Reduce action frequency via decision intervals
  * Keep logging efficient to avoid I/O bottlenecks

---

### 4.4 Deployment Challenges

**Challenge: Continuous control and stability**

* Issue:

  * Live RL deployment needs stable, low-latency decisions and safe fallbacks
* Solutions:

  * Always keep a fixed-time baseline available as fallback
  * Enforce strict safety constraints in `apply_action`
  * Put policy behind an API for easy model swap/rollback

---

## 5. Production Improvements

### 5.1 Short-Term (1–2 Weeks)

**Scenario coverage**

* Add peak, off-peak, and incident scenarios
* Vary OD patterns and demand levels

**Hyperparameter sweeps**

* Learning rate
* Discount factor γ
* Reward weights
* Decision interval

**KPI and threshold tuning**

* Queue thresholds
* Queue pressure parameters

**Logging and monitoring**

* Standardize CSV/JSON logs for training, evaluation, and live runs
* Add health checks (no phase stuck beyond configured maximum)

---

### 5.2 Medium-Term (1–2 Months)

**Multi-junction coordination**

* Extend from single junction to a corridor with 2–4 signals
* Compare green-wave vs RL-based coordination

**Advanced RL**

* Dueling DQN, double DQN, distributional RL
* Multi-objective reward for delay, emissions, and bus priority

**Better calibration**

* Integrate any available local data (counts, studies, sensor logs)

**Explainability**

* Visualize actions vs phase, queue, and speed history
* Episode-level “why did it switch here?” analysis

---

### 5.3 Long-Term (3–6 Months)

**Active learning / continuous improvement**

* Use logs from live deployment or high-fidelity simulations
* Periodically retrain and update the policy

**City-scale simulation**

* Extend to a larger Nairobi subnet (corridor or small CBD grid)

**Fairness and policy**

* Study impact on different modes (buses vs cars)
* Consider pedestrian and public transport priority

**Scalable infrastructure**

* Containerized services (Docker)
* CI/CD and automated retraining pipelines

---

## 6. API Design and Deployment

### 6.1 Example Policy API Specification

**Endpoint (RL policy)**

`POST /api/policy/action`

**Example request**

```json
{
  "state": {
    "phase_binary": 1.0,
    "time_in_phase": 23.5,
    "q_N": 3,
    "q_S": 1,
    "q_E": 7,
    "q_W": 2,
    "v_N": 8.5,
    "v_S": 12.0,
    "v_E": 5.0,
    "v_W": 10.2
  }
}
```

**Example response**

```json
{
  "action": 0,
  "action_meaning": "favor_EW_green",
  "q_values": {
    "favor_EW": 0.87,
    "favor_NS": 0.42
  },
  "model_version": "dqn_v4_best",
  "timestamp_utc": "2025-11-24T10:15:23Z"
}
```

* `action = 0` → favour East–West green
* Controller logic enforces yellow, min/max green, and switch caps around this decision

---

### 6.2 Control Logic and Fallbacks

**High-level loop**

1. Read live state from SUMO or a field controller
2. Call RL policy API for suggested action
3. Apply action through constrained logic:

   * Respect min and max green
   * Never skip yellow
4. If RL API is down or state is invalid:

   * Fall back to fixed-time baseline

---

### 6.3 Deployment Stack

**Backend**

* Python services (RL policy, controller, data collector)
* FastAPI for REST

**Simulation**

* SUMO + TraCI (headless for training, GUI for debugging)

**Dashboard**

* Streamlit or React + Plotly (TrafficRL dashboard)

**Runtime**

* systemd services for background processes
* CSV / JSON logs for monitoring and analysis

**Future production**

* Docker + Kubernetes
* Prometheus + Grafana for central monitoring

---

## 7. Conclusion

This project delivers a complete pipeline for optimizing traffic signals at a key Nairobi junction using **Deep Reinforcement Learning**.

**Key achievements**

* Built a realistic SUMO environment for the Uhuru–Haile Selassie roundabout
* Designed and trained a DQN-based controller with strong safety constraints
* Compared RL vs fixed-time baseline using interpretable KPIs
* Wrapped the system with APIs and dashboards ready for inspection and extension

Despite approximate demand and lack of real detector data, the RL controller already shows consistent improvements in speed, delay, and queue length over the baseline in simulation.

**Next steps**

* Use richer, more accurate demand data where possible
* Extend from a single junction to multi-intersection corridors
* Move towards pilot-scale testing with city partners and real signal hardware

---

## 8. Appendix

### 8.1 Deliverables Checklist

* SUMO network and route files for Uhuru–Haile Selassie junction
* Data ingestion and preprocessing scripts (Google Maps → demand flows)
* Fixed-time baseline controller implementation
* DQN RL controller implementation with safety constraints
* Training script with PER, target network, and logging
* Evaluation scripts for baseline vs RL comparison
* API endpoints for policy inference and KPI retrieval
* Dashboard for visualizing KPIs and episode history
* Technical documentation (this report / README)
* Configuration files and model checkpoints

---

### 8.2 Example Repository Structure

```text
traffic_rl/
  junctions/
    uhuru/
      cross.net.xml          # Network
      base.rou.xml           # Base demand
      live.sumocfg           # Simulation config

  models/
    dqn_model_v4_latest.weights.h5
    dqn_model_v4_best.weights.h5

  logs/
    online_training_v4.csv   # Training curves
    online_eval_v4.csv       # Eval metrics
    episodes/                # Per-episode KPI archives

  shared/
    state.json               # Live state snapshots
    action.json              # Chosen actions (if used)

  src/
    online_train_v4_fixed_final.py   # RL training
    controller_baseline.py           # Fixed-time controller
    controller_rl.py                 # RL live controller
    policy_service.py                # FastAPI RL policy API

    data_pipeline/
      gmaps_sensor.py                # Google Maps Routes ingestion
      flow_builder.py                # Route/demand generation

    eval/
      compare_baseline_rl.py         # KPI comparison scripts
      plotting_utils.py

  dash_public/
    app.py                           # Dashboard entry
    ...                              # Frontend components

  configs/
    scaler_enhanced.json             # Normalization stats
    training_config.yaml             # Training config

  Makefile
  README.md
```

---

### 8.3 Computational Environment

* VM / OS: Ubuntu 22.04 LTS on Azure
* CPU/GPU: Depends on VM size; tested on CPU with optional GPU acceleration
* SUMO: v1.14+
* Python: 3.10+

**Key libraries**

* TensorFlow / Keras
* NumPy, Pandas
* TraCI (SUMO)
* FastAPI, Uvicorn
* Streamlit or React + Plotly

---

### 8.4 Time Investment (Illustrative)

Approximate breakdown:

* Network and demand setup: 2–3 hours
* RL model design and training script: 4–5 hours
* Experiments and evaluation: 3–4 hours
* Dashboard and API integration: 3–4 hours
* Documentation and cleanup: 2 hours

**Total:** ≈ 14–18 hours of focused work (excluding long training runs).

