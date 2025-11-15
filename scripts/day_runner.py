#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
day_runner.py — clean paired baseline vs RL runner (EPISODE-ONLY MODE)

Each episode:
  • Writes SHARED/pair_sync.json for exact alignment (start_at_epoch + step)
  • Clears live KPI files
  • Launches both controllers
  • Waits 10 minutes
  • Forces shutdown
  • Archives into episodes/EP_xxx:
        kpi_baseline.csv
        kpi_rl.csv
  • Runs:
        paired_summary.py <EPISODE_DIR>
        daily_rollup.py
"""

import os, sys, time, json, signal, shutil
import subprocess as sp
from pathlib import Path
from datetime import datetime, time as dt_time

# ------------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------------
ROOT = str(Path.home()) + "/traffic_rl"

VENV_PY       = f"{ROOT}/.venv/bin/python"
LOG_DIR       = f"{ROOT}/logs"
SERV_DIR      = f"{ROOT}/services"
SCRIPTS_DIR   = f"{ROOT}/scripts"
SHARED_DIR    = f"{ROOT}/shared"
EPISODES_DIR  = f"{ROOT}/episodes"
DASH_PUBLIC   = f"{ROOT}/dash_public"

BASELINE_PY   = f"{SERV_DIR}/controller_baseline.py"
RL_PY         = f"{SERV_DIR}/controller_bridge.py"

PAIR_SYNC_PATH   = f"{SHARED_DIR}/pair_sync.json"
KPI_BASELINE_LIVE = f"{LOG_DIR}/kpi_baseline.csv"
KPI_RL_LIVE       = f"{LOG_DIR}/kpi_live.csv"

PAIRED_SUMMARY = f"{SCRIPTS_DIR}/paired_summary.py"
DAILY_ROLLUP   = f"{SCRIPTS_DIR}/daily_rollup.py"

# ------------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------------
SERVICE_START = (5, 0)
SERVICE_END   = (23, 0)

EPISODE_LENGTH_S = 600      # 10 minutes
STEP_SECONDS     = 5        # must match LOG_INTERVAL
WARMUP_S         = 5
GRACE_KILL_S     = 15

# ------------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[day_runner] {ts} | {msg}", flush=True)


def svc_window():
    now = datetime.now().time()
    start = dt_time(*SERVICE_START)
    end   = dt_time(*SERVICE_END)
    return start <= now <= end


def ensure_dirs():
    for p in [LOG_DIR, SHARED_DIR, EPISODES_DIR, DASH_PUBLIC]:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_pair_sync(seed):
    now = time.time()
    aligned = int(now // STEP_SECONDS + 1) * STEP_SECONDS
    start_at_epoch = aligned + WARMUP_S

    pair_id = datetime.now().strftime("EP_%Y%m%d_%H%M%S")

    sync = {
        "pair_id": pair_id,
        "start_at_epoch": int(start_at_epoch),
        "step_seconds": STEP_SECONDS,
        "sumo_seed": seed
    }

    with open(PAIR_SYNC_PATH, "w") as f:
        json.dump(sync, f)

    log(f"pair_sync written: {sync}")
    return sync


def reset_kpis():
    header = "timestamp,avg_speed,avg_wait,queue_len,action\n"

    # baseline
    p_b = Path(KPI_BASELINE_LIVE)
    p_b.parent.mkdir(parents=True, exist_ok=True)
    p_b.write_text(header)

    # rl
    p_r = Path(KPI_RL_LIVE)
    p_r.parent.mkdir(parents=True, exist_ok=True)
    p_r.write_text(header)

    log("reset live KPI files (with header)")


def launch(seed):
    env = os.environ.copy()
    env.update({
        "SUMO_SEED": seed,
        "LOG_INTERVAL": str(STEP_SECONDS),

        # 🔥 NEW — traffic demand scaler (0.3 to 1.0 recommended)
        "DEMAND_SCALE" : "0.7",
    })
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    b_out = open(f"{LOG_DIR}/baseline_{ts}.out", "a", buffering=1)
    b_err = open(f"{LOG_DIR}/baseline_{ts}.err", "a", buffering=1)
    r_out = open(f"{LOG_DIR}/rl_{ts}.out", "a", buffering=1)
    r_err = open(f"{LOG_DIR}/rl_{ts}.err", "a", buffering=1)

    log("Launching controllers…")

    p_base = sp.Popen([VENV_PY, BASELINE_PY], stdout=b_out, stderr=b_err, env=env)
    p_rl   = sp.Popen([VENV_PY, RL_PY],       stdout=r_out, stderr=r_err, env=env)

    log(f"baseline PID={p_base.pid}, rl PID={p_rl.pid}")
    return p_base, p_rl


def wait_episode(start_at_epoch):
    hard_end = start_at_epoch + EPISODE_LENGTH_S
    wait_for = max(0, hard_end - time.time())
    log(f"Episode running… waiting {wait_for:.1f}s")
    time.sleep(wait_for)


def stop(p_base, p_rl):
    for name, p in [("baseline", p_base), ("rl", p_rl)]:
        if p.poll() is None:
            log(f"{name}: SIGTERM")
            p.terminate()

    deadline = time.time() + GRACE_KILL_S
    while time.time() < deadline:
        if p_base.poll() is not None and p_rl.poll() is not None:
            break
        time.sleep(0.2)

    for name, p in [("baseline", p_base), ("rl", p_rl)]:
        if p.poll() is None:
            log(f"{name}: SIGKILL")
            p.kill()

    log("controllers stopped")


def archive_kpis(pair_id):
    ep_dir = Path(EPISODES_DIR) / pair_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(KPI_BASELINE_LIVE, ep_dir / "kpi_baseline.csv")
    shutil.copy2(KPI_RL_LIVE,       ep_dir / "kpi_rl.csv")

    log(f"archived → {ep_dir}")

    Path(KPI_BASELINE_LIVE).write_text("")
    Path(KPI_RL_LIVE).write_text("")

    return ep_dir


def run_paired_summary(ep_dir):
    log(f"Running paired_summary on {ep_dir}")
    sp.run([VENV_PY, PAIRED_SUMMARY, str(ep_dir)], check=False)


def run_daily():
    log("Running daily_rollup.py")
    sp.run([VENV_PY, DAILY_ROLLUP], check=False)

# ------------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------------
def main_loop():
    ensure_dirs()
    log("day_runner started")

    while True:
        if not svc_window():
            log("Outside service window — sleeping 60s")
            time.sleep(60)
            continue

        seed = str(int(time.time()) // 3600)

        sync = write_pair_sync(seed)
        pair_id = sync["pair_id"]

        reset_kpis()

        # 🔥 NEW: update dynamic flows before starting SUMO
        log("Updating dynamic flows from Google Maps congestion…")
        sp.run(
            [VENV_PY, f"{SERV_DIR}/demand_adapter.py"],
            check=False
        )

        p_base, p_rl = launch(seed)

        wait_episode(sync["start_at_epoch"])

        stop(p_base, p_rl)

        ep_dir = archive_kpis(pair_id)

        run_paired_summary(ep_dir)

        run_daily()

        log("episode complete — sleeping 10s\n")
        time.sleep(10)

# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main_loop()
