#!/usr/bin/env python3
# scripts/paired_runner.py
#
# One-shot 10-minute evaluation:
#   - runs baseline controller
#   - runs RL controller
#   - saves logs under episodes/EP_.../
#   - calls paired_rollup.py to compute summary.json

import os
import time
import json
import shutil
import pathlib
import argparse
import datetime as dt
import subprocess

ROOT = pathlib.Path("/home/azureuser/traffic_rl")
LOGS = ROOT / "logs"
EPISODES = ROOT / "episodes"

BASELINE_KPI = LOGS / "kpi_baseline.csv"
RL_KPI = LOGS / "kpi_live.csv"

DEFAULT_DURATION = 600        # seconds per controller (10 minutes)
GRACE = 5                     # extra seconds before stopping

def run_cmd(cmd, **kwargs):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)

def systemctl(action, unit):
    run_cmd(["sudo", "systemctl", action, unit])

def ensure_dirs():
    EPISODES.mkdir(parents=True, exist_ok=True)

def make_episode_id(seed: int | None, profile: str | None) -> str:
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bits = [f"EP_{now}"]
    if profile:
        bits.append(profile)
    if seed is not None:
        bits.append(f"seed{seed}")
    return "_".join(bits)

def run_controller(label: str, service: str, kpi_path: pathlib.Path,
                   duration: int):
    """Restart one controller, wait, then stop it."""
    # clear previous KPI file
    if kpi_path.exists():
        kpi_path.unlink()

    print(f"\n=== Running {label} for {duration}s ===")
    systemctl("restart", service)
    time.sleep(duration + GRACE)
    systemctl("stop", service)

    if not kpi_path.exists() or kpi_path.stat().st_size == 0:
        raise RuntimeError(f"{label} run produced no KPI file at {kpi_path}")

def main():
    p = argparse.ArgumentParser(description="Run 10-min baseline vs RL episode")
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                   help="Seconds per controller (default 600)")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional seed label for episode id")
    p.add_argument("--profile", type=str, default="eval",
                   help="Profile name to embed in episode id")
    args = p.parse_args()

    ensure_dirs()
    episode_id = make_episode_id(args.seed, args.profile)
    ep_dir = EPISODES / episode_id
    ep_dir.mkdir(parents=True, exist_ok=False)
    print(f"\n📦 Episode folder: {ep_dir}")

    # Optionally make sure gmaps sensor is alive
    try:
        systemctl("start", "gmaps-sensor")
    except Exception:
        print("⚠️ gmaps-sensor.service not started (ignoring)")

    # ---- 1. Baseline ----
    run_controller(
        label="baseline",
        service="controller_baseline",
        kpi_path=BASELINE_KPI,
        duration=args.duration,
    )
    shutil.copy(BASELINE_KPI, ep_dir / "baseline.csv")

    # ---- 2. RL (policy-service + controller_rl) ----
    # policy-service should be running before controller_rl
    try:
        systemctl("restart", "policy-service")
    except Exception:
        print("⚠️ policy-service restart failed, check manually.")

    run_controller(
        label="RL",
        service="controller_rl",
        kpi_path=RL_KPI,
        duration=args.duration,
    )
    shutil.copy(RL_KPI, ep_dir / "rl.csv")

    # ---- 3. Roll up and compute summary ----
    summary_path = ep_dir / "summary.json"
    print("\n=== Running paired_rollup.py ===")
    run_cmd([
        "python3",
        str(ROOT / "scripts" / "paired_rollup.py"),
        "--baseline", str(ep_dir / "baseline.csv"),
        "--rl", str(ep_dir / "rl.csv"),
        "--out", str(summary_path),
        "--episode-id", episode_id,
        "--duration", str(args.duration)
    ])

    print(f"\n✅ Episode complete: {episode_id}")
    print(f"   baseline.csv: {ep_dir / 'baseline.csv'}")
    print(f"   rl.csv:       {ep_dir / 'rl.csv'}")
    print(f"   summary.json: {summary_path}")

if __name__ == "__main__":
    main()
