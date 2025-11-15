#!/usr/bin/env python3
"""
paired_summary.py — summarize ONE episode from episodes/ and update dashboard files.

Responsibilities:
- Auto-detect episode folder under ~/traffic_rl/episodes (or take it as an arg)
- Expect:
    <episode_dir>/kpi_baseline.csv
    <episode_dir>/kpi_rl.csv
- Call compare_clean.py for strict paired metrics
- On VALID episode:
    - Write dash_public/paired_metrics.json
    - Append ONE row to dash_public/episodes_history.csv (idempotent)
    - Write dash_public/last_status.json with status="ok"
- On INVALID episode:
    - Do NOT touch paired_metrics.json
    - Do NOT append to episodes_history.csv
    - Write dash_public/last_status.json with status="invalid"
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
ROOT = "/home/azureuser/traffic_rl"
COMPARE = f"{ROOT}/scripts/compare_clean.py"
DASH_DIR = f"{ROOT}/dash_public"
OUTFILE = f"{DASH_DIR}/paired_metrics.json"
HIST_PATH = f"{DASH_DIR}/episodes_history.csv"
STATUS_PATH = f"{DASH_DIR}/last_status.json"
EPISODES_DIR = f"{ROOT}/episodes"


def run_compare(baseline: str, rl: str):
    cmd = ["python3", COMPARE, "--baseline", baseline, "--rl", rl]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Check if this is an "episode invalid" case
        if "EPISODE_INVALID" in (result.stderr or ""):
            reason = result.stderr.strip()
            return None, reason
        print("Error running compare_clean.py:")
        print(result.stderr)
        sys.exit(1)

    return result.stdout.splitlines(), None


def parse_output(lines):
    """
    Parse the three 'Avg ...' lines from compare_clean.py into a dict:
    {
      "avg_speed_baseline": ...,
      "avg_speed_rl": ...,
      "avg_speed_delta": ...,
      "avg_wait_baseline": ...,
      ...
    }
    """
    data = {}

    for line in lines:
        line = line.strip()
        if not line.startswith("Avg"):
            continue

        # e.g. "Avg Speed: baseline=4.804, rl=6.342, Δ=32.02%"
        name, rest = line.split(":", 1)
        parts = rest.split(",")

        baseline_val = float(parts[0].split("=")[1])
        rl_val = float(parts[1].split("=")[1])
        delta_val = float(parts[2].split("=")[1].replace("%", ""))

        key = name.lower().replace(" ", "_")  # "Avg Speed" → "avg_speed"
        data[key + "_baseline"] = baseline_val
        data[key + "_rl"] = rl_val
        data[key + "_delta"] = delta_val

    return data


def write_status(status: str, episode_id: str, message: str = ""):
    os.makedirs(DASH_DIR, exist_ok=True)
    payload = {
        "status": status,          # "ok" | "invalid"
        "episode_id": episode_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message": message,
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def append_history(episode_id: str, data: dict):
    """
    Append a single row to episodes_history.csv if that episode_id
    is not already present.
    Columns:
    episode,timestamp,baseline_speed,rl_speed,delta_speed,
    baseline_wait,rl_wait,delta_wait,
    baseline_queue,rl_queue,delta_queue
    """
    os.makedirs(DASH_DIR, exist_ok=True)
    now_iso = datetime.now().isoformat(timespec="seconds")

    # Build history row
    row = [
        episode_id,
        now_iso,
        data["avg_speed_baseline"],
        data["avg_speed_rl"],
        data["avg_speed_delta"],
        data["avg_wait_baseline"],
        data["avg_wait_rl"],
        data["avg_wait_delta"],
        data["avg_queue_baseline"],
        data["avg_queue_rl"],
        data["avg_queue_delta"],
    ]

    # If file exists, check for existing episode
    existing_eps = set()
    if os.path.exists(HIST_PATH) and os.path.getsize(HIST_PATH) > 0:
        with open(HIST_PATH, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # header
                parts = line.strip().split(",")
                if parts and parts[0]:
                    existing_eps.add(parts[0])

    # Idempotency: don't write duplicates
    write_header = not os.path.exists(HIST_PATH) or os.path.getsize(HIST_PATH) == 0
    if episode_id in existing_eps:
        return

    with open(HIST_PATH, "a") as f:
        if write_header:
            f.write(
                "episode,timestamp,baseline_speed,rl_speed,delta_speed,"
                "baseline_wait,rl_wait,delta_wait,baseline_queue,rl_queue,delta_queue\n"
            )
        f.write(",".join(str(v) for v in row) + "\n")


def pick_episode_dir(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path)

    ep_root = Path(EPISODES_DIR)
    if not ep_root.exists():
        print(f"No episodes directory found at {EPISODES_DIR}")
        sys.exit(1)

    dirs = [d for d in ep_root.iterdir() if d.is_dir()]
    if not dirs:
        print(f"No episode subfolders found under {EPISODES_DIR}")
        sys.exit(1)

    # Pick latest by mtime
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    # Usage:
    #   paired_summary.py              → auto-pick latest episodes/EP_*
    #   paired_summary.py /path/to/EP → use that directory
    ep_arg = sys.argv[1] if len(sys.argv) > 1 else None
    ep_dir = pick_episode_dir(ep_arg)
    episode_id = ep_dir.name

    baseline = ep_dir / "kpi_baseline.csv"
    rl = ep_dir / "kpi_rl.csv"

    if not baseline.exists() or not rl.exists():
        msg = (
            f"EPISODE_INVALID: missing KPI CSVs in {ep_dir} "
            f"(looked for kpi_baseline.csv & kpi_rl.csv)"
        )
        print(msg, file=sys.stderr)
        write_status("invalid", episode_id, msg)
        sys.exit(2)

    lines, invalid_reason = run_compare(str(baseline), str(rl))

    if invalid_reason is not None:
        # Strict pairing failed or queues empty etc.
        print(f"Skipping episode {episode_id}: {invalid_reason}")
        write_status("invalid", episode_id, invalid_reason)
        # Do NOT touch paired_metrics.json or history
        sys.exit(0)

    data = parse_output(lines)

    # Write paired_metrics.json
    os.makedirs(DASH_DIR, exist_ok=True)
    with open(OUTFILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved paired metrics → {OUTFILE}")

    # Append to episodes_history.csv (idempotent)
    append_history(episode_id, data)

    # Mark status = ok
    write_status("ok", episode_id, "episode processed successfully")


if __name__ == "__main__":
    main()
