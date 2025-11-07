#!/usr/bin/env python3
"""
Baseline SUMO Controller (Embedded Mode)
- Fixed-cycle benchmark controller
- Logs KPIs on a WALL-CLOCK aligned grid (every LOG_INTERVAL seconds)
- Timestamp = epoch seconds (float)
- ONE_SHOT=1 archives KPI at the end of an episode
- WALL_DURATION_S: optional wall-clock cap to end by real time
"""

import os
import sys
import time
import logging
from pathlib import Path
import csv
import shutil

import traci

# ----------------------------------------------------------------------------- #
# CONFIG
# ----------------------------------------------------------------------------- #
LOG_DIR = os.path.expanduser("~/traffic_rl/logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Env-driven knobs
ONE_SHOT = os.getenv("ONE_SHOT", "0") == "1"
ARCHIVE_PREFIX = os.getenv("ARCHIVE_PREFIX", "baseline")
SUMO_BINARY = os.getenv("SUMO_BINARY", "/usr/bin/sumo")
SUMO_CFG = os.getenv("SUMO_CFG", "/home/azureuser/traffic_rl/junctions/uhuru_baseline/live.sumocfg")
SUMO_SEED = os.getenv("SUMO_SEED", "123")
KPI_FILE = os.getenv("KPI_FILE", os.path.join(LOG_DIR, "kpi_baseline.csv"))

TLS_ID = os.getenv("TLS_ID", "J0")
STEP_LENGTH = float(os.getenv("STEP_LENGTH", "1.0"))
CYCLE_TIME = float(os.getenv("CYCLE_TIME", "30"))  # seconds, fixed-cycle
GREEN_PHASES = [0, 2]

LOG_INTERVAL = float(os.getenv("LOG_INTERVAL", "5.0"))
SIM_DURATION = float(os.getenv("SIM_DURATION", "600"))
WALL_DURATION_S = float(os.getenv("WALL_DURATION_S", "0"))

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [baseline] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------------------------------------------------------------------- #
# UTILITIES
# ----------------------------------------------------------------------------- #
def ensure_kpi_header(path: str) -> None:
    """Ensure file exists but write no header (paired_rollup expects numeric rows)."""
    p = Path(path)
    if not p.exists():
        p.touch()


def archive_kpi() -> None:
    """Archive KPI and reset the live CSV (no header)."""
    try:
        src = Path(KPI_FILE)
        if src.exists() and src.stat().st_size > 0:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            dst = src.parent / f"kpi_{ARCHIVE_PREFIX}_{stamp}.csv"
            shutil.copy2(src, dst)
            src.write_text("")  # reset without header
            logging.info(f"Archived KPI → {dst} (live file reset)")
    except Exception as e:
        logging.warning(f"Could not archive KPI: {e}")


def _align_next(now: float, interval: float) -> float:
    """Return next epoch second aligned to interval grid (e.g., 0,5,10...)."""
    return (int(now // interval) + 1) * interval

# ----------------------------------------------------------------------------- #
# CORE LOOP
# ----------------------------------------------------------------------------- #
def run_episode() -> int:
    """Run one baseline SUMO episode with wall-clock KPI logging."""
    logging.info("=" * 70)
    logging.info("🚦 Starting Baseline Controller (Embedded SUMO Mode)")
    logging.info(f"Config: SUMO={SUMO_BINARY}, CFG={SUMO_CFG}, SEED={SUMO_SEED}")
    logging.info(
        f"Cycle={CYCLE_TIME}s, Step={STEP_LENGTH}s, TLS={TLS_ID}, "
        f"SimCap={SIM_DURATION:.0f}s, WallCap={WALL_DURATION_S:.0f}s, LogInterval={LOG_INTERVAL:.0f}s"
    )
    logging.info("=" * 70)

    ensure_kpi_header(KPI_FILE)

    sumo_cmd = [
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--no-step-log",
        "--step-length", str(STEP_LENGTH),
        "--start",
        "--quit-on-end", "true",
        "--seed", str(SUMO_SEED),
    ]

    try:
        logging.info("⚙️  Launching SUMO embedded via TraCI…")
        traci.start(sumo_cmd)
        logging.info("✅ SUMO launched. Connected to TraCI.")

        # Fixed-cycle control init
        phase_idx = GREEN_PHASES[0]
        next_switch_simt = CYCLE_TIME
        traci.trafficlight.setPhase(TLS_ID, phase_idx)
        logging.info(f"🟢 Initial phase set to {phase_idx} on {TLS_ID}")

        # Controlled inbound edges
        edges = []
        try:
            links = traci.trafficlight.getControlledLinks(TLS_ID)
            lanes = sorted({l[0][0] for l in links if l and l[0]})
            edges = sorted({traci.lane.getEdgeID(l) for l in lanes})
            logging.info(f"Inbound edges: {edges}")
        except Exception as e:
            logging.warning(f"Could not detect inbound edges: {e}")

        # Timing setup
        sim_start_wall = time.time()
        wall_deadline = sim_start_wall + WALL_DURATION_S if WALL_DURATION_S > 0 else None
        next_log_wall = _align_next(time.time(), LOG_INTERVAL)
        simt = 0.0
        empty_ticks = 0
        switch_since_last_log = 0

        # --------------------------------------------------------------------- #
        # MAIN LOOP
        # --------------------------------------------------------------------- #
        while True:
            traci.simulationStep()
            simt = traci.simulation.getTime()

            # WALL-TIME END CAP
            if wall_deadline and time.time() >= wall_deadline:
                logging.info(f"Reached WALL_DURATION_S={WALL_DURATION_S:.0f}s — ending baseline episode.")
                break

            # Graceful empty-traffic handling (don’t end instantly)
            if traci.simulation.getMinExpectedNumber() <= 0:
                empty_ticks += 1
            else:
                empty_ticks = 0
            if empty_ticks * STEP_LENGTH >= 5.0 and (time.time() - sim_start_wall) >= 20.0:
                logging.info("Traffic empty for 5s after warm-up — ending episode.")
                break

            # Fixed-cycle switching
            if simt >= next_switch_simt:
                i = GREEN_PHASES.index(phase_idx)
                phase_idx = GREEN_PHASES[(i + 1) % len(GREEN_PHASES)]
                traci.trafficlight.setPhase(TLS_ID, phase_idx)
                next_switch_simt += CYCLE_TIME
                switch_since_last_log = 1
                logging.info(f"🔁 Switched phase → {phase_idx} at simt={simt:.1f}s")

            # KPI logging on wall-clock
            now = time.time()
            while now >= next_log_wall:
                try:
                    total_queue = 0
                    total_wait = 0.0
                    total_speed = 0.0
                    n = 0

                    for e in edges:
                        total_queue += traci.edge.getLastStepHaltingNumber(e)
                        total_wait += traci.edge.getWaitingTime(e)
                        total_speed += traci.edge.getLastStepMeanSpeed(e)
                        n += 1

                    avg_speed = (total_speed / n) if n else 0.0
                    avg_wait = (total_wait / n) if n else 0.0

                    with open(KPI_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([next_log_wall, avg_speed, avg_wait, total_queue, switch_since_last_log])

                    logging.info(
                        f"📊 wall={next_log_wall:.0f} | simt={simt:.1f} | "
                        f"AvgSpeed={avg_speed:.2f} | Wait={avg_wait:.1f}s | "
                        f"Queue={total_queue} | action={switch_since_last_log}"
                    )
                    switch_since_last_log = 0
                except Exception as e:
                    logging.debug(f"KPI fetch skipped: {e}")

                next_log_wall += LOG_INTERVAL

            time.sleep(0.005)

        logging.info("✅ Simulation finished. Closing SUMO connection…")
        traci.close(False)
        elapsed = time.time() - sim_start_wall
        logging.info(f"🧹 SUMO closed. Wall time={elapsed:.1f}s, simt={simt:.1f}s")
        return 0

    except Exception as e:
        logging.error(f"💥 Fatal error in baseline loop: {e}", exc_info=True)
        try:
            traci.close(False)
        except Exception:
            pass
        return 1


# ----------------------------------------------------------------------------- #
# ENTRYPOINT
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    if ONE_SHOT:
        rc = run_episode()
        archive_kpi()
        sys.exit(rc)
    else:
        while True:
            try:
                rc = run_episode()
                archive_kpi()
            except KeyboardInterrupt:
                logging.info("🛑 Interrupted manually, shutting down baseline controller.")
                try:
                    traci.close(False)
                except Exception:
                    pass
                sys.exit(0)
            except Exception as e:
                logging.error(f"[baseline] ⚠️ Exception occurred: {e}", exc_info=True)

            logging.info("[baseline] 🔁 Episode finished — restarting in 5 seconds…")
            time.sleep(5)
