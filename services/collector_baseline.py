#!/usr/bin/env python3
"""
collector_baseline.py — Baseline KPI collector (verbose version)
Polls the running SUMO baseline instance via TraCI (embedded or remote),
extracts live performance metrics, and writes to ~/traffic_rl/logs/kpi_baseline.csv
"""

import os
import sys
import csv
import time
import logging
import traci

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
LOG_DIR = os.path.expanduser("~/traffic_rl/logs")
os.makedirs(LOG_DIR, exist_ok=True)

OUT_LOG = os.path.join(LOG_DIR, "collector_baseline.out")
ERR_LOG = os.path.join(LOG_DIR, "collector_baseline.err")
KPI_FILE = os.path.join(LOG_DIR, "kpi_baseline.csv")

logging.basicConfig(
    filename=OUT_LOG,
    level=logging.INFO,
    format="%(asctime)s [collector] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
sys.stderr = open(ERR_LOG, "a", buffering=1)

SUMO_CFG = "/home/azureuser/traffic_rl/junctions/uhuru_baseline/live.sumocfg"
TLS_ID = "J0"
STEP_DELAY = 1.0  # seconds

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def ensure_csv():
    """Ensure CSV header exists."""
    if not os.path.exists(KPI_FILE) or os.path.getsize(KPI_FILE) == 0:
        with open(KPI_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "avg_speed", "avg_wait", "avg_queue", "phase"])
        logging.info(f"Initialized {KPI_FILE}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_collector():
    logging.info("=" * 70)
    logging.info("📈 Starting Collector Baseline")
    logging.info("=" * 70)
    ensure_csv()

    sumoBinary = "/usr/bin/sumo"
    sumoCmd = [
        sumoBinary,
        "-c", SUMO_CFG,
        "--step-length", "1.0",
        "--no-step-log",
        "--start",
        "--quit-on-end", "false",
    ]

    while True:
        try:
            logging.info("⚙️ Launching SUMO in collector mode...")
            traci.start(sumoCmd)
            logging.info("✅ Connected to SUMO successfully")

            edges = []
            try:
                links = traci.trafficlight.getControlledLinks(TLS_ID)
                lanes = sorted({l[0][0] for l in links if l and l[0]})
                edges = sorted({traci.lane.getEdgeID(l) for l in lanes})
                logging.info(f"Detected inbound edges: {edges}")
            except Exception as e:
                logging.warning(f"Could not detect edges: {e}")

            step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                step += 1

                try:
                    total_speed = total_wait = total_queue = 0
                    for e in edges:
                        total_speed += traci.edge.getLastStepMeanSpeed(e)
                        total_wait  += traci.edge.getWaitingTime(e)
                        total_queue += traci.edge.getLastStepHaltingNumber(e)
                    n = len(edges) or 1
                    avg_speed = total_speed / n
                    avg_wait = total_wait / n
                    phase = traci.trafficlight.getPhase(TLS_ID)

                    with open(KPI_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([time.time(), avg_speed, avg_wait, total_queue, phase])

                    if step % 60 == 0:
                        logging.info(f"step={step} | speed={avg_speed:.2f} | wait={avg_wait:.1f} | queue={total_queue} | phase={phase}")
                except Exception as e:
                    logging.warning(f"KPI read failed: {e}")

                time.sleep(STEP_DELAY)

            logging.info("✅ SUMO simulation ended cleanly, closing connection...")
            try:
                traci.close(False)
                logging.info("🧹 Closed TraCI cleanly.")
            except Exception as e:
                logging.warning(f"SUMO already terminated: {e}")

        except Exception as e:
            logging.error(f"💥 Collector crash: {e}")

        logging.info("♻️ Restarting collector in 5s...")
        time.sleep(5)

# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_collector()
    except KeyboardInterrupt:
        logging.info("🛑 Interrupted manually, exiting.")
        try:
            traci.close(False)
        except Exception:
            pass
        sys.exit(0)
