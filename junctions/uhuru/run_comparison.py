#!/usr/bin/env python3
import os, subprocess, sys, xml.etree.ElementTree as ET
HOME = os.path.expanduser("~/traffic_rl/junctions/uhuru")

NET  = f"{HOME}/cross.net.xml"
# CORRECTED: Use the file we validated: stress.rou.xml
ROU  = f"{HOME}/stress.rou.xml"
# CORRECTED: Only load detectors, as our TL logic is in cross.net.xml
ADD  = f"{HOME}/detectors.add.xml"

BASE_CFG = f"{HOME}/baseline.sumocfg"

def run(cmd, env=None):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

def sumo_base():
    # Runs the baseline.sumocfg, which we just fixed
    cmd = ["sumo", "-c", BASE_CFG,
           "--time-to-teleport", "-1",
           "--tripinfo-output", "baseline_trips.xml",
           "--emission-output", "baseline_emissions.xml"]
    run(cmd)

def sumo_dqn():
    py = f"{HOME}/rl/run_infer.py"
    env = os.environ.copy()
    env["SUMO_TRIPINFO"]   = "dqn_trips.xml"
    env["SUMO_EMISSIONS"]  = "dqn_emissions.xml"
    
    # CORRECTED: Must use the python from the venv
    py_exe = "/home/azureuser/traffic_rl/.venv/bin/python3"
    
    run([py_exe, py], env=env)

def parse_tripinfo(path):
    tree = ET.parse(path); root = tree.getroot()
    dlys, lens = [], []
    for v in root.findall(".//tripinfo"):
        dlys.append(float(v.get("waitingTime", "0")))
        lens.append(float(v.get("routeLength", "0")))
    avg_delay = sum(dlys)/len(dlys) if dlys else 0.0
    # As noted in the plan, queue length is a placeholder
    avg_q_len = 0.0  
    return avg_delay, avg_q_len

def parse_emissions(path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(path); root = tree.getroot()
    # SUMO emission-output: CO2 is mg/s per vehicle per timestep.
    # With 1s step, total_g = sum(mg/s * 1s) / 1000.
    total_mg = 0.0
    for v in root.findall(".//vehicle"):
        total_mg += float(v.get("CO2", "0"))
    return total_mg / 1000.0

    tree = ET.parse(path); root = tree.getroot()
    co2 = 0.0
    for v in root.findall(".//vehicle"):
        co2 += float(v.get("CO2", "0"))
    return co2

def table_row(scenario, controller, trips_xml, emis_xml):
    d, q = parse_tripinfo(trips_xml)
    co2  = parse_emissions(emis_xml)
    return [scenario, controller, round(d,2), round(q,2), round(co2,1)]

if __name__ == "__main__":
    print("--- Starting Phase 6: Evaluation ---")
    
    # 1) run baseline
    print("\n[1/2] Running Baseline (Fixed-Time) Simulation...")
    sumo_base()

    # 2) run DQN (inference)
    print("\n[2/2] Running DQN (RL Agent) Simulation...")
    sumo_dqn()

    # 3) print results table
    print("\n--- Generating Final Report ---")
import csv, matplotlib.pyplot as plt
out_csv = "/home/azureuser/traffic_rl/shared/final_report.csv"
out_png = "/home/azureuser/traffic_rl/shared/final_report.png"
    rows = [
      table_row("Stress Test", "Baseline", "baseline_trips.xml", "baseline_emissions.xml"),
      table_row("Stress Test", "DQN",      "dqn_trips.xml",      "dqn_emissions.xml"),
    ]
    hdr = ["Traffic Scenario","Controller","Avg. Delay (s)","Avg. Queue Length (m)","Total CO2 (g)"]
    
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    
    print("\n" + fmt.format(*hdr))
    print("-+-".join("-"*w for w in widths))
    for r in rows: print(fmt.format(*r))
    print("\n--- Evaluation Complete ---")
