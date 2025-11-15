#!/usr/bin/env python3
import json
from pathlib import Path

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
LIVE_SPEEDS = Path("/home/azureuser/traffic_rl/shared/live_speeds.json")

OUTPUT_RL = Path("/home/azureuser/traffic_rl/junctions/uhuru_rl/dynamic_flows.rou.xml")
OUTPUT_BASE = Path("/home/azureuser/traffic_rl/junctions/uhuru_baseline/dynamic_flows.rou.xml")

# 🔥 Episode duration
FLOW_END_TIME = 900  # 600s episode + 300s buffer

# 🔥 BALANCED BASE FLOWS for 2-lane intersection
# These numbers MUST be sustainable without gridlock
# Typical 2-lane capacity: ~700-800 vph per direction
# 🔥 BALANCED BASE FLOWS for 2-lane intersection
# Increased for sustained congestion during RL testing
BASE_VPH = {
    # HIGH STRESS test flows (~600 vph per approach)
    "N_to_S_tr": 400,
    "N_to_E_left": 100,
    "N_to_W_right": 100,

    "S_to_N_tr": 400,
    "S_to_W_left": 100,
    "S_to_E_right": 100,

    "E_to_W_tr": 400,
    "E_to_S_left": 100,
    "E_to_N_right": 100,

    "W_to_E_tr": 400,
    "W_to_N_left": 100,
    "W_to_S_right": 100,
}
# Connection mapping
GMAPS_TO_SUMO = {
    "N": ["N_to_S_tr", "N_to_E_left", "N_to_W_right"],
    "S": ["S_to_N_tr", "S_to_W_left", "S_to_E_right"],
    "E": ["E_to_W_tr", "E_to_S_left", "E_to_N_right"],
    "W": ["W_to_E_tr", "W_to_N_left", "W_to_S_right"],
}

# Congestion scaling limits
MIN_SCALE = 0.3    # don't go below 30% (floor)
MAX_SCALE = 1.5    # don't exceed 150% (prevent gridlock)

# SUMO max capacity per turning movement
MAX_VPH = 500

# ----------------------------------------------------------------------
# LOAD CONGESTION
# ----------------------------------------------------------------------
def load_congestion():
    """
    Loads segment-based congestion from Google Maps live_speeds.json.
    Returns direction ratios: {"N":x, "S":y, "E":z, "W":u}.
    """
    if not LIVE_SPEEDS.exists():
        return {"N":1.0, "S":1.0, "E":1.0, "W":1.0}

    d = json.loads(LIVE_SPEEDS.read_text())
    segs = d.get("segments", [])

    cong = {"N":1.0, "S":1.0, "E":1.0, "W":1.0}

    for seg in segs:
        seg_id = seg["id"]  # e.g. N_approach
        direction = seg_id.split("_")[0]

        dur = float(seg["duration_s"])
        ff = float(seg["static_duration_s"])

        if ff <= 0:
            ratio = 1.0
        else:
            ratio = dur / ff

        # clamp: keep between MIN_SCALE and MAX_SCALE to prevent overload
        cong[direction] = max(MIN_SCALE, min(ratio, MAX_SCALE))

    return cong


# ----------------------------------------------------------------------
# BUILD ROUTES XML
# Use period for continuous spawning + apply congestion scaling
# ----------------------------------------------------------------------
def build_xml(cong):
    lines = []
    lines.append("<routes>")
    lines.append('  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9" '
                 'laneChangeModel="LC2013" lcStrategic="0.6" lcCooperative="0.6" '
                 'lcSpeedGain="0.3" lcKeepRight="0"/>')
    lines.append("")

    for direction, flows in GMAPS_TO_SUMO.items():
        scale = cong[direction]

        for flow_id in flows:
            base_vph = BASE_VPH[flow_id]
            scaled_vph = int(base_vph * scale)

            # CAP to prevent gridlock
            final_vph = min(scaled_vph, MAX_VPH)

            # Convert vph to period (seconds between vehicles)
            period = 3600.0 / max(final_vph, 1)

            frm = flow_id.split("_")[0]   # N_to_S_tr → N
            to = flow_id.split("_")[2]    # N_to_S_tr → S

            lines.append(
                f'  <flow id="{flow_id}" from="{frm}_in" to="{to}_out" '
                f'begin="0" end="{FLOW_END_TIME}" period="{period:.2f}" type="car"/>'
            )

    lines.append("</routes>")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# WRITE TWO OUTPUT FILES
# ----------------------------------------------------------------------
def write_dynamic_flows(cong):
    xml = build_xml(cong)
    OUTPUT_RL.write_text(xml)
    OUTPUT_BASE.write_text(xml)
    print("✅ dynamic_flows.rou.xml updated (balanced demand)")
    print(f"   Congestion scaling: {cong}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    cong = load_congestion()
    write_dynamic_flows(cong)

if __name__ == "__main__":
    main()
