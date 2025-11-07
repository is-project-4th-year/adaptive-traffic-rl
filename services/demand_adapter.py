#!/usr/bin/env python3
import os, json, time
from datetime import datetime, timezone

GMAPS = os.environ.get("GMAPS_JSON", os.path.expanduser("~/traffic_rl/shared/live_speeds.json"))
OUT   = os.environ.get("FLOWS_ADD",  os.path.expanduser("~/traffic_rl/junctions/uhuru/flows.add.xml"))

# Map segment id -> incoming edge id in your net
EDGE_MAP = {
    "N_approach": "N_in",
    "S_approach": "S_in",
    "E_approach": "E_in",
    "W_approach": "W_in",
}

BASE_VPH = 700          # baseline vehicles/hour/edge
DELAY_REF = 120.0       # seconds for ~1x scaling
CLAMP = (0.5, 2.0)      # min/max scale

def clamp(x,a,b): return max(a, min(b, x))

def compute_scale(delay_s):
    # Simple: 1 + delay/DELAY_REF, clamped
    return clamp(1.0 + (delay_s or 0.0)/DELAY_REF, CLAMP[0], CLAMP[1])

def read_gmaps(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def main():
    data = read_gmaps(GMAPS)
    segs = (data or {}).get("segments", [])
    scales = {}

    for seg in segs:
        sid = seg.get("id")
        if not sid or "error" in seg: continue
        delay = seg.get("delay_s")
        scales[sid] = compute_scale(delay)

    # Build per-edge flows for a 2-hour window (0–7200s) with the scaled VPH
    # SUMO <flow> has vehsPerHour and departLane="free"
    lines = []
    lines.append('<additional>')
    for seg_id, edge in EDGE_MAP.items():
        scale = scales.get(seg_id, 1.0)
        vph   = int(BASE_VPH * scale)
        # unique id per direction
        lines.append(
            f'  <flow id="flow_{edge}" from="{edge}" to="J0" begin="0" end="7200" departLane="free" departSpeed="max" vehsPerHour="{vph}"/>'
        )
    lines.append('</additional>')
    xml = "\n".join(lines)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f: f.write(xml)

    now = datetime.now(timezone.utc).isoformat()
    print(f"[adapter] wrote {OUT} at {now}")

if __name__ == "__main__":
    main()
