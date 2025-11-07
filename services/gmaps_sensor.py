#!/usr/bin/env python3
"""
gmaps_sensor.py
---------------
Polls Google Maps Routes API v2 for live travel times on configured segments
and writes a compact JSON snapshot SUMO/your RL loop can read.

Requires:
  pip install requests tenacity
"""

import os, json, time, logging, tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

# --- Load Environment ---
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_MAPS_API_KEY not set")

API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
HEADERS = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": "routes.duration,routes.staticDuration,routes.distanceMeters"
}

CFG_PATH = os.path.expanduser("~/traffic_rl/config/segments.json")
OUT_PATH = os.path.expanduser("~/traffic_rl/shared/live_speeds.json")

# --- Pull settings from environment ---
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "300"))
OPERATE_START = os.getenv("OPERATE_START", "05:00")
OPERATE_END = os.getenv("OPERATE_END", "22:00")
DEPARTURE_OFFSET = int(os.getenv("GMAPS_DEPARTURE_OFFSET", "300"))
ROUTING_PREF = "TRAFFIC_AWARE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("segments", [])
    return cfg

def in_operating_window() -> bool:
    """Return True if current local time is within allowed hours."""
    now = datetime.now().time()
    start = datetime.strptime(OPERATE_START, "%H:%M").time()
    end = datetime.strptime(OPERATE_END, "%H:%M").time()
    return start <= now <= end

@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def query_segment(seg: Dict[str, Any], routing_pref: str) -> Dict[str, Any]:
    """Call Routes API for a single segment; returns JSON."""
    o = seg["origin"]; d = seg["dest"]

    payload = {
        "origin":      {"location": {"latLng": {"latitude": o["lat"], "longitude": o["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": d["lat"], "longitude": d["lng"]}}},
        "travelMode": "DRIVE",
        "routingPreference": routing_pref,
        "departureTime": (datetime.now(timezone.utc) + timedelta(seconds=DEPARTURE_OFFSET))
                            .isoformat().replace("+00:00", "Z"),
        "computeAlternativeRoutes": False,
        "languageCode": "en-US",
        "units": "METRIC",
    }

    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=15)
    
    
    # Log full error for diagnostics
    import logging
    logging.error(f"HTTP {r.status_code} body: {r.text}")
    r.raise_for_status()
    return r.json()

def to_seconds(iso_duration: str) -> float:
    return float(iso_duration.rstrip("s")) if isinstance(iso_duration, str) and iso_duration.endswith("s") else 0.0

def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path)) as tmp:
        json.dump(obj, tmp, separators=(",", ":"), ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def main():
    cfg = load_config()
    segs = cfg["segments"]

    logging.info(f"gmaps_sensor started: {len(segs)} segments | poll={POLL_INTERVAL}s | window={OPERATE_START}-{OPERATE_END} | pref={ROUTING_PREF}")

    while True:
        if not in_operating_window():
            logging.info("Outside operating window — sleeping until next check.")
            time.sleep(POLL_INTERVAL)
            continue

        start = time.time()
        out = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "routingPreference": ROUTING_PREF,
            "segments": []
        }

        for seg in segs:
            seg_id = seg["id"]
            try:
                resp = query_segment(seg, ROUTING_PREF)
                if not resp.get("routes"):
                    logging.warning(f"{seg_id}: no routes in response")
                    out["segments"].append({"id": seg_id, "error": "no_routes"})
                    continue

                route = resp["routes"][0]
                duration_s = to_seconds(route.get("duration", "0s"))
                base_s    = to_seconds(route.get("staticDuration", "0s"))
                delay_s   = max(duration_s - base_s, 0.0)
                dist_m    = int(route.get("distanceMeters", seg.get("distance_m", 0)) or 0)

                if "distance_m" in seg and seg["distance_m"] > 0:
                    dist_m = int(seg["distance_m"])

                speed_mps = (dist_m / duration_s) if duration_s > 0 else 0.0
                speed_kmh = speed_mps * 3.6

                logging.info(f"{seg_id} OK: distance={dist_m}m duration={duration_s:.1f}s static={base_s:.1f}s delay={delay_s:.1f}s speed={speed_kmh:.1f} km/h")

                out["segments"].append({
                    "id": seg_id,
                    "distance_m": dist_m,
                    "duration_s": round(duration_s, 1),
                    "static_duration_s": round(base_s, 1),
                    "delay_s": round(delay_s, 1),
                    "speed_mps": round(speed_mps, 2),
                    "speed_kmh": round(speed_kmh, 1)
                })

                time.sleep(1)

            except RetryError as e:
                logging.error(f"{seg_id}: final failure after retries: {e.last_attempt.exception()}")
                out["segments"].append({"id": seg_id, "error": str(e)})
            except Exception as e:
                logging.error(f"{seg_id}: unexpected error: {e}")
                out["segments"].append({"id": seg_id, "error": str(e)})

        atomic_write_json(OUT_PATH, out)

        took = time.time() - start
        sleep_left = max(0, POLL_INTERVAL - took)
        logging.info(f"cycle done in {took:.1f}s → sleeping {sleep_left:.0f}s")
        time.sleep(sleep_left)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("gmaps_sensor stopped by user")
