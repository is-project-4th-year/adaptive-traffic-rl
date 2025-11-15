#!/usr/bin/env python3
import os
import json
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE = Path.home() / "traffic_rl"
LOGS = BASE / "logs"
API = FastAPI(title="Traffic RL KPI API", version="1.1")

API.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # optionally restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def read_csv_safe(path: Path, limit: int = 50):
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path).tail(limit)
        return df.to_dict(orient="records")
    except Exception as e:
        return [{"error": str(e)}]

def read_json_safe(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------------------
# EXISTING ENDPOINTS
# ----------------------------------------------------------------------
@API.get("/api/kpi/live")
def kpi_live(limit: int = 50):
    """Return recent RL controller KPIs"""
    return read_csv_safe(LOGS / "kpi_live.csv", limit)

@API.get("/api/kpi/baseline")
def kpi_baseline(limit: int = 50):
    """Return recent baseline controller KPIs"""
    return read_csv_safe(LOGS / "kpi_baseline.csv", limit)

# ----------------------------------------------------------------------
# NEW ENDPOINTS FOR DASHBOARD INTEGRATION
# ----------------------------------------------------------------------
@API.get("/api/paired/summary")
def get_paired_summary():
    """Return full paired_summary.csv content"""
    path = LOGS / "paired_summary.csv"
    if not path.exists():
        return JSONResponse(content={"error": "paired_summary.csv not found"}, status_code=404)
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@API.get("/api/paired/day")
def get_paired_day():
    """Return daily aggregated KPIs from paired_day.json"""
    path = LOGS / "paired_day.json"
    if not path.exists():
        return JSONResponse(content={"error": "paired_day.json not found"}, status_code=404)
    with open(path) as f:
        data = f.read()
    return JSONResponse(content=json.loads(data))

# ----------------------------------------------------------------------
# COMPAT ALIASES (OLD ROUTES)
# ----------------------------------------------------------------------
@API.get("/api/pairs")
def pairs():
    return read_csv_safe(LOGS / "paired_summary.csv", 1000)

@API.get("/api/day")
def day():
    return read_json_safe(LOGS / "paired_day.json")

# ----------------------------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(API, host="0.0.0.0", port=8600)
