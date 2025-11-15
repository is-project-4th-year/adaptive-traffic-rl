#!/usr/bin/env python3
# paired_rollup.py — FINAL FIXED VERSION
#
# Handles string → float conversion safely before filtering.
# Eliminates: TypeError: '<' not supported between instances of 'str' and 'float'

import argparse
import json
import pathlib
import pandas as pd

NUM_COLS = ["t", "speed", "wait", "queue", "action"]

def load_kpi(path: pathlib.Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} CSV missing or empty: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=NUM_COLS,
        dtype=str   # load everything as string first
    )

    # Convert all numeric columns safely
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Remove startup noise: speed=0 & queue=0 often means "no vehicles"
    df = df[~((df["speed"] == 0) & (df["queue"] == 0))]

    # Remove SUMO ghost spikes (13.9 m/s)
    df = df[df["speed"] < 13.5]

    return df.sort_values("t").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return dict(
            rows=0,
            avg_speed=0.0,
            avg_wait=0.0,
            avg_queue=0.0,
            throughput=0.0,
            t_start=None,
            t_end=None,
        )

    t_start = float(df["t"].min())
    t_end   = float(df["t"].max())
    dur     = max(t_end - t_start, 1.0)

    return dict(
        rows=int(len(df)),
        avg_speed=float(df["speed"].mean()),
        avg_wait=float(df["wait"].mean()),
        avg_queue=float(df["queue"].mean()),
        throughput=float(len(df) * 3600.0 / dur),
        t_start=t_start,
        t_end=t_end,
    )


def pct_gain(rl: float, base: float) -> float:
    if base == 0:
        return 0.0
    return float((rl - base) / base * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, help="Path to EP_xxx folder")
    args = ap.parse_args()

    ep_dir = pathlib.Path(args.episode_dir).resolve()
    b_path = ep_dir / "baseline.csv"
    r_path = ep_dir / "rl.csv"
    out    = ep_dir / "summary.json"

    print(f"[paired_rollup] Baseline CSV: {b_path}")
    print(f"[paired_rollup] RL CSV:       {r_path}")
    print(f"[paired_rollup] Output JSON:  {out}")

    df_b = load_kpi(b_path, "baseline")
    df_r = load_kpi(r_path, "rl")

    summ_b = summarize(df_b)
    summ_r = summarize(df_r)

    gains = dict(
        speed_pct       = pct_gain(summ_r["avg_speed"], summ_b["avg_speed"]),
        wait_pct        = pct_gain(summ_b["avg_wait"],  summ_r["avg_wait"]),
        queue_pct       = pct_gain(summ_b["avg_queue"], summ_r["avg_queue"]),
        throughput_pct  = pct_gain(summ_r["throughput"], summ_b["throughput"]),
    )

    summary = dict(
        episode_id=ep_dir.name,
        baseline=summ_b,
        rl=summ_r,
        gains=gains,
    )

    with out.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[paired_rollup] wrote summary.json for {ep_dir.name}")


if __name__ == "__main__":
    main()
