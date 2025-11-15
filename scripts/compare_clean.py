#!/usr/bin/env python3
"""
compare_clean.py — strict paired comparison of baseline vs RL KPIs

- Loads two KPI CSVs (baseline, rl)
- Rounds timestamps onto a 5s wall-clock grid
- STRICT inner join: only timestamps present in BOTH
- Filters out ticks where BOTH queues are zero
- Requires at least ~90% of expected ticks (600s / 5s = 120)
- Prints clean summary lines for parsing by paired_summary.py

If episode is invalid (too few paired ticks), exits non-zero and
prints an EPISODE_INVALID marker on stderr for the caller.
"""

import argparse
import os
import sys
import pandas as pd


STEP_SECONDS = 5
EPISODE_SECONDS = 600
EXPECTED_STEPS = EPISODE_SECONDS // STEP_SECONDS  # 120


def load_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")

    df = pd.read_csv(path)
    # Expect header: timestamp,avg_speed,avg_wait,queue_len,action
    required = {"timestamp", "avg_speed", "avg_wait", "queue_len"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    # Round timestamps to grid (e.g. 1763051060.0 → 1763051060)
    df["t_round"] = (df["timestamp"] // STEP_SECONDS) * STEP_SECONDS
    return df


def compare_clean(baseline_path: str, rl_path: str):
    b = load_file(baseline_path)
    r = load_file(rl_path)

    # Aggregate to one row per t_round
    b_agg = (
        b.groupby("t_round", as_index=False)[["avg_speed", "avg_wait", "queue_len"]]
        .mean()
    )
    r_agg = (
        r.groupby("t_round", as_index=False)[["avg_speed", "avg_wait", "queue_len"]]
        .mean()
    )

    # STRICT inner join on t_round only
    df = b_agg.merge(
        r_agg,
        on="t_round",
        how="inner",
        suffixes=("_b", "_r"),
    )

    # Enforce minimum number of paired ticks
    if len(df) < EXPECTED_STEPS * 0.90:  # < 108 for a 120-tick episode
        msg = (
            f"EPISODE_INVALID: only {len(df)} paired rows "
            f"(expected ~{EXPECTED_STEPS})"
        )
        print(msg, file=sys.stderr)
        sys.exit(2)

    # Drop rows where BOTH queues are zero (no vehicles anywhere)
    df_f = df[~((df["queue_len_b"] == 0) & (df["queue_len_r"] == 0))]

    if df_f.empty:
        msg = "EPISODE_INVALID: all paired rows are empty queue (no vehicles)"
        print(msg, file=sys.stderr)
        sys.exit(3)

    metrics = {}

    def pct_delta(bv: float, rv: float) -> float:
        if abs(bv) < 1e-6:
            return 0.0
        return (rv - bv) / bv * 100.0

    # Speed
    bs = df_f["avg_speed_b"].mean()
    rs = df_f["avg_speed_r"].mean()
    metrics["avg_speed_baseline"] = float(bs)
    metrics["avg_speed_rl"] = float(rs)
    metrics["avg_speed_delta"] = float(pct_delta(bs, rs))

    # Wait
    bw = df_f["avg_wait_b"].mean()
    rw = df_f["avg_wait_r"].mean()
    metrics["avg_wait_baseline"] = float(bw)
    metrics["avg_wait_rl"] = float(rw)
    metrics["avg_wait_delta"] = float(pct_delta(bw, rw))

    # Queue
    bq = df_f["queue_len_b"].mean()
    rq = df_f["queue_len_r"].mean()
    metrics["avg_queue_baseline"] = float(bq)
    metrics["avg_queue_rl"] = float(rq)
    metrics["avg_queue_delta"] = float(pct_delta(bq, rq))

    improvements = {
        "rows_used": int(len(df_f)),
    }

    return metrics, improvements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--rl", required=True)
    args = parser.parse_args()

    metrics, imp = compare_clean(args.baseline, args.rl)

    print("\n=== CLEAN KPI COMPARISON (Paired + Filtered) ===\n")
    print(f"Rows used: {imp['rows_used']}")
    print()
    print(
        f"Avg Speed: baseline={metrics['avg_speed_baseline']:.3f}, "
        f"rl={metrics['avg_speed_rl']:.3f}, "
        f"Δ={metrics['avg_speed_delta']:.2f}%"
    )
    print(
        f"Avg Wait:  baseline={metrics['avg_wait_baseline']:.1f}, "
        f"rl={metrics['avg_wait_rl']:.1f}, "
        f"Δ={metrics['avg_wait_delta']:.2f}%"
    )
    print(
        f"Avg Queue: baseline={metrics['avg_queue_baseline']:.1f}, "
        f"rl={metrics['avg_queue_rl']:.1f}, "
        f"Δ={metrics['avg_queue_delta']:.2f}%"
    )


if __name__ == "__main__":
    main()
