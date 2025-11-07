#!/usr/bin/env python3
import os, glob, json
import pandas as pd

LOG_DIR = os.path.expanduser("~/traffic_rl/logs")
PAIR_CSV = os.path.join(LOG_DIR, "paired_summary.csv")
DAY_JSON = os.path.join(LOG_DIR, "paired_day.json")

def read_metrics(path):
    if not os.path.getsize(path):
        print(f"[skip] {os.path.basename(path)} (empty)")
        return None

    # handle optional header
    first = open(path).readline().strip().lower()
    skip = 1 if "timestamp" in first else 0
    df = pd.read_csv(
        path,
        header=None,
        names=["ts","avg_speed","avg_wait","queue_len","action"],
        skiprows=skip,
        on_bad_lines="skip"
    )

    # keep numeric rows only
    for c in ["ts","avg_speed","avg_wait","queue_len"]:
        df = df[pd.to_numeric(df[c], errors="coerce").notna()]

    if df.empty:
        print(f"[skip] {os.path.basename(path)} (no numeric rows)")
        return None

    # ---- Drop first minute (warm-up) ----
    try:
        t0 = float(df["ts"].min())
        df = df[df["ts"] >= (t0 + 60.0)]
    except Exception:
        pass

    if df.empty:
        print(f"[skip] {os.path.basename(path)} (only warm-up)")
        return None

    # summary
    s = {
        "rows": int(df.shape[0]),
        "avg_speed": round(float(df["avg_speed"].mean()), 4),
        "avg_wait":  round(float(df["avg_wait"].mean()), 2),
        "queue":     round(float(df["queue_len"].mean()), 2),
    }
    print(f"[ok]   {os.path.basename(path):<30}  rows={s['rows']}  s={s['avg_speed']}, w={s['avg_wait']}, q={s['queue']}")
    return s

# list archives
b_files = sorted(glob.glob(os.path.join(LOG_DIR, "kpi_baseline_*.csv")))
r_files = sorted(glob.glob(os.path.join(LOG_DIR, "kpi_rl_*.csv")))

pairs = list(zip(b_files, r_files))  # pair in chronological order
rows = []
for idx, (bf, rf) in enumerate(pairs, 1):
    bm = read_metrics(bf)
    rm = read_metrics(rf)
    if not bm or not rm:
        print(f"[pair {idx}] skipped (bad file)")
        continue
    row = {
        "pair": idx,
        "baseline_file": os.path.basename(bf),
        "rl_file": os.path.basename(rf),
        "baseline_speed": bm["avg_speed"],
        "rl_speed": rm["avg_speed"],
        "baseline_wait": bm["avg_wait"],
        "rl_wait": rm["avg_wait"],
        "baseline_queue": bm["queue"],
        "rl_queue": rm["queue"],
        "speed_impr_%": round((rm["avg_speed"]/bm["avg_speed"]-1)*100, 2) if bm["avg_speed"]>0 else 0.0,
        "wait_red_%":   round((1-rm["avg_wait"]/bm["avg_wait"])*100, 2)   if bm["avg_wait"]>0   else 0.0,
        "queue_red_%":  round((1-rm["queue"]/bm["queue"])*100, 2)         if bm["queue"]>0      else 0.0,
    }
    rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(PAIR_CSV, index=False)

    day = {
        "pairs": int(df.shape[0]),
        "avg_speed_baseline": round(float(df["baseline_speed"].mean()), 3),
        "avg_speed_rl":       round(float(df["rl_speed"].mean()), 3),
        "avg_wait_baseline":  round(float(df["baseline_wait"].mean()), 2),
        "avg_wait_rl":        round(float(df["rl_wait"].mean()), 2),
        "avg_queue_baseline": round(float(df["baseline_queue"].mean()), 2),
        "avg_queue_rl":       round(float(df["rl_queue"].mean()), 2),
        "speed_impr_%":       round(float((df["rl_speed"].mean()/df["baseline_speed"].mean()-1)*100) if df["baseline_speed"].mean()>0 else 0.0, 2),
        "wait_red_%":         round(float((1 - df["rl_wait"].mean()/df["baseline_wait"].mean())*100) if df["baseline_wait"].mean()>0 else 0.0, 2),
        "queue_red_%":        round(float((1 - df["rl_queue"].mean()/df["baseline_queue"].mean())*100) if df["baseline_queue"].mean()>0 else 0.0, 2),
    }
    with open(DAY_JSON, "w") as f:
        json.dump(day, f, indent=2)
    print(json.dumps(day, indent=2))
else:
    print("{}")
