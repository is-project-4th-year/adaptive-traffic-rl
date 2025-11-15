// src/components/ReliabilityStrip.tsx
import React, { useMemo } from "react";
import type { PairedRow } from "./OverviewDashboard";

interface ReliabilityStripProps {
  rows: PairedRow[];
}

function classify(waitRedPct: number): "excellent" | "good" | "neutral" | "bad" {
  const reduction = -waitRedPct; // negative => RL better
  if (reduction >= 40) return "excellent";
  if (reduction >= 20) return "good";
  if (reduction >= 5) return "neutral";
  return "bad";
}

export const ReliabilityStrip: React.FC<ReliabilityStripProps> = ({ rows }) => {
  const classes = useMemo(
    () =>
      rows.map((r) => ({
        id: r.pair,
        cls: classify(r.wait_red_pct),
      })),
    [rows]
  );

  const counts = useMemo(() => {
    let excellent = 0;
    let good = 0;
    let neutral = 0;
    let bad = 0;
    for (const c of classes) {
      if (c.cls === "excellent") excellent += 1;
      else if (c.cls === "good") good += 1;
      else if (c.cls === "neutral") neutral += 1;
      else bad += 1;
    }
    return { excellent, good, neutral, bad };
  }, [classes]);

  if (!classes.length) return null;

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">
        Reliability across episodes
      </h3>
      <p className="text-xs text-[#6B7280] mb-3">
        Each square is an episode, colored by how much RL reduced delay.
      </p>

      <div className="flex flex-wrap gap-1 mb-3">
        {classes.map((c, idx) => {
          let bg = "bg-rose-200";
          if (c.cls === "excellent") bg = "bg-emerald-500";
          else if (c.cls === "good") bg = "bg-emerald-300";
          else if (c.cls === "neutral") bg = "bg-slate-200";

          return (
            <div
              key={c.id + "-" + idx}
              className={
                "w-3 h-3 rounded-[4px] " +
                bg +
                " border border-white shadow-sm"
              }
              title={c.id}
            />
          );
        })}
      </div>

      <div className="flex items-center justify-between text-[11px] text-[#6B7280]">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-[4px] bg-emerald-500" />{" "}
            Excellent ({counts.excellent})
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-[4px] bg-emerald-300" /> Good (
            {counts.good})
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-[4px] bg-slate-200" /> Neutral (
            {counts.neutral})
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-[4px] bg-rose-200" /> Worse (
            {counts.bad})
          </span>
        </div>
        <div className="text-right">
          Total episodes:{" "}
          <span className="font-medium text-[#111827]">
            {rows.length}
          </span>
        </div>
      </div>
    </div>
  );
};
