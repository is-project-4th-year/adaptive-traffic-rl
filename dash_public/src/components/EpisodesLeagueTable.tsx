// src/components/EpisodesLeagueTable.tsx
import React, { useMemo } from "react";
import type { PairedRow } from "./OverviewDashboard";

interface EpisodesLeagueTableProps {
  rows: PairedRow[];
}

function classifyBadge(waitRedPct: number) {
  const reduction = -waitRedPct; // negative = reduction
  if (reduction >= 40) return { label: "Excellent", color: "bg-emerald-100 text-emerald-700" };
  if (reduction >= 20) return { label: "Good", color: "bg-sky-100 text-sky-700" };
  if (reduction >= 5) return { label: "Neutral", color: "bg-slate-100 text-slate-700" };
  return { label: "Poor", color: "bg-rose-100 text-rose-700" };
}

export const EpisodesLeagueTable: React.FC<EpisodesLeagueTableProps> = ({ rows }) => {
  const { top, bottom } = useMemo(() => {
    const byWait = [...rows].sort((a, b) => a.wait_red_pct - b.wait_red_pct); // more negative = better
    return {
      top: byWait.slice(0, 5),
      bottom: byWait.slice(-3),
    };
  }, [rows]);

  function renderRow(r: PairedRow) {
    const badge = classifyBadge(r.wait_red_pct);
    const speed = r.speed_impr_pct;
    const wait = -r.wait_red_pct; // positive reduction
    const queue = -r.queue_red_pct;

    return (
      <tr key={r.pair} className="border-t border-[#E5E7EB]">
        <td className="py-2 pr-3 px-3 text-xs font-medium text-[#111827]">
          {r.pair}
        </td>
        <td className="py-2 pr-3 text-xs text-emerald-700">
          {speed.toFixed(1)}%
        </td>
        <td className="py-2 pr-3 text-xs text-emerald-700">
          {wait.toFixed(1)}%
        </td>
        <td className="py-2 pr-3 text-xs text-emerald-700">
          {queue.toFixed(1)}%
        </td>
        <td className="py-2 pr-3 text-xs">
          <span
            className={
              "inline-flex items-center rounded-full px-2 py-0.5 " + badge.color
            }
          >
            {badge.label}
          </span>
        </td>
      </tr>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5 h-full flex flex-col">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">
        Episodes league table
      </h3>
      <p className="text-xs text-[#6B7280] mb-4">
        Top 5 and bottom 3 episodes ranked by delay reduction (%). Use the
        Episodes page to deep-dive a specific run.
      </p>

      <div className="border border-[#E5E7EB] rounded-xl overflow-hidden bg-white">
        <table className="w-full border-collapse text-left text-xs">
          <thead className="bg-[#F9FAFB] text-[#6B7280]">
            <tr>
              <th className="py-2 px-3 font-medium">Episode</th>
              <th className="py-2 px-3 font-medium">Speed ↑</th>
              <th className="py-2 px-3 font-medium">Wait ↓</th>
              <th className="py-2 px-3 font-medium">Queue ↓</th>
              <th className="py-2 px-3 font-medium">Rating</th>
            </tr>
          </thead>
          <tbody>
            {top.map(renderRow)}
            {bottom.length > 0 && (
              <tr className="bg-[#F3F4F6]">
                <td colSpan={5} className="py-1 px-3 text-[10px] text-[#6B7280]">
                  Challenging runs (bottom 3 by wait reduction)
                </td>
              </tr>
            )}
            {bottom.map(renderRow)}
          </tbody>
        </table>
      </div>
    </div>
  );
};
