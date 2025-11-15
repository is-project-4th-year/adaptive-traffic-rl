// src/components/ImprovementVsTime.tsx
import React, { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PairedRow } from "./OverviewDashboard";

interface ImprovementVsTimeProps {
  rows: PairedRow[];
}

function parseHourFromEpisodeId(id: string): number | null {
  // expects EP_YYYYMMDD_HHMMSS
  const parts = id.split("_");
  const last = parts[parts.length - 1];
  if (last.length < 2) return null;
  const hour = parseInt(last.slice(0, 2), 10);
  return Number.isFinite(hour) ? hour : null;
}

function bucketHour(hour: number | null): string {
  if (hour === null) return "Unknown";
  if (hour < 6) return "00–06";
  if (hour < 8) return "06–08";
  if (hour < 10) return "08–10";
  if (hour < 12) return "10–12";
  if (hour < 14) return "12–14";
  if (hour < 17) return "14–17";
  if (hour < 20) return "17–20";
  return "20–24";
}

export const ImprovementVsTime: React.FC<ImprovementVsTimeProps> = ({ rows }) => {
  const data = useMemo(() => {
    const buckets: {
      [bucket: string]: { wait: number[]; queue: number[] };
    } = {};

    for (const r of rows) {
      const hour = parseHourFromEpisodeId(r.pair);
      const key = bucketHour(hour);
      if (!buckets[key]) {
        buckets[key] = { wait: [], queue: [] };
      }
      buckets[key].wait.push(-r.wait_red_pct); // convert to positive reduction
      buckets[key].queue.push(-r.queue_red_pct);
    }

    const ordered = Object.keys(buckets).sort();
    return ordered.map((k) => {
      const b = buckets[k];
      const avg = (arr: number[]) =>
        arr.length ? arr.reduce((a, c) => a + c, 0) / arr.length : 0;
      return {
        bucket: k,
        wait: avg(b.wait),
        queue: avg(b.queue),
      };
    });
  }, [rows]);

  if (!data.length) return null;

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">
        RL impact vs time of day
      </h3>
      <p className="text-xs text-[#6B7280] mb-4">
        Average reduction in wait and queue for episodes starting in each time
        window.
      </p>
      <div className="h-52">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => v.toFixed(0) + "%"}
            />
            <Tooltip
              formatter={(value: any) => Number(value).toFixed(1) + "%"}
              contentStyle={{
                borderRadius: 8,
                borderColor: "#E5E7EB",
                fontSize: 12,
              }}
            />
            <Bar dataKey="wait" name="Wait reduction" fill="#10B981" />
            <Bar dataKey="queue" name="Queue reduction" fill="#A7F3D0" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-[#9CA3AF] mt-2">
        This shows whether RL is especially helpful at peak hours (bigger bars)
        or mainly smoothing mild congestion.
      </p>
    </div>
  );
};
