// src/components/DistributionBars.tsx
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

interface DistributionBarsProps {
  rows: PairedRow[];
}

export const DistributionBars: React.FC<DistributionBarsProps> = ({ rows }) => {
  const data = useMemo(() => {
    const n = rows.length || 1;

    const speedWins = rows.filter((r) => r.speed_impr_pct > 0).length;
    const waitWins = rows.filter((r) => r.wait_red_pct < 0).length;
    const queueWins = rows.filter((r) => r.queue_red_pct < 0).length;

    return [
      {
        metric: "Speed",
        value: (speedWins / n) * 100,
      },
      {
        metric: "Wait",
        value: (waitWins / n) * 100,
      },
      {
        metric: "Queue",
        value: (queueWins / n) * 100,
      },
    ];
  }, [rows]);

  if (!data.length) return null;

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">
        How often does RL win?
      </h3>
      <p className="text-xs text-[#6B7280] mb-4">
        Percentage of episodes where RL beats baseline on each metric.
      </p>

      <div className="h-44">
        <ResponsiveContainer>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 10, right: 20, bottom: 10, left: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#F3F4F6" />
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => v.toFixed(0) + "%"}
            />
            <YAxis
              type="category"
              dataKey="metric"
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(value: any) => Number(value).toFixed(1) + "%"}
              contentStyle={{
                borderRadius: 8,
                borderColor: "#E5E7EB",
                fontSize: 12,
              }}
            />
            <Bar dataKey="value" name="RL wins" fill="#3B82F6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[11px] text-[#9CA3AF] mt-2">
        This condenses the story: if these bars are high, RL is reliably better
        than the fixed cycle.
      </p>
    </div>
  );
};
