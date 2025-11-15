// src/components/ScatterQueueVsBaseline.tsx
import React, { useMemo } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";
import type { PairedRow } from "./OverviewDashboard";

interface Props {
  rows: PairedRow[];
}

export const ScatterQueueVsBaseline: React.FC<Props> = ({ rows }) => {
  const data = useMemo(
    () =>
      rows.map((r) => ({
        id: r.pair,
        baseline: r.baseline_queue,
        rl: r.rl_queue,
      })),
    [rows]
  );

  if (!data.length) return null;

  const xs = data.map((d) => d.baseline);
  const ys = data.map((d) => d.rl);
  const minV = Math.min(Math.min(...xs), Math.min(...ys));
  const maxV = Math.max(Math.max(...xs), Math.max(...ys));
  const padMin = Math.max(0, minV * 0.95);
  const padMax = maxV * 1.05;

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">
        Queue: RL vs baseline
      </h3>
      <p className="text-xs text-[#6B7280] mb-4">
        Points below the diagonal mean RL held shorter queues than baseline.
      </p>
      <div className="h-56">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid stroke="#F3F4F6" />
            <XAxis
              type="number"
              dataKey="baseline"
              name="Baseline queue"
              tickFormatter={(v) => v.toFixed(1)}
              domain={[padMin, padMax]}
              tick={{ fontSize: 11 }}
              label={{ value: "Baseline (veh)", position: "insideBottom", dy: 12 }}
            />
            <YAxis
              type="number"
              dataKey="rl"
              name="RL queue"
              tickFormatter={(v) => v.toFixed(1)}
              domain={[padMin, padMax]}
              tick={{ fontSize: 11 }}
              label={{
                value: "RL (veh)",
                angle: -90,
                position: "insideLeft",
                dx: -4,
              }}
            />
            <ReferenceLine
              segment={[
                { x: padMin, y: padMin },
                { x: padMax, y: padMax },
              ]}
              stroke="#E5E7EB"
              strokeDasharray="4 4"
            />
            <Scatter data={data} fill="#6366F1" opacity={0.85} />
            <Tooltip
              formatter={(value: any, name: string) =>
                name === "baseline" || name === "rl"
                  ? Number(value).toFixed(1) + " veh"
                  : value
              }
              labelFormatter={(_, idx) => data[idx]?.id || ""}
              contentStyle={{
                borderRadius: 8,
                borderColor: "#E5E7EB",
                fontSize: 12,
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
