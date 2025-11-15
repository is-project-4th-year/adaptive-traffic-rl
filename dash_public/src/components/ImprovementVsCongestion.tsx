// src/components/ImprovementVsCongestion.tsx
import React, { useMemo } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Line,
} from "recharts";
import type { PairedRow } from "./OverviewDashboard";

interface ImprovementVsCongestionProps {
  rows: PairedRow[];
}

function safe(r: any, k1: string, k2?: string): number {
  if (r[k1] !== undefined) return Number(r[k1]) || 0;
  if (k2 && r[k2] !== undefined) return Number(r[k2]) || 0;
  return 0;
}

export const ImprovementVsCongestion: React.FC<
  ImprovementVsCongestionProps
> = ({ rows }) => {
  const usable = useMemo(
    () =>
      rows.filter(
        (r) =>
          typeof r.congestion_ratio === "number" && r.congestion_ratio > 0
      ),
    [rows]
  );

  if (!usable.length) {
    return (
      <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5">
        <h3 className="text-sm font-semibold text-[#111827] mb-1">
          RL impact vs congestion
        </h3>
        <p className="text-xs text-[#6B7280]">
          Congestion ratio is not available yet. Once added (Google Maps
          current/free-flow), this will show whether RL helps more during heavy
          traffic.
        </p>
      </div>
    );
  }

  // ---- FIXED DATA MAPPING ----
  const data = usable.map((r) => ({
    id: r.pair,
    x: r.congestion_ratio!,
    waitReduction: -safe(r, "wait_red_%", "wait_red_pct"), // NEGATIVE → positive reduction
  }));

  const xs = data.map((d) => d.x);
  const ys = data.map((d) => d.waitReduction);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">
        RL impact vs congestion level
      </h3>
      <p className="text-xs text-[#6B7280] mb-4">
        Each dot = one episode: congestion ratio (X) vs. wait reduction from RL
        (Y).
      </p>

      <div className="h-52">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid stroke="#F3F4F6" />

            <XAxis
              type="number"
              dataKey="x"
              name="Congestion ratio"
              tickFormatter={(v) => v.toFixed(2)}
              domain={[minX * 0.95, maxX * 1.05]}
              tick={{ fontSize: 11 }}
            />

            <YAxis
              type="number"
              dataKey="waitReduction"
              name="Wait reduction"
              tickFormatter={(v) => `${v.toFixed(0)}%`}
              domain={[minY * 0.95, maxY * 1.05]}
              tick={{ fontSize: 11 }}
            />

            <Scatter data={data} fill="#059669" opacity={0.85} />

            {/* zero-line guide */}
            <Line
              type="monotone"
              dataKey="zero"
              data={[
                { x: minX, zero: 0 },
                { x: maxX, zero: 0 },
              ]}
              stroke="#E5E7EB"
              dot={false}
            />

            <Tooltip
              formatter={(value: any, name: string) =>
                name === "x"
                  ? `${Number(value).toFixed(2)}x`
                  : `${Number(value).toFixed(1)}%`
              }
              labelFormatter={(_, idx) => data[idx]?.id ?? ""}
              contentStyle={{
                borderRadius: 8,
                borderColor: "#E5E7EB",
                fontSize: 12,
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[11px] text-[#9CA3AF] mt-2">
        Ideally the trend slopes upward: heavier congestion → larger RL benefit.
      </p>
    </div>
  );
};
