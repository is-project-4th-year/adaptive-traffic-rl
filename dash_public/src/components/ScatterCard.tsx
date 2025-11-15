// src/components/ScatterCard.tsx
import React from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PairedRow } from "./OverviewDashboard";

interface ScatterCardProps {
  title: string;
  xLabel: string;
  yLabel: string;
  unit: string;
  rows: PairedRow[];
  xAccessor: (r: PairedRow) => number;
  yAccessor: (r: PairedRow) => number;
  invert?: boolean;
}

export const ScatterCard: React.FC<ScatterCardProps> = ({
  title,
  xLabel,
  yLabel,
  unit,
  rows,
  xAccessor,
  yAccessor,
  invert,
}) => {
  const data = rows.map((r) => ({
    id: r.pair,
    x: xAccessor(r),
    y: yAccessor(r),
  }));

  const allX = data.map((d) => d.x);
  const allY = data.map((d) => d.y);
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const min = Math.min(minX, minY);
  const max = Math.max(maxX, maxY);

  const betterCount = data.filter((d) =>
    invert ? d.y < d.x : d.y > d.x
  ).length;

  const percentBetter = rows.length
    ? Math.round((betterCount / rows.length) * 100)
    : 0;

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-[#111827]">{title}</h3>
          <p className="text-xs text-[#6B7280] mt-1">
            Points {invert ? "below" : "above"} the diagonal mean RL outperforms
            the baseline.
          </p>
        </div>
        <div className="text-right text-xs">
          <p className="text-[#6B7280]">Episodes with RL better</p>
          <p className="font-semibold text-[#111827]">
            {betterCount}/{rows.length} ({percentBetter}%)
          </p>
        </div>
      </div>

      <div className="h-56">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid stroke="#F3F4F6" />
            <XAxis
              type="number"
              dataKey="x"
              name="Baseline"
              domain={[min * 0.95, max * 1.05]}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="RL"
              domain={[min * 0.95, max * 1.05]}
              tick={{ fontSize: 11 }}
            />
            {/* diagonal */}
            <Line
              type="monotone"
              dataKey="x"
              data={[
                { x: min, y: min },
                { x: max, y: max },
              ]}
              stroke="#E5E7EB"
              dot={false}
            />
            <Scatter
              name="Episodes"
              data={data}
              fill="#059669"
              shape="circle"
              opacity={0.8}
            />
            <Tooltip
              formatter={(value: any, name: string) =>
                `${Number(value).toFixed(2)} ${unit}`
              }
              labelFormatter={(_, idx) => data[idx]?.id ?? ""}
              contentStyle={{
                borderRadius: 8,
                borderColor: "#E5E7EB",
                fontSize: 12,
              }}
            />
            <Legend formatter={() => <span className="text-xs">Episodes</span>} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[11px] text-[#9CA3AF]">
        This chart is what convinces people: if most dots are on the{" "}
        {invert ? "lower" : "upper"} side of the line, RL is consistently
        better across episodes.
      </p>
    </div>
  );
};
