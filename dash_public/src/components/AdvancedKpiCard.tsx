// src/components/AdvancedKpiCard.tsx
import React from "react";
import { Info } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  YAxis,
} from "recharts";

interface BandPoint {
  x: number;
  median: number;
  p25: number;
  p75: number;
}

interface AdvancedKpiCardProps {
  title: string;
  description: string;
  unit: string;
  headlineValue: number;
  headlineLabel: string;
  reliability: number; // 0–100
  range: { min: number; max: number };
  rlLabel: string;
  baselineLabel: string;
  rlValue: number;
  baselineValue: number;
  rlUnit: string;
  baselineUnit: string;
  bandData: BandPoint[];
  invert?: boolean; // when more negative is good (wait, queue)
}

export const AdvancedKpiCard: React.FC<AdvancedKpiCardProps> = ({
  title,
  description,
  unit,
  headlineValue,
  headlineLabel,
  reliability,
  range,
  rlLabel,
  baselineLabel,
  rlValue,
  baselineValue,
  rlUnit,
  baselineUnit,
  bandData,
  invert,
}) => {
  const safeHeadline = Number.isFinite(headlineValue) ? headlineValue : 0;
  const rel = Math.round(reliability);

  const good = invert ? safeHeadline < 0 : safeHeadline > 0;
  const color = good ? "#059669" : "#DC2626";

  const formattedHeadline =
    unit === "%"
      ? safeHeadline.toFixed(1) + unit
      : safeHeadline.toFixed(1) + " " + unit;

  const rangeLabel =
    unit === "%"
      ? range.min.toFixed(1) + " to " + range.max.toFixed(1) + unit
      : range.min.toFixed(1) + "–" + range.max.toFixed(1) + " " + unit;

  const gradientId =
    "band-" + title.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9\-]/g, "");

  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] shadow-sm p-5 flex flex-col gap-4">
      {/* Title + Headline */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-[#9CA3AF]">
            {title}
          </p>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-semibold" style={{ color }}>
              {formattedHeadline}
            </span>
            <span className="text-xs text-[#6B7280]">{headlineLabel}</span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-xs text-[#6B7280] max-w-[180px] text-right">
          <Info className="w-3 h-3 text-[#9CA3AF]" />
          <span>{description}</span>
        </div>
      </div>

      {/* Reliability + range */}
      <div className="flex items-center justify-between text-xs text-[#6B7280]">
        <span>
          RL better in{" "}
          <span className="font-semibold text-[#111827]">{rel}%</span> of
          episodes
        </span>
        <span className="text-right">
          Range:{" "}
          <span className="font-medium text-[#111827]">{rangeLabel}</span>
        </span>
      </div>

      {/* Band sparkline */}
      <div className="h-20">
        <ResponsiveContainer>
          <AreaChart data={bandData}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#059669" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#059669" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
            <YAxis hide />
            {/* IQR-ish band */}
            <Area
              type="monotone"
              dataKey="p75"
              stroke="none"
              fill={ "url(#" + gradientId + ")" }
            />
            {/* Median line */}
            <Area
              type="monotone"
              dataKey="median"
              stroke={color}
              strokeWidth={2}
              fill="none"
              isAnimationActive={false}
            />
            <Tooltip
              formatter={(value: any) =>
                Number(value).toFixed(1) + " " + unit
              }
              labelFormatter={() => ""}
              contentStyle={{
                borderRadius: 8,
                borderColor: "#E5E7EB",
                fontSize: 12,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* RL vs Baseline snapshot */}
      <div className="flex items-center justify-between text-xs bg-[#F9FAFB] rounded-xl px-3 py-2">
        <div>
          <p className="text-[#6B7280]">{rlLabel}</p>
          <p className="font-medium text-[#111827]">
            {rlValue.toFixed(2)} {rlUnit}
          </p>
        </div>
        <div className="h-8 w-px bg-[#E5E7EB]" />
        <div className="text-right">
          <p className="text-[#6B7280]">{baselineLabel}</p>
          <p className="font-medium text-[#111827]">
            {baselineValue.toFixed(2)} {baselineUnit}
          </p>
        </div>
      </div>
    </div>
  );
};
