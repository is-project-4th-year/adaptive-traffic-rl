interface SlopeChartProps {
  data: Array<{
    metric: string;
    baseline: number;
    rl: number;
    unit: string;
  }>;
}

export function SlopeChart({ data }: SlopeChartProps) {
  return (
    <div className="space-y-8">
      {data.map((item) => {
        const max = Math.max(item.baseline, item.rl);
        const baselinePercent = (item.baseline / max) * 100;
        const rlPercent = (item.rl / max) * 100;
        
        return (
          <div key={item.metric} className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[#6B7280]">{item.metric}</span>
              <span className="text-[#059669]">
                {((1 - item.rl / item.baseline) * 100).toFixed(1)}% ↓
              </span>
            </div>
            
            <div className="relative h-16">
              {/* Baseline line */}
              <div className="absolute left-0 right-0 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="size-3 rounded-full bg-[#4B5563]" />
                  <span className="text-[#4B5563]">{item.baseline} {item.unit}</span>
                </div>
              </div>
              
              {/* RL line */}
              <div className="absolute left-0 right-0 top-8 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="size-3 rounded-full bg-[#2ECC71]" />
                  <span className="text-[#2ECC71]">{item.rl} {item.unit}</span>
                </div>
              </div>
              
              {/* Connecting line */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <line
                  x1="6"
                  y1="6"
                  x2="6"
                  y2="38"
                  stroke="#E5E7EB"
                  strokeWidth="2"
                />
              </svg>
            </div>
          </div>
        );
      })}
    </div>
  );
}
