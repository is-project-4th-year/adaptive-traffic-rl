import { TrendingUp, TrendingDown } from 'lucide-react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Sparkline } from './Sparkline';

interface KPICardProps {
  title: string;
  value: string;
  unit: string;
  delta?: number;
  trend: 'up' | 'down';
  rlValue?: number;
  baselineValue?: number;
  sparklineData?: number[];
}

export function KPICard({
  title,
  value,
  unit,
  delta = 0,
  trend,
  rlValue = 0,
  baselineValue = 0,
  sparklineData = [],
}: KPICardProps) {
  // For speed and throughput, up is good. For wait time and queue, down is good.
  const isPositive =
    (trend === 'up' && (title.includes('Speed') || title.includes('Throughput'))) ||
    (trend === 'down' && (title.includes('Wait') || title.includes('Queue')));

  const safeDelta = Number.isFinite(delta) ? delta : 0;
  const safeRL = Number.isFinite(rlValue) ? rlValue : 0;
  const safeBaseline = Number.isFinite(baselineValue) ? baselineValue : 0;

  return (
    <Card className="p-6 border-[#E2E8F0] hover:shadow-lg hover:border-[#059669]/30 transition-all duration-200 bg-white">
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-[#64748B] mb-1">{title}</p>
          <div className="flex items-baseline gap-2">
            <span className="text-[#0F172A]">{value ?? "0.0"}</span>
            <span className="text-[#94A3B8]">{unit}</span>
          </div>
        </div>

        <Badge
          variant="outline"
          className={`gap-1 ${
            isPositive
              ? 'border-[#059669] text-[#059669] bg-[#DCFCE7]'
              : 'border-[#DC2626] text-[#DC2626] bg-[#FEE2E2]'
          }`}
          aria-label={`${isPositive ? 'Improved' : 'Decreased'} by ${Math.abs(safeDelta).toFixed(0)}%`}
        >
          {trend === 'up' ? (
            <TrendingUp className="size-3" />
          ) : (
            <TrendingDown className="size-3" />
          )}
          {Math.abs(safeDelta).toFixed(0)}%
        </Badge>
      </div>

      <div className="mb-4">
        <Sparkline data={sparklineData || []} color={isPositive ? '#059669' : '#64748B'} />
      </div>

      <div className="flex items-center justify-between pt-4 border-t border-[#E2E8F0]">
        <div>
          <p className="text-[#94A3B8] mb-1">RL</p>
          <p className="text-[#059669]">{Number(safeRL).toFixed(1)}</p>
        </div>
        <div className="text-right">
          <p className="text-[#94A3B8] mb-1">Baseline</p>
          <p className="text-[#64748B]">{Number(safeBaseline).toFixed(1)}</p>
        </div>
      </div>
    </Card>
  );
}

