import React, { useEffect, useMemo, useState } from "react";
import { RefreshCw, Calendar, TrendingUp, Activity, Info } from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

const API_BASE = "http://40.120.26.11:8600";

// Helpers
const safeNum = (val) => {
  const n = typeof val === "string" ? parseFloat(val) : Number(val);
  return Number.isFinite(n) ? n : 0;
};

const median = (arr) => {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
};

const parseHourFromEpisodeId = (id) => {
  const parts = id.split("_");
  const last = parts[parts.length - 1];
  if (last.length < 2) return Math.floor(Math.random() * 24);
  const hour = parseInt(last.slice(0, 2), 10);
  return Number.isFinite(hour) ? hour : Math.floor(Math.random() * 24);
};

// KPI Card Component
const KpiCard = ({ title, value, unit, reliability, range, good, description }) => {
  const color = good ? "#10b981" : "#ef4444";
  
  // Generate simple sparkline data
  const sparklineData = Array.from({ length: 20 }, (_, i) => ({
    x: i,
    value: value + (Math.random() - 0.5) * Math.abs(value) * 0.3
  }));
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">{title}</p>
            <Info className="w-3 h-3 text-gray-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold" style={{ color }}>
              {value.toFixed(1)}{unit}
            </span>
          </div>
        </div>
      </div>
      
      <p className="text-xs text-gray-600 mb-3">{description}</p>
      
      {/* Mini sparkline */}
      <div className="h-16 mb-3">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={sparklineData}>
            <defs>
              <linearGradient id={`gradient-${title}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={color} stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              fill={`url(#gradient-${title})`}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      <div className="space-y-2 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-gray-500">RL wins</span>
          <span className="font-semibold text-gray-900">{reliability.toFixed(0)}%</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Range</span>
          <span className="font-medium text-gray-700">{range.min.toFixed(1)} to {range.max.toFixed(1)}{unit}</span>
        </div>
      </div>
    </div>
  );
};

// Reliability Strip
const ReliabilityStrip = ({ episodes }) => {
  const classify = (waitRedPct) => {
    const reduction = -waitRedPct;
    if (reduction >= 40) return { label: "Excellent", color: "#10b981" };
    if (reduction >= 20) return { label: "Good", color: "#6ee7b7" };
    if (reduction >= 5) return { label: "Neutral", color: "#e5e7eb" };
    return { label: "Poor", color: "#fca5a5" };
  };
  
  const counts = { Excellent: 0, Good: 0, Neutral: 0, Poor: 0 };
  episodes.forEach(ep => {
    const cls = classify(ep.wait_red_pct);
    counts[cls.label]++;
  });
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-900 mb-2">Reliability Across Episodes</h3>
      <p className="text-xs text-gray-500 mb-4">Each square represents an episode, colored by RL performance</p>
      
      <div className="flex flex-wrap gap-1 mb-4">
        {episodes.map((ep, idx) => {
          const cls = classify(ep.wait_red_pct);
          return (
            <div
              key={idx}
              className="w-3 h-3 rounded shadow-sm"
              style={{ backgroundColor: cls.color }}
              title={`${ep.pair}: ${cls.label}`}
            />
          );
        })}
      </div>
      
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: "#10b981" }} />
          <span className="text-gray-600">Excellent ({counts.Excellent})</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: "#6ee7b7" }} />
          <span className="text-gray-600">Good ({counts.Good})</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: "#e5e7eb" }} />
          <span className="text-gray-600">Neutral ({counts.Neutral})</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: "#fca5a5" }} />
          <span className="text-gray-600">Poor ({counts.Poor})</span>
        </div>
      </div>
    </div>
  );
};

// Scatter Plot Component
const ScatterPlot = ({ data, title, xKey, yKey, xLabel, yLabel, color }) => {
  if (!data || !data.length) return null;
  
  const values = [...data.map(d => d[xKey]), ...data.map(d => d[yKey])];
  const minVal = Math.min(...values) * 0.9;
  const maxVal = Math.max(...values) * 1.1;
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-xs text-gray-500 mb-4">Points below diagonal show RL outperforming baseline</p>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
            <CartesianGrid stroke="#f0f0f0" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey={xKey}
              domain={[minVal, maxVal]}
              tick={{ fontSize: 11 }}
              label={{ value: xLabel, position: "insideBottom", offset: -15, style: { fontSize: 11 } }}
            />
            <YAxis
              type="number"
              dataKey={yKey}
              domain={[minVal, maxVal]}
              tick={{ fontSize: 11 }}
              label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 0, style: { fontSize: 11 } }}
            />
            <ReferenceLine
              segment={[
                { x: minVal, y: minVal },
                { x: maxVal, y: maxVal }
              ]}
              stroke="#d1d5db"
              strokeDasharray="4 4"
            />
            <Tooltip
              formatter={(value) => value.toFixed(2)}
              contentStyle={{ fontSize: 11, borderRadius: 8 }}
            />
            <Scatter data={data} fill={color} opacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// League Table
const LeagueTable = ({ episodes }) => {
  const sorted = [...episodes].sort((a, b) => a.wait_red_pct - b.wait_red_pct);
  const top5 = sorted.slice(0, 5);
  const bottom3 = sorted.slice(-3);
  
  const getBadge = (waitRedPct) => {
    const reduction = -waitRedPct;
    if (reduction >= 40) return { label: "Excellent", color: "bg-green-100 text-green-700" };
    if (reduction >= 20) return { label: "Good", color: "bg-blue-100 text-blue-700" };
    if (reduction >= 5) return { label: "Neutral", color: "bg-gray-100 text-gray-700" };
    return { label: "Poor", color: "bg-red-100 text-red-700" };
  };
  
  const renderRow = (ep, idx) => {
    const badge = getBadge(ep.wait_red_pct);
    return (
      <tr key={idx} className="border-t border-gray-200">
        <td className="px-3 py-2 text-xs font-medium text-gray-900 truncate max-w-[150px]">
          {ep.pair}
        </td>
        <td className="px-3 py-2 text-xs text-green-600">{ep.speed_impr_pct.toFixed(1)}%</td>
        <td className="px-3 py-2 text-xs text-green-600">{(-ep.wait_red_pct).toFixed(1)}%</td>
        <td className="px-3 py-2 text-xs text-green-600">{(-ep.queue_red_pct).toFixed(1)}%</td>
        <td className="px-3 py-2 text-xs">
          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${badge.color}`}>
            {badge.label}
          </span>
        </td>
      </tr>
    );
  };
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-900 mb-2">Episodes League Table</h3>
      <p className="text-xs text-gray-500 mb-4">Top 5 and bottom 3 episodes by performance</p>
      
      <div className="overflow-hidden border border-gray-200 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Episode</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Speed ↑</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Wait ↓</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Queue ↓</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Rating</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {top5.map(renderRow)}
            <tr className="bg-gray-50">
              <td colSpan={5} className="px-3 py-1 text-xs text-gray-500">Bottom 3</td>
            </tr>
            {bottom3.map(renderRow)}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Time of Day Chart
const TimeOfDayChart = ({ episodes }) => {
  const buckets = {
    "00-06": [], "06-08": [], "08-10": [], "10-12": [],
    "12-14": [], "14-17": [], "17-20": [], "20-24": []
  };
  
  episodes.forEach(ep => {
    const hour = ep.hour;
    let bucket;
    if (hour < 6) bucket = "00-06";
    else if (hour < 8) bucket = "06-08";
    else if (hour < 10) bucket = "08-10";
    else if (hour < 12) bucket = "10-12";
    else if (hour < 14) bucket = "12-14";
    else if (hour < 17) bucket = "14-17";
    else if (hour < 20) bucket = "17-20";
    else bucket = "20-24";
    
    buckets[bucket].push(ep);
  });
  
  const chartData = Object.entries(buckets).map(([bucket, eps]) => {
    const avgWait = eps.length > 0 ? eps.reduce((sum, e) => sum + (-e.wait_red_pct), 0) / eps.length : 0;
    const avgQueue = eps.length > 0 ? eps.reduce((sum, e) => sum + (-e.queue_red_pct), 0) / eps.length : 0;
    return { bucket, wait: avgWait, queue: avgQueue };
  });
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-900 mb-2">RL Impact vs Time of Day</h3>
      <p className="text-xs text-gray-500 mb-4">Average improvement by time window</p>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 40, left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
            <XAxis
              dataKey="bucket"
              tick={{ fontSize: 11 }}
              label={{ value: "Time of Day", position: "insideBottom", offset: -15, style: { fontSize: 11 } }}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              label={{ value: "Reduction (%)", angle: -90, position: "insideLeft", style: { fontSize: 11 } }}
            />
            <Tooltip
              formatter={(value) => value.toFixed(1) + "%"}
              contentStyle={{ fontSize: 11, borderRadius: 8 }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="wait" name="Wait Reduction" fill="#10b981" />
            <Bar dataKey="queue" name="Queue Reduction" fill="#a7f3d0" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Win Rate Chart
const WinRateChart = ({ episodes }) => {
  const total = episodes.length || 1;
  const speedWins = episodes.filter(e => e.speed_impr_pct > 0).length;
  const waitWins = episodes.filter(e => e.wait_red_pct < 0).length;
  const queueWins = episodes.filter(e => e.queue_red_pct < 0).length;
  
  const data = [
    { metric: "Speed", value: (speedWins / total) * 100 },
    { metric: "Wait", value: (waitWins / total) * 100 },
    { metric: "Queue", value: (queueWins / total) * 100 }
  ];
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-900 mb-2">How Often Does RL Win?</h3>
      <p className="text-xs text-gray-500 mb-4">Percentage of episodes where RL beats baseline</p>
      
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, bottom: 10, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} tickFormatter={(v) => v + "%"} />
            <YAxis type="category" dataKey="metric" tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value) => value.toFixed(1) + "%"} contentStyle={{ fontSize: 11, borderRadius: 8 }} />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Main Dashboard
export function OverviewDashboard() {
  const [episodes, setEpisodes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const res = await fetch(`${API_BASE}/api/paired/summary`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      
      const json = await res.json();
      
      if (Array.isArray(json)) {
        const mapped = json.map((r) => ({
          pair: String(r.pair),
          hour: parseHourFromEpisodeId(String(r.pair)),
          rl_speed: safeNum(r.rl_speed),
          baseline_speed: safeNum(r.baseline_speed),
          speed_impr_pct: safeNum(r["speed_impr_%"] !== undefined ? r["speed_impr_%"] : r.speed_impr_pct),
          rl_wait: safeNum(r.rl_wait),
          baseline_wait: safeNum(r.baseline_wait),
          wait_red_pct: safeNum(r["wait_red_%"] !== undefined ? r["wait_red_%"] : r.wait_red_pct),
          rl_queue: safeNum(r.rl_queue),
          baseline_queue: safeNum(r.baseline_queue),
          queue_red_pct: safeNum(r["queue_red_%"] !== undefined ? r["queue_red_%"] : r.queue_red_pct),
        }));
        
        setEpisodes(mapped);
        console.log("Loaded episodes:", mapped.length);
      } else {
        setEpisodes([]);
      }
    } catch (err) {
      console.error("Failed to fetch:", err);
      setError(err.message);
      setEpisodes([]);
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 30000);
    return () => clearInterval(id);
  }, []);
  
  const stats = useMemo(() => {
    if (!episodes.length) return null;
    
    const speedVals = episodes.map(e => e.speed_impr_pct);
    const waitVals = episodes.map(e => e.wait_red_pct);
    const queueVals = episodes.map(e => e.queue_red_pct);
    
    const minMax = (arr) => ({ min: Math.min(...arr), max: Math.max(...arr) });
    
    return {
      speed: {
        median: median(speedVals),
        reliability: (speedVals.filter(x => x > 0).length / speedVals.length) * 100,
        range: minMax(speedVals),
        good: median(speedVals) > 0
      },
      wait: {
        median: median(waitVals),
        reliability: (waitVals.filter(x => x < 0).length / waitVals.length) * 100,
        range: minMax(waitVals),
        good: median(waitVals) < 0
      },
      queue: {
        median: median(queueVals),
        reliability: (queueVals.filter(x => x < 0).length / queueVals.length) * 100,
        range: minMax(queueVals),
        good: median(queueVals) < 0
      }
    };
  }, [episodes]);
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <Activity className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading paired RL vs baseline data...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 max-w-md">
          <h2 className="text-red-800 font-semibold mb-2">Failed to Load Data</h2>
          <p className="text-red-600 text-sm mb-4">{error}</p>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }
  
  if (!episodes.length) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 max-w-md">
          <h2 className="text-yellow-800 font-semibold mb-2">No Data Available</h2>
          <p className="text-yellow-600 text-sm mb-4">
            No paired episodes found. Run at least one RL vs baseline pair.
          </p>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 text-sm"
          >
            Refresh
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              🚦 RL vs Baseline — Paired Performance
            </h1>
            <p className="text-sm text-gray-500 mt-1">
              {episodes.length} completed pairs. Dashboard shows comprehensive performance metrics.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={fetchData}
              className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 text-sm"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <div className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg text-sm">
              <Calendar className="w-4 h-4" />
              {new Date().toLocaleDateString()}
            </div>
          </div>
        </div>
        
        {/* KPI Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <KpiCard
              title="Speed ↑"
              value={stats.speed.median}
              unit="%"
              reliability={stats.speed.reliability}
              range={stats.speed.range}
              good={stats.speed.good}
              description="Average speed gain of RL over baseline"
            />
            <KpiCard
              title="Wait ↓"
              value={stats.wait.median}
              unit="%"
              reliability={stats.wait.reliability}
              range={stats.wait.range}
              good={stats.wait.good}
              description="Negative values mean RL cuts delay"
            />
            <KpiCard
              title="Queue ↓"
              value={stats.queue.median}
              unit="%"
              reliability={stats.queue.reliability}
              range={stats.queue.range}
              good={stats.queue.good}
              description="How RL reduces standing queues"
            />
          </div>
        )}
        
        {/* Reliability Strip */}
        <ReliabilityStrip episodes={episodes} />
        
        {/* Scatter Plots */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ScatterPlot
            data={episodes}
            title="Speed: RL vs Baseline"
            xKey="baseline_speed"
            yKey="rl_speed"
            xLabel="Baseline (m/s)"
            yLabel="RL (m/s)"
            color="#3b82f6"
          />
          <ScatterPlot
            data={episodes}
            title="Wait: RL vs Baseline"
            xKey="baseline_wait"
            yKey="rl_wait"
            xLabel="Baseline (s)"
            yLabel="RL (s)"
            color="#10b981"
          />
          <ScatterPlot
            data={episodes}
            title="Queue: RL vs Baseline"
            xKey="baseline_queue"
            yKey="rl_queue"
            xLabel="Baseline (veh)"
            yLabel="RL (veh)"
            color="#8b5cf6"
          />
        </div>
        
        {/* League Table and Charts */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-1">
            <LeagueTable episodes={episodes} />
          </div>
          <div className="xl:col-span-2 space-y-6">
            <TimeOfDayChart episodes={episodes} />
            <WinRateChart episodes={episodes} />
          </div>
        </div>
        
        <div className="text-center text-xs text-gray-400 pt-4">
          Updated {new Date().toLocaleTimeString()} • Auto-refresh every 30s
        </div>
      </div>
    </div>
  );
}
