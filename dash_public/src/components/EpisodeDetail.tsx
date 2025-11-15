import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Download, ChevronRight } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { SlopeChart } from "./SlopeChart";
import { BoxPlotChart } from "./BoxPlotChart";
import { DataTable } from "./DataTable";
import { Link } from "react-router-dom";
const API_BASE = import.meta.env.VITE_API_BASE || "http://40.120.26.11:8600";

interface EpisodeSummary {
  delay_rl: number;
  delay_base: number;
  queue_rl: number;
  queue_base: number;
  speed_rl: number;
  speed_base: number;
  delay_improvement_pct: number;
  queue_improvement_pct: number;
  speed_improvement_pct: number;
}

interface PerApproachRow {
  approach: string;
  rlDelay: number;
  baselineDelay: number;
  rlQueue: number;
  baselineQueue: number;
}

interface EpisodePayload {
  episode: string;
  timestamp: string;
  seed: number | null;
  profile: string | null;
  duration_min: number | null;
  summary: EpisodeSummary;
  per_approach: PerApproachRow[];
}

export function EpisodeDetail() {
  const { episodeId } = useParams<{ episodeId: string }>();
  const [episode, setEpisode] = useState<EpisodePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const effectiveEndpoint =
    episodeId === "latest"
      ? `${API_BASE}/api/episodes/latest`
      : `${API_BASE}/api/episodes/${episodeId}`;

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(effectiveEndpoint);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`HTTP ${res.status}: ${text}`);
        }
        const json = (await res.json()) as EpisodePayload;
        if (!cancelled) {
          setEpisode(json);
        }
      } catch (e: any) {
        console.error("Failed to load episode:", e);
        if (!cancelled) {
          setError("Could not load episode details from API.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [effectiveEndpoint]);

  if (loading) {
    return (
      <div className="p-8 max-w-[1440px] mx-auto text-[#6B7280]">
        Loading episode details…
      </div>
    );
  }

  if (error || !episode) {
    return (
      <div className="p-8 max-w-[1440px] mx-auto">
        <p className="text-red-600 mb-2">{error ?? "Episode not found."}</p>
      </div>
    );
  }

  const epiIdLabel = episode.episode || (episodeId ?? "");
  const headerTimestamp = episode.timestamp ?? "";

  const kpiComparison = [
    {
      metric: "Avg Delay",
      rl: episode.summary.delay_rl,
      baseline: episode.summary.delay_base,
      unit: "s",
      improvement: episode.summary.delay_improvement_pct,
    },
    {
      metric: "Queue Length",
      rl: episode.summary.queue_rl,
      baseline: episode.summary.queue_base,
      unit: "veh",
      improvement: episode.summary.queue_improvement_pct,
    },
    {
      metric: "Avg Speed",
      rl: episode.summary.speed_rl,
      baseline: episode.summary.speed_base,
      unit: "m/s",
      improvement: episode.summary.speed_improvement_pct,
    },
  ];

  const approachData = episode.per_approach ?? [];

  return (
    <div className="p-8 max-w-[1440px] mx-auto">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 text-[#6B7280] mb-6">
      <Link to="/episodes" className="hover:text-[#1F2937] cursor-pointer">
        Episodes
      </Link>

        <ChevronRight className="size-4" />
        <span className="text-[#1F2937]">Episode {epiIdLabel}</span>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-[#1F2937] mb-2">
            Episode {epiIdLabel} — Deep Dive
          </h1>
          <div className="flex flex-wrap items-center gap-4 text-[#6B7280]">
            {headerTimestamp && <span>{headerTimestamp}</span>}
            <div className="h-4 w-px bg-[#E5E7EB]" />
            <span>Seed: {episode.seed ?? "—"}</span>
            <div className="h-4 w-px bg-[#E5E7EB]" />
            <span>Profile: {episode.profile ?? "Dynamic"}</span>
            <div className="h-4 w-px bg-[#E5E7EB]" />
            <span>
              Duration: {episode.duration_min ? `${episode.duration_min} min` : "10 min"}
            </span>
          </div>
        </div>
        <Button variant="outline" className="gap-2">
          <Download className="size-4" />
          Export Data
        </Button>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {kpiComparison.map((kpi) => (
          <Card key={kpi.metric} className="p-6 border-[#E5E7EB]">
            <p className="text-[#6B7280] mb-3">{kpi.metric}</p>
            <div className="space-y-2 mb-3">
              <div className="flex items-baseline gap-2">
                <span className="text-[#22C55E]">
                  {kpi.rl.toFixed(kpi.metric === "Avg Speed" ? 2 : 1)}
                </span>
                <span className="text-[#9CA3AF]">{kpi.unit}</span>
                <Badge
                  variant="outline"
                  className="ml-auto border-[#22C55E] text-[#22C55E]"
                >
                  RL
                </Badge>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-[#4B5563]">
                  {kpi.baseline.toFixed(kpi.metric === "Avg Speed" ? 2 : 1)}
                </span>
                <span className="text-[#9CA3AF]">{kpi.unit}</span>
                <Badge
                  variant="outline"
                  className="ml-auto border-[#4B5563] text-[#4B5563]"
                >
                  Base
                </Badge>
              </div>
            </div>
            <div className="pt-3 border-t border-[#E5E7EB]">
              <p className="text-[#059669]">
                {kpi.improvement.toFixed(1)}% better
              </p>
            </div>
          </Card>
        ))}
      </div>

      {/* Tabs */}
      <Tabs defaultValue="comparison" className="space-y-6">
        <TabsList>
          <TabsTrigger value="comparison">RL vs Baseline</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>

        <TabsContent value="comparison" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-6 border-[#E5E7EB]">
              <h3 className="text-[#1F2937] mb-6">Performance Gains</h3>
              <SlopeChart data={kpiComparison} />
            </Card>

            <Card className="p-6 border-[#E5E7EB]">
              <h3 className="text-[#1F2937] mb-6">Per-Approach Analysis</h3>
              <DataTable data={approachData} />
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="distribution">
          <Card className="p-6 border-[#E5E7EB]">
            <h3 className="text-[#1F2937] mb-6">Delay Distribution</h3>
            <BoxPlotChart />
          </Card>
        </TabsContent>

        <TabsContent value="trends">
          <Card className="p-6 border-[#E5E7EB]">
            <p className="text-[#6B7280]">
              Time-series trend analysis will appear here once we log per-step
              delay distributions.
            </p>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Notes */}
      <Card className="p-6 border-[#E5E7EB] mt-6">
        <h3 className="text-[#1F2937] mb-4">Episode Notes</h3>
        <ul className="space-y-2 text-[#6B7280]">
          <li>• No anomalies detected during simulation</li>
          <li>• 0 teleports recorded</li>
          <li>• Controller: DQN-v2.1.3, ε=0.05, lr=0.0001</li>
          <li>• Config & weather notes can be wired from API later</li>
        </ul>
      </Card>
    </div>
  );
}
