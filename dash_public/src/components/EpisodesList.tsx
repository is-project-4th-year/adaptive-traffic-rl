
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { ArrowRight } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://40.120.26.11:8600";

// Matches EXACTLY what your backend now returns
interface EpisodeItem {
  episode: string;
  timestamp: string;
  seed: number | null;
  profile: string;
  duration_min: number;

  summary: {
    speed_rl: number;
    speed_base: number;
    speed_improvement_pct: number;

    delay_rl: number;
    delay_base: number;
    delay_improvement_pct: number;

    queue_rl: number;
    queue_base: number;
    queue_improvement_pct: number;
  };

  per_approach: Array<{
    approach: string;
    rlDelay: number;
    baselineDelay: number;
    rlQueue: number;
    baselineQueue: number;
  }>;
}

export function EpisodesList() {
  const [episodes, setEpisodes] = useState<EpisodeItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${API_BASE}/api/episodes/all`);
        const json = await res.json();

        if (Array.isArray(json)) {
          setEpisodes(json);
        } else {
          console.error("Invalid response:", json);
        }
      } catch (e) {
        console.error("Failed to load episodes list:", e);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  if (loading) {
    return <div className="p-8 text-[#6B7280]">Loading episodes…</div>;
  }

  return (
    <div className="p-8 max-w-[1440px] mx-auto">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-[#0F172A] text-xl font-semibold">Episodes</h1>

        <Link to="/episodes/latest">
          <Button variant="outline" className="gap-2">
            View Latest
            <ArrowRight className="size-4" />
          </Button>
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {episodes.map((ep) => {
          const s = ep.summary;

          return (
            <Card
              key={ep.episode}
              className="p-6 border-[#E2E8F0] hover:shadow-md hover:border-[#059669]/30 transition-all"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-[#64748B] text-sm mb-1">
                    {ep.timestamp || "—"}
                  </p>
                  <h2 className="text-[#0F172A] font-semibold">{ep.episode}</h2>
                </div>

                <Link to={`/episodes/${ep.episode}`}>
                  <Button size="sm" variant="outline" className="gap-2">
                    View
                    <ArrowRight className="size-4" />
                  </Button>
                </Link>
              </div>

              {/* KPIs */}
              <div className="grid grid-cols-3 gap-4 text-center">

                {/* Speed */}
                <div>
                  <p className="text-[#64748B] text-xs mb-1">Speed</p>
                  <p className="text-[#059669] font-semibold">
                    {s.speed_improvement_pct.toFixed(1)}%
                  </p>
                  <Badge
                    variant="outline"
                    className="text-[#059669] border-[#059669]"
                  >
                    RL ↑
                  </Badge>
                </div>

                {/* Wait */}
                <div>
                  <p className="text-[#64748B] text-xs mb-1">Wait</p>
                  <p className="text-[#DC2626] font-semibold">
                    {s.delay_improvement_pct.toFixed(1)}%
                  </p>
                  <Badge
                    variant="outline"
                    className="text-[#DC2626] border-[#DC2626]"
                  >
                    RL ↓
                  </Badge>
                </div>

                {/* Queue */}
                <div>
                  <p className="text-[#64748B] text-xs mb-1">Queue</p>
                  <p className="text-[#DC2626] font-semibold">
                    {s.queue_improvement_pct.toFixed(1)}%
                  </p>
                  <Badge
                    variant="outline"
                    className="text-[#DC2626] border-[#DC2626]"
                  >
                    RL ↓
                  </Badge>
                </div>

              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
