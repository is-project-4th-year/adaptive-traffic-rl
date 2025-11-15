import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Download, Circle } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { TimeSeriesChart } from './TimeSeriesChart';

interface LiveCompareProps {
  userRole: 'admin' | 'engineer' | 'viewer';
}

export function LiveCompare({ userRole }: LiveCompareProps) {
  const [isRunning, setIsRunning] = useState(true);
  const canControl = userRole !== 'viewer';

  const API_BASE = import.meta.env.VITE_API_BASE || "http://40.120.26.11:8600";

  // --- LIVE DATA STATE ---
  const [liveSeries, setLiveSeries] = useState<any[]>([]);
  const [queues, setQueues] = useState<any>({});
  const [phase, setPhase] = useState<any>({});
  const [actions, setActions] = useState<any[]>([]);
  const [episode, setEpisode] = useState<number | null>(null);
  const [windowTime, setWindowTime] = useState<string>("");

  // --- FETCH LIVE DATA ---
  async function fetchLive() {
    try {
      const res = await fetch(`${API_BASE}/api/live`);
      if (!res.ok) return;

      const json = await res.json();

      // Convert time series
      setLiveSeries(
        (json.series || []).map((p: any) => ({
          time: new Date(p.t * 1000).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
          }),
          rl: p.rl,
          baseline: p.baseline,
        }))
      );

      // Queues
      setQueues(json.queues || {});

      // Phase
      setPhase(json.phase || {});

      // Action log
      setActions(json.actions || []);

      // Episode #
      setEpisode(json.episode ?? null);

      // Time window
      setWindowTime(json.window ?? "");
    } catch (err) {
      console.error("LIVE FETCH ERROR:", err);
    }
  }

  // --- AUTO REFRESH EVERY 5s ---
  useEffect(() => {
    fetchLive();
    const t = setInterval(fetchLive, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="p-8 max-w-[1440px] mx-auto">
      
      {/* STATUS STRIP */}
      <div className="bg-white rounded-lg p-4 border border-[#E5E7EB] shadow-sm mb-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Live Indicator */}
          <div className="flex items-center gap-2">
            <Circle
              className={`size-3 ${
                isRunning ? "fill-[#2ECC71] text-[#2ECC71]" : "fill-[#6B7280] text-[#6B7280]"
              } ${isRunning ? "animate-pulse" : ""}`}
            />
            <span className={isRunning ? "text-[#2ECC71]" : "text-[#6B7280]"}>
              {isRunning ? "🟢 Live" : "Paused"}
            </span>
          </div>

          <div className="h-4 w-px bg-[#E5E7EB]" />

          {/* Time Window */}
          <span className="text-[#6B7280]">
            {windowTime || "—"}
          </span>

          <div className="h-4 w-px bg-[#E5E7EB]" />

          {/* Episode */}
          <Badge variant="outline">
            Episode #{episode ?? "—"}
          </Badge>

          <div className="h-4 w-px bg-[#E5E7EB]" />
          <span className="text-[#6B7280]">Peak</span>
          <div className="h-4 w-px bg-[#E5E7EB]" />
          <span className="text-[#6B7280]">Seed: 42</span>
        </div>

        {/* CONTROL BUTTONS */}
        {canControl && (
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsRunning(!isRunning)}
              className="gap-2 min-w-[44px] min-h-[44px]"
            >
              {isRunning ? <Pause className="size-4" /> : <Play className="size-4" />}
              {isRunning ? "Pause" : "Start"}
            </Button>

            <Button variant="outline" size="sm" className="gap-2 min-w-[44px] min-h-[44px]">
              <RotateCcw className="size-4" />
              Restart
            </Button>

            <Button variant="outline" size="sm" className="gap-2 min-w-[44px] min-h-[44px]">
              <Download className="size-4" />
              Export
            </Button>
          </div>
        )}
      </div>

      {/* MAIN CHART */}
      <Card className="p-6 border-[#E5E7EB] shadow-sm mb-6">
        <h3 className="text-[#1F2937] mb-6">Avg Delay (s) — Real-time Comparison</h3>
        <TimeSeriesChart data={liveSeries} />
      </Card>

      {/* THREE SMALL CARDS */}
      <div className="grid grid-cols-3 gap-6 mb-6">
        
        {/* QUEUE BY APPROACH */}
        <Card className="p-6 border-[#E5E7EB] shadow-sm">
          <h3 className="text-[#1F2937] mb-4">Queue by Approach</h3>
          <div className="space-y-4">
            {["N", "S", "E", "W"].map((dir) => {
              const rl = queues[dir] ?? 0;
              const baseline = Math.max(rl * 1.5, 1);

              return (
                <div key={dir}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[#6B7280]">
                      {dir === "N" ? "North" : dir === "S" ? "South" : dir === "E" ? "East" : "West"}
                    </span>
                    <div className="flex gap-3">
                      <span className="text-[#2ECC71]">{rl}</span>
                      <span className="text-[#4B5563]">{baseline.toFixed(0)}</span>
                    </div>
                  </div>

                  <div className="relative h-2 bg-[#E5E7EB] rounded-full overflow-hidden">
                    <div
                      className="absolute h-full bg-[#2ECC71] rounded-full"
                      style={{ width: `${(rl / 15) * 100}%` }}
                    />
                    <div
                      className="absolute h-full bg-[#4B5563] opacity-40 rounded-full"
                      style={{ width: `${(baseline / 15) * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* CURRENT PHASE */}
        <Card className="p-6 border-[#E5E7EB] shadow-sm">
          <h3 className="text-[#1F2937] mb-4">Current Phase</h3>
          <div className="text-center py-6">
            <div className="inline-flex items-center justify-center size-20 rounded-full bg-[#2ECC71]/10 mb-4">
              <Circle className="size-10 fill-[#2ECC71] text-[#2ECC71]" />
            </div>
            <p className="text-[#1F2937] mb-2">{phase.name ?? "—"}</p>
            <p className="text-[#6B7280]">
              {(phase.elapsed ?? 0)}s elapsed
            </p>
          </div>
        </Card>

        {/* ACTION LOG */}
        <Card className="p-6 border-[#E5E7EB] shadow-sm">
          <h3 className="text-[#1F2937] mb-4">Action Log</h3>
          <div className="space-y-3">
            {actions.map((log, i) => (
              <div key={i} className="pb-3 border-b border-[#E5E7EB] last:border-0 last:pb-0">
                <p className="text-[#9CA3AF] mb-1">{log.time}</p>
                <p className="text-[#1F2937] mb-1">{log.desc}</p>
                <p className="text-[#6B7280]">{log.reason}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* KEYBOARD SHORTCUTS */}
      {canControl && (
        <div className="p-4 bg-[#F9FAFB] rounded-lg border border-[#E5E7EB]">
          <p className="text-[#6B7280] mb-2">Keyboard shortcuts:</p>
          <div className="flex gap-6 text-[#6B7280]">
            <span>
              <kbd className="px-2 py-1 bg-white rounded border border-[#E5E7EB]">Space</kbd> Pause/Resume
            </span>
            <span>
              <kbd className="px-2 py-1 bg-white rounded border border-[#E5E7EB]">R</kbd> Restart
            </span>
            <span>
              <kbd className="px-2 py-1 bg-white rounded border border-[#E5E7EB]">E</kbd> Export
            </span>
            <span>
              <kbd className="px-2 py-1 bg-white rounded border border-[#E5E7EB]">Esc</kbd> Stop
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
