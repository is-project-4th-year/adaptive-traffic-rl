import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AppShell from "./components/AppShell";
import { OverviewDashboard } from "./components/OverviewDashboard";
import { LiveCompare } from "./components/LiveCompare";
import { EpisodeDetail } from "./components/EpisodeDetail";
import { AdminReports } from "./components/AdminReports";
import { EpisodesList } from "./components/EpisodesList";
import "./index.css";

function RootApp() {
  // temp: hardcoded role; later you can load from login
  const userRole: 'admin' | 'engineer' | 'viewer' = 'engineer';

  return (
    <BrowserRouter>
      <AppShell userRole={userRole} />

      {/* Main content */}
      <main className="ml-64 pt-16">
        <Routes>
          <Route path="/" element={<OverviewDashboard />} />

          <Route path="/live" element={<LiveCompare userRole={userRole} />} />

          {/* FIXED — Episodes list */}
          <Route path="/episodes" element={<EpisodesList />} />

          {/* Single episode */}
          <Route path="/episodes/:episodeId" element={<EpisodeDetail />} />

          {/* Admin */}
          {userRole === "admin" && (
            <Route path="/admin" element={<AdminReports userRole={userRole} />} />
          )}

          {/* fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(<RootApp />);
