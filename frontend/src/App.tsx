import { lazy, Suspense } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { Shell } from "./components/Shell";
import { ProjectsPage } from "./pages/ProjectsPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import { SettingsPage } from "./pages/SettingsPage";
import { PlaygroundPage } from "./pages/PlaygroundPage";
import { LabelingPage } from "./pages/LabelingPage";
import { AutopilotPage } from "./pages/AutopilotPage";
import { Spinner } from "./components/ui";

const AutopilotRunPage = lazy(() => import("./pages/AutopilotRunPage").then((m) => ({ default: m.AutopilotRunPage })));

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: false, retry: 1 } },
});

function LegacyRedirect() {
  const { pid = "", tab = "pipeline" } = useParams();
  return <Navigate to={`/experiments/${pid}/${tab}`} replace />;
}

export function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<ProjectsPage />} />
            <Route path="experiments/:pid" element={<Navigate to="pipeline" replace />} />
            <Route path="experiments/:pid/:tab" element={<WorkspacePage />} />
            {/* links from before the rename */}
            <Route path="projects/:pid/:tab?" element={<LegacyRedirect />} />
            <Route path="playground" element={<PlaygroundPage />} />
            <Route path="labelling" element={<LabelingPage />} />
            <Route path="labelling/:lsid" element={<LabelingPage />} />
            <Route path="autopilot" element={<AutopilotPage />} />
            <Route
              path="autopilot/runs/:runId"
              element={
                <Suspense fallback={<div className="p-6"><Spinner /></div>}>
                  <AutopilotRunPage />
                </Suspense>
              }
            />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
