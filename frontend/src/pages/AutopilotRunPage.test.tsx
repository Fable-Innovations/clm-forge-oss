import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AutopilotRunPage } from "./AutopilotRunPage";
import { autopilotRoute } from "../testing/fixtures";
import { stubFetch } from "../testing/stubFetch";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderPage(runId: string) {
  const calls = stubFetch(autopilotRoute());
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[`/autopilot/runs/${runId}`]}>
        <Routes>
          <Route path="/autopilot/runs/:runId" element={<AutopilotRunPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { get urls() { return calls.map((c) => c.url); } };
}

describe("AutopilotRunPage", () => {
  it("renders a running run: header, progress, timeline by head, results, counts, journal", async () => {
    renderPage("run-running");
    await waitFor(() => expect(screen.getByRole("heading", { name: "Support ticket heads" })).toBeTruthy());
    const header = screen.getByLabelText("Run header");
    expect(within(header).getByText("running")).toBeTruthy();
    expect(within(header).getByText("on Qwen3-0.6B (MLX bf16)")).toBeTruthy();
    expect(within(header).getByText(/^3 of 5 heads · Generating rows for ticket-language \(round 1\) · 2 h 55 min elapsed · ~26 min left$/)).toBeTruthy();
    const timeline = screen.getByLabelText("Step timeline");
    expect(within(timeline).getByLabelText("ticket-language")).toBeTruthy();
    expect(within(timeline).getByText("Generating rows for ticket-language (round 1)").closest("li")!.getAttribute("data-state")).toBe("running");
    expect(within(timeline).getByText("Promoting ticket-escalation").closest("li")!.getAttribute("data-state")).toBe("pending");
    const table = screen.getByLabelText("Head results");
    const row = within(table).getByText("ticket-sentiment").closest("tr")!;
    expect(within(row).getByText("Kept")).toBeTruthy();
    expect(within(row).getByText("0.706")).toBeTruthy();
    expect(screen.getByLabelText("Row counts").textContent).toBe("Generated 4,800 rows · labelled 312 real rows · excluded 27");
    await waitFor(() => expect(screen.getAllByTestId("journal-entry").length).toBeGreaterThan(0));
  });

  it("polls while the run is running", async () => {
    const page = renderPage("run-running");
    await waitFor(() => expect(screen.getByLabelText("Run header")).toBeTruthy());
    const runCalls = () => page.urls.filter((u) => u.pathname === "/api/autopilot/runs/run-running").length;
    const first = runCalls();
    await waitFor(() => expect(runCalls()).toBeGreaterThan(first), { timeout: 5000 });
  }, 8000);

  it("a run from before progress was recorded says so, and a finished run stops polling", async () => {
    const page = renderPage("run-old");
    await waitFor(() => expect(screen.getByText("Progress not recorded for this run.")).toBeTruthy());
    const n = page.urls.filter((u) => u.pathname === "/api/autopilot/runs/run-old").length;
    await new Promise((r) => setTimeout(r, 3500));
    expect(page.urls.filter((u) => u.pathname === "/api/autopilot/runs/run-old").length).toBe(n);
  }, 8000);
});
