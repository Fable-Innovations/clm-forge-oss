import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ScheduleEditor } from "./ScheduleEditor";
import type { AutopilotSchedule } from "../api/types";

let puts: { path: string; body: Record<string, unknown> }[];
beforeEach(() => {
  puts = [];
  vi.stubGlobal("fetch", async (input: string, init?: RequestInit) => {
    if (String(input).endsWith("/api/session")) return new Response(JSON.stringify({ token: "test-session" }), { status: 200, headers: { "Content-Type": "application/json" } });
    const url = new URL(input, "http://localhost");
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    if ((init?.method ?? "GET") === "PUT") {
      puts.push({ path: url.pathname, body });
      if (!/^\d\d:\d\d$/.test(body.time_local)) return new Response(JSON.stringify({ detail: "time_local must be HH:MM (24 h)" }), { status: 422, headers: { "Content-Type": "application/json" } });
      return new Response(JSON.stringify({ target_id: "example", ...body, next_run: "2026-10-18T03:00:00+00:00", last_run: null }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } });
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderEditor(schedule: AutopilotSchedule | null) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ScheduleEditor targetId="example" schedule={schedule} />
    </QueryClientProvider>,
  );
}

describe("ScheduleEditor", () => {
  it("sets a weekly schedule with weekday and time", async () => {
    renderEditor(null);
    fireEvent.change(screen.getByLabelText("Weekday"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Time"), { target: { value: "04:30" } });
    fireEvent.click(screen.getByRole("button", { name: "Set schedule" }));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ path: "/api/autopilot/targets/example/schedule", body: { cadence: "weekly", weekday: 0, time_local: "04:30", enabled: true } });
  });

  it("daily drops the weekday, and saving is only offered when something changed", async () => {
    renderEditor({ target_id: "example", cadence: "weekly", weekday: 6, time_local: "03:00", next_run: "2026-10-18T03:00:00+00:00", last_run: null, enabled: true });
    const save = screen.getByRole("button", { name: "Save schedule" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    expect(screen.getByText(/next run/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Cadence"), { target: { value: "daily" } });
    expect(screen.queryByLabelText("Weekday")).toBeNull();
    expect(save.disabled).toBe(false);
    fireEvent.click(save);
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0].body).toEqual({ cadence: "daily", weekday: null, time_local: "03:00", enabled: true });
  });

  it("shows the server's refusal", async () => {
    renderEditor(null);
    fireEvent.change(screen.getByLabelText("Time"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Set schedule" }));
    expect(await screen.findByText("time_local must be HH:MM (24 h)")).toBeTruthy();
  });
});
