import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ActivityJournal, countsText } from "./ActivityJournal";
import type { JournalEntry } from "../api/types";

const E = (id: number, action: string, head: string | null, reason: string, counts: Record<string, unknown> = {}): JournalEntry => ({
  id,
  run_id: "r1",
  target_id: "example",
  head,
  round: action === "round" ? 2 : null,
  action,
  reason,
  counts,
  ts: "2026-10-14T09:00:00+00:00",
});
const ENTRIES = [
  E(4, "promote", "ticket-spam", "promoted to /out/ticket-spam"),
  E(3, "round", "ticket-spam", "round 2: val macro-F1 0.900, val gate passed", { val_macro_f1: 0.9 }),
  E(2, "need", "ticket-priority", "urgent holds 4.0% of train, target 22.0%", { kind: "imbalance", deficit: 120 }),
  E(1, "run_started", null, "schedule run"),
];

let urls: URL[];
beforeEach(() => {
  urls = [];
  vi.stubGlobal("fetch", async (input: string) => {
    const url = new URL(input, "http://localhost");
    urls.push(url);
    const head = url.searchParams.get("head");
    const action = url.searchParams.get("action");
    const xs = ENTRIES.filter((e) => (!head || e.head === head) && (!action || e.action === action));
    return new Response(JSON.stringify(xs), { status: 200, headers: { "Content-Type": "application/json" } });
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderJournal() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ActivityJournal target="example" />
    </QueryClientProvider>,
  );
}

describe("ActivityJournal", () => {
  it("lists every entry with its reason, head, round and counts", async () => {
    renderJournal();
    await waitFor(() => expect(screen.getAllByTestId("journal-entry")).toHaveLength(4));
    expect(screen.getByText("round 2: val macro-F1 0.900, val gate passed")).toBeTruthy();
    expect(screen.getByText("round 2")).toBeTruthy();
    expect(screen.getByText("kind imbalance · deficit 120")).toBeTruthy();
    expect(urls[0].pathname).toBe("/api/autopilot/journal");
    expect(urls[0].searchParams.get("target")).toBe("example");
  });

  it("filters by head and by action through the API", async () => {
    renderJournal();
    await waitFor(() => expect(screen.getAllByTestId("journal-entry")).toHaveLength(4));
    fireEvent.change(screen.getByLabelText("Filter by head"), { target: { value: "ticket-spam" } });
    await waitFor(() => expect(screen.getAllByTestId("journal-entry")).toHaveLength(2));
    expect(urls.some((u) => u.searchParams.get("head") === "ticket-spam")).toBe(true);
    fireEvent.change(screen.getByLabelText("Filter by action"), { target: { value: "promote" } });
    await waitFor(() => expect(screen.getAllByTestId("journal-entry")).toHaveLength(1));
    expect(screen.getByText("promoted to /out/ticket-spam")).toBeTruthy();
    // the menus still offer every head and action
    const options = [...(screen.getByLabelText("Filter by action") as HTMLSelectElement).options].map((o) => o.value);
    expect(options).toEqual(["", "need", "promote", "round", "run_started"]);
  });

  it("formats counts and skips empty values", () => {
    expect(countsText({ a: 1, b: null, c: { x: 2 }, d: "" })).toBe('a 1 · c {"x":2}');
  });
});
