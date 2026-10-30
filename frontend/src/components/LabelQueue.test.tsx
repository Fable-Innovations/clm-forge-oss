import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LabelQueue } from "./LabelQueue";
import { LabelSetStub, realRow } from "../testing/labelSet";
import type { QueueFilter, TaskSchema } from "../api/types";

const TASK: TaskSchema = {
  name: "router",
  question_id: "queue",
  type: "choice",
  instructions: "Which queue handles this ticket?",
  criteria: { billing: "invoices and refunds", technical: "bugs and errors", account: "login and profile" },
  state_description: "ticket",
  state_example: "Subject: hi",
};

let set: LabelSetStub;
let calls: { method: string; path: string; body?: unknown }[];
let editable = true;

/** Serves the label-set routes the component calls from an in-memory queue with the backend's rules. */
function stubFetch() {
  const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  vi.stubGlobal("fetch", async (input: string, init?: RequestInit) => {
    if (String(input).endsWith("/api/session")) return new Response(JSON.stringify({ token: "test-session" }), { status: 200, headers: { "Content-Type": "application/json" } });
    const url = new URL(input, "http://localhost");
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path: url.pathname, body });
    const sub = url.pathname.replace(/^\/api\/labelsets\/[^/]+/, "");
    if (sub === "/queue") {
      const pg = set.page((url.searchParams.get("state") ?? "needs_label") as QueueFilter, 0, 100);
      return json({ id: "p:x", name: "x", kind: "project", ...pg, policy: "user", state: editable ? "label_review" : "training", editable });
    }
    const m = sub.match(/^\/rows\/(.+)$/);
    if (m) {
      const row = body.skip ? set.skip(decodeURIComponent(m[1])) : set.label(decodeURIComponent(m[1]), body.label);
      return json({ row: set.out(row), counts: set.counts() });
    }
    if (sub === "/undo") {
      const row = set.undo();
      return json({ row: row ? set.out(row) : null, counts: set.counts() });
    }
    return json({ detail: "not found" }, 404);
  });
}

function renderQueue(onOverride?: (rid: string, label: string) => Promise<unknown>) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <LabelQueue lsid="p:x" onOverride={onOverride} />
    </QueryClientProvider>,
  );
}

const shownState = () => screen.getByTestId("label-state").textContent;
const key = (k: string, extra: Partial<KeyboardEventInit> = {}) => fireEvent.keyDown(window, { key: k, ...extra });

beforeEach(() => {
  set = new LabelSetStub(TASK);
  set.add([realRow("a", "I was charged twice", "router"), realRow("b", "The app crashes on start", "router"), realRow("c", "Cannot reset my password", "router")]);
  calls = [];
  editable = true;
  stubFetch();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("LabelQueue", () => {
  it("shows the state, the question and numbered options with their criteria", async () => {
    renderQueue();
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    expect(screen.getByText("Which queue handles this ticket?")).toBeTruthy();
    expect(screen.getByText("invoices and refunds")).toBeTruthy();
    expect(screen.getByText("0 of 3 labelled")).toBeTruthy();
  });

  it("labels with number keys and moves on to the next row", async () => {
    renderQueue();
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    key("1");
    await waitFor(() => expect(shownState()).toBe("The app crashes on start"));
    expect(calls.find((c) => c.method === "POST")).toEqual({ method: "POST", path: "/api/labelsets/p%3Ax/rows/a", body: { label: "billing" } });
    expect(set.rows[0]).toMatchObject({ label: "billing", label_source: "user", label_state: "labelled" });
    await waitFor(() => expect(screen.getByText("1 of 3 labelled")).toBeTruthy());
    key("9"); // no ninth option: nothing happens
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("skips with S or the right arrow; skipped rows wait at the end", async () => {
    renderQueue();
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    key("s");
    await waitFor(() => expect(set.rows[0].skipped).toBe(true));
    await waitFor(() => expect(shownState()).toBe("The app crashes on start"));
    key("ArrowRight");
    await waitFor(() => expect(set.rows[1].skipped).toBe(true));
    await waitFor(() => expect(shownState()).toBe("Cannot reset my password"));
    expect(set.rows.every((r) => r.label_state === "needs_label")).toBe(true);
  });

  it("outside the needs-label view the arrows only move between rows", async () => {
    renderQueue();
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    fireEvent.click(screen.getByRole("tab", { name: "All" }));
    key("ArrowRight");
    await waitFor(() => expect(shownState()).toBe("The app crashes on start"));
    key("ArrowLeft");
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(0);
  });

  it("undoes the last action with ⌘Z or U and returns to that row", async () => {
    renderQueue();
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    key("2");
    await waitFor(() => expect(shownState()).toBe("The app crashes on start"));
    key("z", { metaKey: true });
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    expect(set.rows[0]).toMatchObject({ label: null, label_state: "needs_label", label_source: null });
    key("3");
    await waitFor(() => expect(set.rows[0].label).toBe("account"));
    key("u");
    await waitFor(() => expect(set.rows[0].label).toBe(null));
    expect(calls.filter((c) => c.path.endsWith("/undo"))).toHaveLength(2);
  });

  it("ignores keys typed into a form field", async () => {
    renderQueue();
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    const input = document.createElement("input");
    document.body.appendChild(input);
    fireEvent.keyDown(input, { key: "1" });
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(0);
    input.remove();
  });

  it("records an override while a stage locks the queue, and never writes the row", async () => {
    editable = false;
    const overrides: [string, string][] = [];
    renderQueue(async (rid, label) => overrides.push([rid, label]));
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    expect(screen.getByText(/recorded as an override and applied at the next labelling stage/)).toBeTruthy();
    key("2");
    await waitFor(() => expect(overrides).toEqual([["a", "technical"]]));
    expect(await screen.findByText(/Override technical recorded/)).toBeTruthy();
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(0);
    expect(set.rows[0].label).toBe(null);
  });

  it("shows the agent's labels and its exclusions with their reason", async () => {
    set.add([realRow("d", "Mixed billing and login issue", "router")]);
    set.agentLabel((r) => ({ label: r.rid === "a" ? "billing" : "technical", conf: 0.9 }));
    renderQueue();
    await waitFor(() => expect(screen.getByText(/Agent · 3/)).toBeTruthy());
    expect(screen.getByText(/3 of 4 labelled/)).toBeTruthy();
    expect(screen.getByText(/1 excluded by the agent/)).toBeTruthy();
    fireEvent.click(screen.getByText(/Excluded · 1/));
    await waitFor(() => expect(shownState()).toBe("Mixed billing and login issue"));
    expect(screen.getByText("excluded by the agent")).toBeTruthy();
    expect(screen.getByText(/adjudicator abstained/)).toBeTruthy();
    fireEvent.click(screen.getByText(/Agent · 3/));
    await waitFor(() => expect(shownState()).toBe("I was charged twice"));
    expect(screen.getByTitle(/Labelled by the labelling agent/)).toBeTruthy();
  });
});
