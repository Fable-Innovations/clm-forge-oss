import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TargetDialog } from "./TargetDialog";
import type { AutopilotTarget, HeadsetPreset } from "../api/types";
import { autopilotRoute, HEADS, HEADS_BASE, PRESET, target } from "../testing/fixtures";
import { stubFetch, type Call } from "../testing/stubFetch";

// jsdom has no <dialog> modal API
HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
HTMLDialogElement.prototype.close = function () {
  this.open = false;
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

/** The fixture backend answers everything, except what a test overrides. */
function backend(override?: (c: Call) => [number, unknown] | undefined, presets?: HeadsetPreset[]): Call[] {
  const route = autopilotRoute(presets);
  return stubFetch((c) => override?.(c) ?? route(c));
}

function open(t?: AutopilotTarget) {
  const onClose = vi.fn();
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <TargetDialog open onClose={onClose} target={t} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return onClose;
}

const ready = () => waitFor(() => expect((screen.getByLabelText("Head set") as HTMLSelectElement).value).toBe(PRESET.key));

describe("TargetDialog", () => {
  it("creates a head-set target from a preset and its picks", async () => {
    const calls = backend();
    const ask = vi.fn(() => false);
    vi.stubGlobal("confirm", ask);
    open();
    await ready();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Ticket heads 0.6B" } });
    expect((screen.getByLabelText("Id") as HTMLInputElement).value).toBe("ticket-heads-0.6b");
    await waitFor(() => expect(screen.getAllByRole("option").some((o) => o.textContent?.startsWith("Qwen3-0.6B (MLX 4-bit)"))).toBe(true));
    // no app-wide default: the encoder starts on the preset's recommendation, and says why
    expect((screen.getByLabelText("Encoder") as HTMLSelectElement).value).toBe("qwen3-4b");
    expect(screen.getByText(/Recommended: Qwen3-4B \(MLX 4-bit\): the encoder this head set is built for/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Encoder"), { target: { value: "qwen3-0.6b" } });
    expect((screen.getByLabelText("Output folder") as HTMLInputElement).value).toBe(`${HEADS_BASE}-qwen3-0.6b`);
    await waitFor(() => expect(within(screen.getByRole("group", { name: "Heads" })).getAllByRole("button").length).toBe(HEADS.length));
    fireEvent.click(screen.getByRole("button", { name: "ticket-priority" }));
    // each preset source renders from the preset: its label, a select for a single setting, boxes for a multiple one
    expect((screen.getByLabelText("Event log Stage") as HTMLSelectElement).value).toBe("qa");
    expect((screen.getByLabelText("Shared export Stages prod") as HTMLInputElement).checked).toBe(true);
    fireEvent.change(screen.getByLabelText("Event log Stage"), { target: { value: "dev" } });
    fireEvent.click(screen.getByLabelText("Shared export"));
    expect(ask).not.toHaveBeenCalled(); // a new target has no stored rows to lose
    fireEvent.click(screen.getByRole("button", { name: "Create target" }));
    await waitFor(() => expect(screen.getByText(/Created\. Set its schedule below/)).toBeTruthy());
    const post = calls.find((c) => c.method === "POST" && c.url.pathname === "/api/autopilot/targets")!;
    expect(post.body).toMatchObject({
      id: "ticket-heads-0.6b",
      name: "Ticket heads 0.6B",
      kind: "headset",
      headset: {
        preset: PRESET.key,
        encoder: "qwen3-0.6b",
        output_root: `${HEADS_BASE}-qwen3-0.6b`,
        heads: ["ticket-priority"],
        sources: [
          { kind: "event_log", enabled: true, settings: { stage: "dev" } },
          { kind: "shared_export", enabled: false, settings: { stages: ["qa", "prod"] } },
        ],
      },
    });
    // created: the dialog now edits it, with its schedule
    expect(screen.getByRole("form", { name: "Schedule for ticket-heads-0.6b" })).toBeTruthy();
  });

  it("edits a target with a full replace", async () => {
    const calls = backend();
    const ask = vi.fn(() => true);
    vi.stubGlobal("confirm", ask);
    const onClose = open(target());
    await ready();
    await waitFor(() => expect((screen.getByLabelText("Encoder") as HTMLSelectElement).value).toBe("qwen3-0.6b-bf16"));
    expect(screen.queryByLabelText("Id")).toBeNull(); // the id only shows on create
    fireEvent.change(screen.getByLabelText("Encoder"), { target: { value: "qwen3-0.6b-8bit" } });
    fireEvent.change(screen.getByLabelText("Event log Stage"), { target: { value: "prod" } });
    // another value: the rows read under the old one are removed at the next run, so it asks first
    expect(ask).toHaveBeenCalledWith(expect.stringMatching(/^Turn off Event log \(qa\)\? .*removes every stored real row/));
    fireEvent.click(screen.getByLabelText("Shared export Stages qa"));
    expect(ask).toHaveBeenCalledWith(expect.stringMatching(/^Turn off Shared export \(qa\)\?/));
    fireEvent.click(screen.getByLabelText("Replace a failing head with a better failing one"));
    fireEvent.click(screen.getByRole("button", { name: "Save target" }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const put = calls.find((c) => c.method === "PUT" && c.url.pathname === "/api/autopilot/targets/support-ticket-heads")!;
    const h = (put.body as { headset: Record<string, unknown> }).headset;
    expect(h.encoder).toBe("qwen3-0.6b-8bit");
    expect(h.output_root).toBe(`${HEADS_BASE}-qwen3-0.6b-8bit`);
    expect(h.adapter).toBeUndefined(); // the command is the preset's, never sent by the page
    expect(h.sources).toEqual([
      { kind: "event_log", enabled: true, settings: { stage: "prod" } },
      { kind: "shared_export", enabled: true, settings: { stages: ["prod"] } },
    ]);
    expect((h.policy as { promote_over_failing_incumbent?: boolean }).promote_over_failing_incumbent).toBe(true);
  });

  it("asks before turning a source off and keeps it on when cancelled", async () => {
    const calls = backend();
    const ask = vi.fn(() => false);
    vi.stubGlobal("confirm", ask);
    open(target());
    await ready();
    const box = screen.getByLabelText("Event log") as HTMLInputElement;
    expect(box.checked).toBe(true);
    fireEvent.click(box);
    expect(ask).toHaveBeenCalledWith(expect.stringMatching(/the rows generated from them and the unpromoted heads trained on them/));
    expect((screen.getByLabelText("Event log") as HTMLInputElement).checked).toBe(true);
    ask.mockReturnValue(true);
    fireEvent.click(screen.getByLabelText("Event log"));
    expect((screen.getByLabelText("Event log") as HTMLInputElement).checked).toBe(false);
    expect((screen.getByLabelText("Event log Stage") as HTMLSelectElement).disabled).toBe(true);
    fireEvent.click(screen.getByLabelText("Event log")); // turning it on asks nothing
    expect(ask).toHaveBeenCalledTimes(2);
    expect(calls.some((c) => c.method !== "GET")).toBe(false);
  });

  it("shows the server's field errors next to their fields", async () => {
    backend((c) =>
      c.method === "POST" && c.url.pathname === "/api/autopilot/targets"
        ? [422, { detail: "x", errors: [
            { field: "headset.output_root", message: `Output folder is inside a protected folder (${HEADS_BASE}); the autopilot never writes there` },
            { field: "headset.policy.min_count", message: "must be at least 1" },
            { field: "headset.sources.0.settings.stage", message: "must be one of 'qa', 'prod' or 'dev'" },
            { field: "sources.1.settings.stages", message: "pick at least one" },
          ] }]
        : undefined,
    );
    open();
    await ready();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Bad" } });
    fireEvent.click(screen.getByRole("button", { name: "Create target" }));
    await waitFor(() => expect(screen.getByText(/Output folder is inside a protected folder/)).toBeTruthy());
    expect(screen.getByText("must be at least 1")).toBeTruthy();
    expect(within(screen.getByRole("listitem", { name: "Event log source" })).getByText("must be one of 'qa', 'prod' or 'dev'")).toBeTruthy();
    // an error path without the headset. prefix lands on its source too
    expect(within(screen.getByRole("listitem", { name: "Shared export source" })).getByText("pick at least one")).toBeTruthy();
    // the policy section opens to show its error
    expect((screen.getByText("Policy").closest("details") as HTMLDetailsElement).open).toBe(true);
  });

  it("says where head-set presets come from when the server has none, and project targets still work", async () => {
    const calls = backend(undefined, []);
    open();
    await waitFor(() => expect(screen.getByRole("status", { name: "No head-set presets" })).toBeTruthy());
    expect(screen.getByText("headset_presets.json")).toBeTruthy();
    expect(screen.getByText("FORGE_HEADSET_PRESETS")).toBeTruthy();
    expect(screen.queryByLabelText("Head set")).toBeNull();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Heads" } });
    expect((screen.getByRole("button", { name: "Create target" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("tab", { name: "Project" }));
    expect(screen.queryByRole("status", { name: "No head-set presets" })).toBeNull();
    expect(screen.getByLabelText("Project")).toBeTruthy();
    expect(calls.some((c) => c.method !== "GET")).toBe(false);
  });
});
