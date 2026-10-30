import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { RunProgress, progressText } from "./RunProgress";
import type { AutopilotRun } from "../api/types";

afterEach(cleanup);

const RUN: AutopilotRun = {
  id: "r1",
  target_id: "t",
  trigger: "manual",
  state: "running",
  step_cursor: null,
  round: 1,
  result: {},
  started: null,
  ended: null,
  created: "2026-10-14T09:00:00+00:00",
  progress: {
    fraction: 0.4,
    heads_done: 3,
    heads_total: 9,
    current: { label: "Training ticket-escalation (round 1)", head: "ticket-escalation" },
    detail: "training seed 2 of 3",
    elapsed_s: 42 * 60,
    eta_s: 70 * 60,
    eta_note: null,
  },
};

describe("RunProgress", () => {
  it("says where a live run is, how long it took and how long is left", () => {
    render(<RunProgress run={RUN} />);
    expect(screen.getByText("3 of 9 heads · Training ticket-escalation (round 1) · 42 min elapsed · ~1 h 10 min left")).toBeTruthy();
    expect(screen.getByText("training seed 2 of 3")).toBeTruthy();
  });

  it("waits for the first head before an ETA, and a finished run says how long it took", () => {
    expect(progressText({ ...RUN.progress!, eta_s: null, eta_note: "ETA after the first head" }, true)).toBe(
      "3 of 9 heads · Training ticket-escalation (round 1) · 42 min elapsed · ETA after the first head",
    );
    expect(progressText({ ...RUN.progress!, heads_done: 9, current: null, elapsed_s: 3 * 3600 + 5 * 60 }, false)).toBe("9 of 9 heads · took 3 h 5 min");
  });

  it("a run without a recorded plan says so, with no bar", () => {
    render(<RunProgress run={{ ...RUN, progress: null }} />);
    expect(screen.getByText("Progress not recorded for this run.")).toBeTruthy();
    expect(screen.queryByLabelText("Run progress")).toBeNull();
  });
});
