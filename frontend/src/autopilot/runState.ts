import type { RunState } from "../api/types";

/** Chip tone per run state, shared by the target cards, the runs list and the run page. */
export const RUN_TONE: Record<RunState, "good" | "warn" | "bad" | "accent" | "neutral"> = {
  done: "good",
  done_with_failures: "warn",
  failed: "bad",
  interrupted: "warn",
  running: "accent",
  queued: "neutral",
};
