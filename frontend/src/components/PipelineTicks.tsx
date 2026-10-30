import type { ProjectState } from "../api/types";
import { MAIN_STATES } from "../api/types";
import { STATE_LABEL } from "./StateBadge";

/** Compact progress through the main state chain. Off-chain states (no_fit, cancelled, errored) mark where they left. */
export function PipelineTicks({ state, lastMain }: { state: ProjectState; lastMain?: ProjectState }) {
  const idx = MAIN_STATES.indexOf(state);
  const off = idx < 0;
  const reached = off ? MAIN_STATES.indexOf(lastMain ?? "analyzing") : idx;
  const tone = state === "errored" ? "var(--bad)" : off ? "var(--muted)" : state === "delivered" ? "var(--good)" : state === "proposed" ? "var(--option)" : "var(--accent)";
  return (
    <div className="flex items-center gap-[3px]" aria-label={`Pipeline: ${STATE_LABEL[state]}`}>
      {MAIN_STATES.slice(1).map((s, i) => {
        const k = i + 1;
        const done = k < reached || (k === reached && state === "delivered");
        const cur = k === reached && state !== "delivered";
        return (
          <span
            key={s}
            title={STATE_LABEL[s]}
            className="h-[6px] rounded-[2px]"
            style={{
              width: cur ? 14 : 8,
              background: done ? tone : cur ? tone : "var(--line)",
              opacity: done && !off ? 0.55 : 1,
              boxShadow: cur && !off ? `0 0 8px ${tone}` : undefined,
            }}
          />
        );
      })}
    </div>
  );
}
