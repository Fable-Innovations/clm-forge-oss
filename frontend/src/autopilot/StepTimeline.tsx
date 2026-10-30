import { CircleCheck, Circle, LoaderCircle } from "lucide-react";
import type { TimelineEntry } from "../api/types";
import { clock, cn, duration } from "../lib/format";

const secs = (a: string | null, b: string | null) => (a && b ? (new Date(b).getTime() - new Date(a).getTime()) / 1000 : null);

function Row({ e }: { e: TimelineEntry }) {
  const Icon = e.state === "done" ? CircleCheck : e.state === "running" ? LoaderCircle : Circle;
  return (
    <li className={cn("grid grid-cols-[16px_minmax(0,1fr)_auto] gap-2 items-center py-0.5", e.state === "pending" && "text-muted")} data-state={e.state}>
      <Icon size={13} className={cn(e.state === "done" && "text-good", e.state === "running" && "text-accent animate-spin")} aria-hidden />
      <span className="truncate">{e.label}</span>
      <span className="text-[11.5px] text-muted tnum">
        {e.state === "done" && e.started ? duration(secs(e.started, e.ended)) : e.state === "running" && e.started ? `since ${clock(e.started)}` : ""}
      </span>
    </li>
  );
}

/** A run's steps, grouped by head (run-level steps first), each group collapsible; pending steps come from the plan. */
export function StepTimeline({ entries }: { entries: TimelineEntry[] }) {
  const groups: { head: string | null; items: TimelineEntry[] }[] = [];
  for (const e of entries) {
    const g = groups.find((x) => x.head === e.head);
    if (g) g.items.push(e);
    else groups.push({ head: e.head, items: [e] });
  }
  if (!entries.length) return <p className="text-[12.5px] text-muted">No steps recorded yet.</p>;
  return (
    <div className="space-y-1.5 text-[12.5px]" aria-label="Step timeline">
      {groups.map((g) => {
        const done = g.items.filter((x) => x.state === "done").length;
        const running = g.items.some((x) => x.state === "running");
        const allPending = done === 0 && !running;
        return (
          <details key={g.head ?? "(run)"} open={running} className="border border-line-soft rounded-[10px] px-3 py-1.5" aria-label={g.head ?? "Run steps"}>
            <summary className="cursor-pointer select-none flex items-center gap-2">
              <span className={cn("font-medium", g.head && "font-mono text-[12px]")}>{g.head ?? "Run"}</span>
              <span className="text-muted text-[11.5px]">
                {running ? "in progress · " : allPending ? "not started · " : ""}
                {done} of {g.items.length} steps
              </span>
            </summary>
            <ol className="mt-1">
              {g.items.map((e) => (
                <Row key={e.key} e={e} />
              ))}
            </ol>
          </details>
        );
      })}
    </div>
  );
}
