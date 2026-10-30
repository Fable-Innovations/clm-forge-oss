import type { AutopilotRun, RunProgress as Progress } from "../api/types";
import { Meter } from "../components/ui";
import { cn, duration } from "../lib/format";

/** "3 of 9 heads · Training ticket-escalation (round 1) · 42 min elapsed · ~1 h 10 min left" (backend numbers only). */
export function progressText(p: Progress, live: boolean): string {
  const parts = [`${p.heads_done} of ${p.heads_total} heads`];
  if (live && p.current) parts.push(p.current.label);
  if (p.elapsed_s != null) parts.push(live ? `${duration(p.elapsed_s)} elapsed` : `took ${duration(p.elapsed_s)}`);
  if (live) parts.push(p.eta_s != null ? `~${duration(p.eta_s)} left` : (p.eta_note ?? ""));
  return parts.filter(Boolean).join(" · ");
}

/** A run's progress bar and its line of words; a run from before progress was recorded says so instead. */
export function RunProgress({ run, className, showDetail = true }: { run: AutopilotRun; className?: string; showDetail?: boolean }) {
  const p = run.progress;
  if (!p) return <p className={cn("text-[12px] text-muted", className)}>Progress not recorded for this run.</p>;
  const live = run.state === "running" || run.state === "queued";
  const color = run.state === "failed" ? "var(--bad)" : run.state === "done_with_failures" || run.state === "interrupted" ? "var(--warn)" : "var(--accent)";
  return (
    <div className={cn("space-y-1 min-w-0", className)} aria-label="Run progress">
      <Meter value={p.fraction} color={color} />
      <p className="text-[12px] text-ink-2 truncate" title={progressText(p, live)}>
        {progressText(p, live)}
      </p>
      {showDetail && live && p.detail && <p className="text-[11.5px] text-muted font-mono truncate" title={p.detail}>{p.detail}</p>}
    </div>
  );
}
