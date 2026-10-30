import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { api } from "../api/client";
import { isLive, qk, RUN_POLL_MS } from "../api/queries";
import type { AutopilotRun } from "../api/types";
import { ActivityJournal } from "../autopilot/ActivityJournal";
import { HeadResultTable } from "../autopilot/HeadResultTable";
import { RunDetail } from "../autopilot/RunDetail";
import { RunProgress } from "../autopilot/RunProgress";
import { RUN_TONE } from "../autopilot/runState";
import { StepTimeline } from "../autopilot/StepTimeline";
import { Chip, Panel, Spinner } from "../components/ui";
import { clock, int } from "../lib/format";

const TRIGGER: Record<AutopilotRun["trigger"], string> = { schedule: "Scheduled run", manual: "Manual run", data: "New-data run" };

/** One autopilot run: where it is (progress, step timeline), what it produced (per-head results, row counts) and
 * why (its journal). Polls every 3 s while the run is queued or running. */
export function AutopilotRunPage() {
  const { runId = "" } = useParams();
  const run = useQuery({
    queryKey: qk.autopilotRun(runId),
    queryFn: () => api.autopilotRun(runId),
    refetchInterval: (q) => (isLive(q.state.data?.state) ? RUN_POLL_MS : false),
  });
  if (run.isLoading)
    return (
      <div className="p-6 flex items-center gap-2 text-ink-2">
        <Spinner /> Loading the run…
      </div>
    );
  if (run.isError)
    return (
      <div className="p-6 text-[13px]">
        <p className="text-bad">Couldn't load this run: {(run.error as Error).message}</p>
        <Link to="/autopilot" className="text-accent underline">
          Back to the autopilot
        </Link>
      </div>
    );
  const r = run.data!;
  const live = isLive(r.state);
  const total = r.counts?.total;
  return (
    <div className="p-4 max-w-[1400px] mx-auto space-y-4">
      <Panel collapsible={false} title={undefined}>
        <header className="space-y-2" aria-label="Run header">
          <div className="flex flex-wrap items-center gap-2">
            <Link to={`/autopilot?target=${encodeURIComponent(r.target_id)}`} className="p-1 -ml-1 rounded text-muted hover:text-ink hover:bg-sunken" aria-label="Back to the autopilot">
              <ArrowLeft size={16} />
            </Link>
            <h1 className="font-display text-[18px] font-medium tracking-[-0.02em]">{r.target_removed ? "Target removed" : (r.target_name ?? r.target_id)}</h1>
            <Chip tone={RUN_TONE[r.state]}>{r.state.replace(/_/g, " ")}</Chip>
            <span className="text-[12.5px] text-ink-2">{TRIGGER[r.trigger]}</span>
            {r.encoder && <span className="text-[12.5px] text-muted">on {r.encoder}</span>}
            <span className="text-[12.5px] text-muted">
              started {clock(r.first_started ?? r.started)}
              {r.ended && <> · ended {clock(r.ended)}</>}
            </span>
          </div>
          <RunProgress run={r} />
          {r.result.error && <p className="text-[12.5px] text-bad font-mono whitespace-pre-wrap">{r.result.error}</p>}
        </header>
      </Panel>
      <div className="grid gap-4 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start">
        <div className="space-y-4 min-w-0">
          <Panel title="Steps" sub="Grouped by head; pending steps are the plan for the heads still to come." collapsible={false}>
            <StepTimeline entries={r.timeline ?? []} />
          </Panel>
          <Panel title="Results" collapsible={false}>
            <div className="space-y-3">
              <HeadResultTable rows={r.results ?? []} />
              {total && (
                <p className="text-[12.5px] text-ink-2" aria-label="Row counts">
                  Generated {int(total.generated)} rows · labelled {int(total.labelled)} real rows · excluded {int(total.excluded)}
                </p>
              )}
            </div>
          </Panel>
        </div>
        <div className="space-y-4 min-w-0">
          <Panel title="Journal" sub="What this run did and why." collapsible={false}>
            <ActivityJournal run={r.id} refetchMs={live ? RUN_POLL_MS : false} />
          </Panel>
        </div>
      </div>
      <Panel title="Per head" sub="Needs, remedies, rounds and the promotion decision." collapsible={false}>
        <RunDetail runId={r.id} />
      </Panel>
    </div>
  );
}
