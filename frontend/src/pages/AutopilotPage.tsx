import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, Navigate, useSearchParams } from "react-router-dom";
import { Bot, Pencil, Play, Plus, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { isLive, qk, RUN_POLL_MS } from "../api/queries";
import type { AutopilotTarget } from "../api/types";
import { ActivityJournal } from "../autopilot/ActivityJournal";
import { GroupReadiness } from "../autopilot/GroupReadiness";
import { ProvenancePanel } from "../autopilot/ProvenancePanel";
import { RunProgress } from "../autopilot/RunProgress";
import { ScheduleEditor } from "../autopilot/ScheduleEditor";
import { TargetDialog } from "../autopilot/TargetDialog";
import { Button, Chip, Dialog, Empty, Panel, Spinner } from "../components/ui";
import { clock, cn } from "../lib/format";
import { RUN_TONE } from "../autopilot/runState";

const runLink = (id: string) => `/autopilot/runs/${encodeURIComponent(id)}`;

function TargetCard({ t, selected, onSelect, onEdit }: { t: AutopilotTarget; selected: boolean; onSelect: () => void; onEdit: () => void }) {
  const qc = useQueryClient();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: qk.autopilotTargets });
    qc.invalidateQueries({ queryKey: qk.autopilotRuns(t.id) });
  };
  const [confirm, setConfirm] = useState(false);
  const run = useMutation({ mutationFn: () => api.runTarget(t.id), onSettled: refresh });
  const toggle = useMutation({ mutationFn: (on: boolean) => api.setTargetEnabled(t.id, on), onSettled: refresh });
  const del = useMutation({
    mutationFn: () => api.deleteTarget(t.id),
    onSuccess: () => {
      setConfirm(false);
      qc.invalidateQueries({ queryKey: qk.encoders });
    },
    onSettled: refresh,
  });
  const last = t.last_run;
  const error = [run, toggle].find((m) => m.isError)?.error as Error | undefined;
  return (
    <li className={cn("p-3 rounded-[12px] border space-y-2", selected ? "border-accent bg-accent-soft/40" : "border-line")} aria-label={`Target ${t.name}`}>
      <div className="flex flex-wrap items-center gap-2">
        <button className="text-[13.5px] font-medium cursor-pointer hover:underline" onClick={onSelect}>
          {t.name}
        </button>
        <Chip>{t.kind === "headset" ? "head set" : "project"}</Chip>
        {!t.enabled && <Chip tone="warn">disabled</Chip>}
        {t.encoder_name && <span className="text-[12px] text-muted">{t.encoder_name}</span>}
        <span className="flex-1" />
        <label className="inline-flex items-center gap-1.5 text-[12px] text-ink-2 cursor-pointer" title={t.enabled ? "Disable: no scheduled or manual runs" : "Enable"}>
          <input type="checkbox" aria-label={`${t.name} enabled`} checked={t.enabled} disabled={toggle.isPending} onChange={(e) => toggle.mutate(e.target.checked)} />
          enabled
        </label>
        <Button size="sm" variant="ghost" icon={<Pencil size={12} />} onClick={onEdit}>
          Edit
        </Button>
        <Button size="sm" variant="ghost" icon={<Trash2 size={12} />} onClick={() => setConfirm(true)} aria-label={`Delete ${t.name}`}>
          Delete
        </Button>
        <Button size="sm" variant="primary" icon={<Play size={12} />} disabled={!t.enabled || run.isPending} onClick={() => run.mutate()}>
          Run now
        </Button>
      </div>
      {last && (
        <Link to={runLink(last.id)} className="block rounded-[9px] p-2 -mx-1 hover:bg-sunken/60" aria-label={`Last run of ${t.name}`}>
          <span className="flex flex-wrap items-center gap-1.5 text-[12px] text-muted mb-1">
            last run <Chip tone={RUN_TONE[last.state]}>{last.state.replace(/_/g, " ")}</Chip> {clock(last.ended ?? last.first_started ?? last.started ?? last.created)}
            {last.encoder && <span>on {last.encoder}</span>}
          </span>
          <RunProgress run={last} showDetail={false} />
        </Link>
      )}
      {(t.pending_actions ?? []).map((a) => (
        <p key={a.id} className="text-[12px] text-ink-2" aria-label={`Pending action ${a.action}`}>
          <Chip tone="warn">{a.action === "resplit_val" ? "redraw val" : a.action}</Chip>{" "}
          {a.action === "resplit_val" ? "Redraws every head's validation split the way its frozen test was built, in the next scheduled or manual run" : a.reason}
          <span className="text-muted"> ({a.approved})</span>
          {a.last_error && <span className="block text-bad">Last attempt failed: {a.last_error}</span>}
        </p>
      ))}
      <ScheduleEditor targetId={t.id} schedule={t.schedule} />
      {error && <p className="text-[12px] text-bad">{error.message}</p>}
      <Dialog open={confirm} onClose={() => setConfirm(false)} title={`Delete ${t.name}?`} width={420}>
        <p className="text-[13px] text-ink-2">The autopilot stops training this target. Runs and journal stay.</p>
        {del.isError && <p className="text-[12.5px] text-bad mt-2">{(del.error as Error).message}</p>}
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="ghost" onClick={() => setConfirm(false)}>
            Keep it
          </Button>
          <Button variant="danger" disabled={del.isPending} onClick={() => del.mutate()}>
            {del.isPending ? "Deleting…" : "Delete target"}
          </Button>
        </div>
      </Dialog>
    </li>
  );
}

/**
 * The autopilot: what each target needs, what it did about it, and why. Targets (head sets through their preset's
 * adapter, Forge projects through the orchestrator) run on a schedule or on demand; every decision lands in the
 * journal. Each run has its own page (/autopilot/runs/:runId).
 */
export function AutopilotPage() {
  const [params, setParams] = useSearchParams();
  const targets = useQuery({
    queryKey: qk.autopilotTargets,
    queryFn: api.autopilotTargets,
    refetchInterval: (q) => (q.state.data?.some((t) => isLive(t.last_run?.state)) ? RUN_POLL_MS : 10_000),
  });
  const sel = params.get("target") ?? targets.data?.[0]?.id ?? "";
  const selTarget = targets.data?.find((t) => t.id === sel);
  const runs = useQuery({
    queryKey: qk.autopilotRuns(sel),
    queryFn: () => api.autopilotRuns(sel),
    enabled: !!sel,
    refetchInterval: (q) => (q.state.data?.some((r) => isLive(r.state)) ? RUN_POLL_MS : 10_000),
  });
  const queue = useQuery({ queryKey: qk.modelQueue, queryFn: api.modelQueue, refetchInterval: 5_000 });
  const [dialog, setDialog] = useState<{ open: boolean; target?: AutopilotTarget }>({ open: false });
  const heads = Object.keys(runs.data?.[0]?.result.heads ?? {});
  const liveSel = isLive(selTarget?.last_run?.state);
  const legacyRun = params.get("run");
  if (legacyRun) return <Navigate to={runLink(legacyRun)} replace />;
  const select = (tid: string) => {
    const n = new URLSearchParams(params);
    n.set("target", tid);
    setParams(n, { replace: true });
  };

  return (
    <div className="p-4 max-w-[1400px] mx-auto grid gap-4 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start">
      <div className="space-y-4 min-w-0">
        <Panel
          title="Autopilot"
          sub="Detects what each target needs (labels, balance, shape, controls, staleness), acts on it, and journals why."
          collapsible={false}
          aside={
            !!targets.data?.length && (
              <Button size="sm" icon={<Plus size={12} />} onClick={() => setDialog({ open: true })}>
                New target
              </Button>
            )
          }
        >
          {targets.isLoading ? (
            <Spinner />
          ) : targets.isError ? (
            <p className="text-[12.5px] text-bad">{(targets.error as Error).message}</p>
          ) : !targets.data?.length ? (
            <Empty
              icon={<Bot size={22} />}
              title="No targets yet"
              action={
                <Button variant="primary" icon={<Plus size={13} />} onClick={() => setDialog({ open: true })}>
                  New target
                </Button>
              }
            >
              A target is a head set or a project the autopilot keeps trained: on a schedule, on demand, or when new data arrives.
            </Empty>
          ) : (
            <ul className="space-y-2">
              {targets.data.map((t) => (
                <TargetCard key={t.id} t={t} selected={t.id === sel} onSelect={() => select(t.id)} onEdit={() => setDialog({ open: true, target: t })} />
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Model-job queue" sub="One model or teacher job at a time, machine-wide; project stages and autopilot steps share it." collapsible={false}>
          {!queue.data ? (
            <Spinner />
          ) : !queue.data.active.length ? (
            <p className="text-[12.5px] text-muted">Idle.</p>
          ) : (
            <ol className="space-y-1 text-[12.5px]">
              {queue.data.active.map((j) => (
                <li key={j.id} className="flex flex-wrap items-center gap-2">
                  <Chip tone={j.state === "running" ? "accent" : j.state === "deferred" ? "warn" : "neutral"}>{j.state}</Chip>
                  <span className="font-mono">{j.kind}</span>
                  <span className="text-muted font-mono truncate">{j.owner}</span>
                  {j.error && <span className="text-warn">{j.error}</span>}
                </li>
              ))}
            </ol>
          )}
        </Panel>
        {sel && (
          <Panel title="Runs" sub={selTarget?.name} collapsible={false}>
            {!runs.data?.length ? (
              <p className="text-[12.5px] text-muted">No runs yet.</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {runs.data.slice(0, 12).map((r) => (
                  <Link
                    key={r.id}
                    to={runLink(r.id)}
                    className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-full border border-line text-[12px] hover:border-accent"
                    title={r.encoder ? `on ${r.encoder}` : undefined}
                  >
                    <Chip tone={RUN_TONE[r.state]}>{r.state.replace(/_/g, " ")}</Chip> {clock(r.first_started ?? r.created)}
                  </Link>
                ))}
              </div>
            )}
          </Panel>
        )}
      </div>
      <div className="space-y-4 min-w-0">
        <Panel title="Activity journal" sub="Every need, remedy, round and decision, with its reason and counts." collapsible={false}>
          {sel ? <ActivityJournal target={sel} refetchMs={liveSel ? RUN_POLL_MS : 10_000} /> : <p className="text-[12.5px] text-muted">Pick a target.</p>}
        </Panel>
        <Panel title="Call-group readiness" collapsible={false}>
          <GroupReadiness />
        </Panel>
        <Panel title="Provenance" sub="What a promoted head was trained on." collapsible={false}>
          <ProvenancePanel heads={heads} />
        </Panel>
      </div>
      <TargetDialog open={dialog.open} target={dialog.target} onClose={() => setDialog({ open: false })} />
    </div>
  );
}
