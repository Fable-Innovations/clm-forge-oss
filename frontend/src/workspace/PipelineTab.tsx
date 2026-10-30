import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowRightLeft, Boxes, ClipboardCheck, FileText, Info, RotateCcw, ShieldAlert, Square, ThumbsUp, PencilLine } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { ForgeEvent } from "../api/types";
import { Button, Chip, Dialog, Panel, Segmented, Textarea } from "../components/ui";
import { ActorIcon, STATE_LABEL } from "../components/StateBadge";
import { StateDiagram } from "../viz/StateDiagram";
import { clock, cn } from "../lib/format";
import { useWs } from "./context";
import { LabelGateActions } from "./LabelTab";

type Filter = "all" | "transitions" | "rejected" | "logs";

function Row({ e }: { e: ForgeEvent }) {
  const [open, setOpen] = useState(false);
  let icon = <Info size={13} />;
  let tone = "";
  let body: React.ReactNode = null;
  let title: React.ReactNode = null;
  switch (e.type) {
    case "state":
      icon = <ArrowRightLeft size={13} />;
      title = (
        <span>
          <span className="text-muted">{STATE_LABEL[e.from as keyof typeof STATE_LABEL] ?? e.from}</span> →{" "}
          <span className="font-semibold">{STATE_LABEL[e.to] ?? e.to}</span>
          <span className="text-muted font-normal"> via {e.event}</span>
        </span>
      );
      body = (
        <span className="inline-flex items-start gap-1.5">
          <ActorIcon actor={e.actor} size={12} className="mt-[3px] shrink-0 text-muted" />
          <span>
            <span className="text-muted">{e.actor}: </span>
            {e.reason}
          </span>
        </span>
      );
      break;
    case "rejected":
      icon = <ShieldAlert size={13} />;
      tone = "bad";
      title = (
        <span className="text-bad font-semibold">
          Rejected: {e.event} <span className="font-normal">by {e.actor}</span>
        </span>
      );
      body = (
        <>
          <div className="text-ink">{e.error}</div>
          <div className="text-muted mt-0.5">Proposed because: {e.reason}</div>
        </>
      );
      break;
    case "error":
      icon = <AlertTriangle size={13} />;
      tone = "bad";
      title = <span className="text-bad font-semibold">{e.stage} error</span>;
      body = (
        <>
          <div>{e.message}</div>
          {e.traceback && (
            <button onClick={() => setOpen((o) => !o)} className="text-accent text-[11.5px] mt-1 cursor-pointer">
              {open ? "Hide traceback" : "Show traceback"}
            </button>
          )}
          {open && <pre className="mt-1.5 text-[11px] font-mono bg-code border border-line rounded-[6px] p-2 overflow-x-auto whitespace-pre">{e.traceback}</pre>}
        </>
      );
      break;
    case "log":
      icon = e.level === "warn" ? <AlertTriangle size={13} /> : <FileText size={13} />;
      tone = e.level === "warn" ? "warn" : e.level === "error" ? "bad" : "";
      title = (
        <span className="font-normal">
          <span className="text-muted">{e.stage}</span> {e.message}
        </span>
      );
      break;
    case "proposal":
      icon = <ClipboardCheck size={13} />;
      title = (
        <span>
          Proposal{e.proposal.revision ? ` revision ${e.proposal.revision}` : ""}: <span className="font-normal">{e.proposal.verdict === "fit" ? `${e.proposal.opportunities.length} opportunities` : "no fit"}</span>
        </span>
      );
      break;
    case "eval":
      icon = <ClipboardCheck size={13} />;
      title = (
        <span>
          Evaluation report: <span className={e.report.gate_passed ? "text-good" : "text-bad"}>gate {e.report.gate_passed ? "passed" : "failed"}</span>
        </span>
      );
      break;
    case "package":
      icon = <Boxes size={13} />;
      title = (
        <span>
          Package {e.manifest.name} {e.manifest.version} <span className="font-normal text-muted">({e.manifest.files.length} files)</span>
        </span>
      );
      break;
    default:
      return null;
  }
  return (
    <li
      className={cn(
        "relative pl-7 pr-3 py-2 text-[12.5px] leading-snug border-b border-line-soft",
        tone === "bad" && "bg-bad-soft/70 border-l-2 border-l-bad",
        tone === "warn" && "bg-warn-soft/50",
      )}
    >
      <span className={cn("absolute left-2 top-[9px]", tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : "text-muted")}>{icon}</span>
      <div className="flex items-baseline justify-between gap-3">
        <div className={cn("min-w-0", e.type === "log" ? "text-ink-2" : "font-medium")}>{title}</div>
        <span className="text-[11px] text-muted tnum shrink-0">{clock(e.ts)}</span>
      </div>
      {body && <div className="text-ink-2 mt-0.5">{body}</div>}
    </li>
  );
}

export function PipelineTab() {
  const { pid, store, detail, goTab } = useWs();
  const [filter, setFilter] = useState<Filter>("all");
  const [revOpen, setRevOpen] = useState(false);
  const [feedback, setFeedback] = useState("");
  const qc = useQueryClient();
  // label_override is always on offer (recorded from the Labels tab); it is not a "your move" action
  const allowed = (detail?.allowed_human_events ?? []).filter((e) => e !== "label_override");
  const act = useMutation({
    mutationFn: async (ev: string) => {
      if (ev === "cancel") return api.cancel(pid);
      if (ev === "retry") return api.retry(pid);
      if (ev === "revise") return api.revise(pid, feedback.trim());
    },
    onSuccess: (_d, ev) => {
      if (ev === "revise") {
        setRevOpen(false);
        setFeedback("");
      }
    },
    onSettled: () => qc.invalidateQueries({ queryKey: qk.project(pid) }),
  });

  const items = useMemo(() => {
    const xs = store.timeline.filter((e) =>
      filter === "all" ? true : filter === "transitions" ? e.type === "state" : filter === "rejected" ? e.type === "rejected" || e.type === "error" : e.type === "log",
    );
    return xs.slice().reverse();
  }, [store.timeline, filter]);

  const cur = store.state;

  return (
    <div className="p-4 grid gap-4 grid-cols-1 lg:grid-cols-[minmax(360px,440px)_minmax(0,1fr)] items-start">
      <div className="space-y-4 min-w-0">
        <Panel
          title="Your move"
          sub={
            allowed.length === 0
              ? cur && ["delivered", "no_fit", "cancelled"].includes(cur)
                ? "This experiment is final."
                : "Nothing needs you right now. The agent and system jobs are driving."
              : allowed.includes("accept")
                ? "The agent is waiting for you to accept the proposal or request changes."
                : allowed.includes("review_done")
                  ? `${detail?.labels?.needs_label ? `${detail.labels.needs_label} real rows wait for your review` : "Every real row is labelled"} before training can start.`
                : allowed.includes("retry")
                  ? "A stage failed. Fix the cause, then retry from where it stopped."
                  : "You can stop the run at any time."
          }
        >
          <div className="flex flex-wrap gap-2">
            {allowed.includes("accept") && (
              <Button variant="option" icon={<ThumbsUp size={14} />} onClick={() => goTab("proposal")}>
                Review and accept
              </Button>
            )}
            {allowed.includes("revise") && (
              <Button icon={<PencilLine size={14} />} onClick={() => setRevOpen(true)}>
                Request changes
              </Button>
            )}
            {allowed.includes("review_done") && <LabelGateActions pid={pid} detail={detail} onLabel={() => goTab("label")} />}
            {allowed.includes("retry") && (
              <Button variant="primary" icon={<RotateCcw size={14} />} onClick={() => act.mutate("retry")} disabled={act.isPending}>
                Retry {store.errors.at(-1)?.stage ?? ""}
              </Button>
            )}
            {allowed.includes("cancel") && (
              <Button
                variant="danger"
                icon={<Square size={12} />}
                onClick={() => confirm("Cancel this experiment?") && act.mutate("cancel")}
                disabled={act.isPending}
              >
                Cancel run
              </Button>
            )}
            {detail && detail.attempts > 0 && (
              <Chip className="self-center" title="curate → train → evaluate cycles used">
                attempt {detail.attempts} of 3
              </Chip>
            )}
          </div>
          {act.isError && <p className="text-[12px] text-bad mt-2">{(act.error as Error).message}</p>}
        </Panel>
        <Panel
          expandable title="State machine"
          sub="The rules layer owns every transition; the agent can only propose its own events. Hover an edge for who fired it and why."
          bodyClass="px-2 pb-3 pt-1"
        >
          <StateDiagram transitions={store.transitions} current={cur} />
          <div className="flex flex-wrap gap-x-4 gap-y-1 px-2 text-[11.5px] text-muted">
            <span className="inline-flex items-center gap-1">
              <ActorIcon actor="agent" /> agent
            </span>
            <span className="inline-flex items-center gap-1">
              <ActorIcon actor="human" /> human only
            </span>
            <span className="inline-flex items-center gap-1">
              <ActorIcon actor="system" /> system job
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-4 h-[2px] bg-accent inline-block" /> path taken
            </span>
          </div>
        </Panel>
      </div>
      <Panel
        expandable title="Event timeline"
        sub={`${store.transitions.length} transitions${store.rejected.length ? `, ${store.rejected.length} rejected by the guard` : ""}${store.errors.length ? `, ${store.errors.length} errors` : ""}`}
        aside={
          <Segmented
            value={filter}
            onChange={setFilter}
            options={[
              { value: "all", label: "All" },
              { value: "transitions", label: "Transitions" },
              { value: "rejected", label: `Rejected · errors${store.rejected.length + store.errors.length ? ` (${store.rejected.length + store.errors.length})` : ""}` },
              { value: "logs", label: "Logs" },
            ]}
          />
        }
        bodyClass="p-0"
      >
        <ol className="max-h-[calc(100vh-190px)] overflow-y-auto border-t border-line-soft">
          {items.length === 0 && <li className="p-6 text-center text-[13px] text-muted">No events of this kind yet.</li>}
          {items.map((e) => (
            <Row key={e.seq} e={e} />
          ))}
        </ol>
      </Panel>

      <Dialog open={revOpen} onClose={() => setRevOpen(false)} title="Request changes to the proposal">
        <p className="text-[13px] text-ink-2 mb-3">The agent goes back to analysis with your feedback and returns a new revision.</p>
        <Textarea rows={4} autoFocus value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="e.g. Focus on the urgency scoring instead; the router is being rewritten next quarter." />
        <div className="flex justify-end gap-2 mt-3">
          <Button variant="ghost" onClick={() => setRevOpen(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!feedback.trim() || act.isPending} onClick={() => act.mutate("revise")}>
            Send feedback
          </Button>
        </div>
      </Dialog>
    </div>
  );
}
