import { forwardRef, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowRight, Check, CircleSlash, Loader2, LayoutDashboard, FlaskConical, PartyPopper, RotateCcw, ThumbsUp } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { ProjectDetail, ProjectState } from "../api/types";
import { MAIN_STATES } from "../api/types";
import type { ProjectStore } from "../store/projectStore";
import { Button } from "../components/ui";
import { LabelGateActions } from "./LabelTab";
import { cn } from "../lib/format";

/** The workspace tabs, in the order an experiment moves through them. Stages own a slice of the state machine;
 * Overview and Playground are tools that are always available. */
export const TABS = [
  { id: "pipeline", label: "Overview", key: "1", kind: "tool" as const, states: [] as ProjectState[] },
  { id: "proposal", label: "Proposal", key: "2", kind: "stage" as const, states: ["created", "analyzing", "proposed"] as ProjectState[] },
  { id: "data", label: "Data", key: "3", kind: "stage" as const, states: ["accepted", "curating"] as ProjectState[] },
  { id: "label", label: "Labels", key: "4", kind: "stage" as const, states: ["labelling"] as ProjectState[] },
  { id: "training", label: "Training", key: "5", kind: "stage" as const, states: ["selecting", "training"] as ProjectState[] },
  { id: "evaluation", label: "Evaluation", key: "6", kind: "stage" as const, states: ["evaluating", "reviewing"] as ProjectState[] },
  { id: "package", label: "Package", key: "7", kind: "stage" as const, states: ["packaging", "delivered"] as ProjectState[] },
  { id: "playground", label: "Playground", key: "8", kind: "tool" as const, states: [] as ProjectState[] },
];

const PHASE_TEXT: Record<string, string> = {
  importing: "importing your real rows",
  mining: "mining examples from the code",
  generating: "generating examples for each label",
  reviewing: "quality-checking generated examples",
  labelling: "relabelling with an independent teacher",
  dedup: "removing duplicates",
  splitting: "splitting into train, validation and test",
  done: "done",
};

export type StageStatus = "done" | "live" | "attention" | "upcoming" | "failed" | "tool";

/** Where the project has got to on the main chain. Off-chain states resolve to the last main state they left from. */
function reachedIndex(state: ProjectState | null, transitions: ProjectStore["transitions"]): number {
  if (!state) return -1;
  const i = MAIN_STATES.indexOf(state);
  if (i >= 0) return i;
  for (let k = transitions.length - 1; k >= 0; k--) {
    const j = MAIN_STATES.indexOf(transitions[k].from as ProjectState);
    if (j >= 0) return j;
  }
  return 0;
}

export function stageStatus(tab: (typeof TABS)[number], state: ProjectState | null, store: ProjectStore): StageStatus {
  if (tab.kind === "tool") return "tool";
  const reached = reachedIndex(state, store.transitions);
  const idx = tab.states.map((s) => MAIN_STATES.indexOf(s));
  const lo = Math.min(...idx);
  const hi = Math.max(...idx);
  if (state === "delivered" && tab.id === "package") return "done";
  if (reached > hi) return "done";
  if (reached < lo) return "upcoming";
  if (state === "errored") return "failed";
  if (state === "proposed" || state === "label_review") return "attention";
  if (state === "no_fit" || state === "cancelled") return "failed";
  return "live";
}

function StageMark({ status, n }: { status: StageStatus; n: number }) {
  const base = "w-[18px] h-[18px] rounded-full grid place-items-center text-[10px] font-semibold shrink-0 transition-colors";
  if (status === "done") return <span className={cn(base, "bg-accent-soft text-accent")}><Check size={11} strokeWidth={3} /></span>;
  if (status === "live")
    return (
      <span className={cn(base, "bg-accent text-accent-ink shadow-[0_0_12px_rgba(var(--glow),0.9)]")}>
        <Loader2 size={11} className="animate-spin" />
      </span>
    );
  if (status === "attention") return <span className={cn(base, "bg-option text-white shadow-[0_0_12px_var(--option)] pulse-dot")}>!</span>;
  if (status === "failed") return <span className={cn(base, "bg-bad-soft text-bad")}><AlertTriangle size={10} /></span>;
  return <span className={cn(base, "border border-line text-muted")}>{n}</span>;
}

const STATUS_TEXT: Record<StageStatus, string> = {
  done: "done",
  live: "in progress",
  attention: "needs you",
  upcoming: "not started",
  failed: "stopped here",
  tool: "",
};

export const JourneyTabs = forwardRef<HTMLElement, { pid: string; active: string; state: ProjectState | null; store: ProjectStore }>(function JourneyTabs(
  { pid, active, state, store },
  ref,
) {
  const inner = useRef<HTMLElement | null>(null);
  const [light, setLight] = useState<{ left: number; width: number } | null>(null);
  useLayoutEffect(() => {
    const measure = () => {
      const el = inner.current?.querySelector<HTMLElement>(`[data-tab="${active}"]`);
      if (el) setLight({ left: el.offsetLeft + 6, width: el.offsetWidth - 12 });
    };
    measure();
    window.addEventListener("resize", measure);
    void document.fonts?.ready.then(measure);
    return () => window.removeEventListener("resize", measure);
  }, [active]);

  let stageNo = 0;
  return (
    <nav
      ref={(n) => {
        inner.current = n;
        if (typeof ref === "function") ref(n);
        else if (ref) ref.current = n;
      }}
      className="glass relative flex items-center gap-0.5 px-3 border-b border-line-soft overflow-x-auto shrink-0"
      role="tablist"
      aria-label="Experiment stages"
    >
      {TABS.map((t, i) => {
        const st = stageStatus(t, state, store);
        const isActive = active === t.id;
        if (t.kind === "stage") stageNo++;
        const prev = TABS[i - 1];
        const connector = t.kind === "stage" && prev?.kind === "stage";
        return (
          <div key={t.id} className="flex items-center">
            {connector && (
              <span aria-hidden className={cn("w-4 h-px mx-0.5 shrink-0", st === "upcoming" ? "bg-line" : "bg-accent/60")} />
            )}
            {t.kind === "tool" && i > 0 && <span aria-hidden className="w-px h-5 bg-line mx-2 shrink-0" />}
            <NavLink
              to={`/experiments/${pid}/${t.id}`}
              data-tab={t.id}
              role="tab"
              aria-selected={isActive}
              title={`${t.label}${STATUS_TEXT[st] ? ` — ${STATUS_TEXT[st]}` : ""} (${t.key})`}
              className={cn(
                "proximity relative inline-flex items-center gap-2 h-11 px-2.5 rounded-[8px] text-[13px] font-medium whitespace-nowrap transition-colors",
                isActive ? "text-ink" : st === "upcoming" ? "text-muted hover:text-ink-2" : "text-ink-2 hover:text-ink",
              )}
            >
              {t.kind === "stage" ? (
                <StageMark status={st} n={stageNo} />
              ) : t.id === "pipeline" ? (
                <LayoutDashboard size={14} className="text-muted" />
              ) : (
                <FlaskConical size={14} className="text-muted" />
              )}
              {t.label}
              <span className="sr-only">{STATUS_TEXT[st]}</span>
            </NavLink>
          </div>
        );
      })}
      {light && <span aria-hidden className="tab-light" style={{ left: light.left, width: light.width }} />}
    </nav>
  );
});

/** One line that answers "what is happening, and what should I do?" for the current state. */
export function NowStrip({ pid, state, store, detail, tab, goTab }: { pid: string; state: ProjectState | null; store: ProjectStore; detail?: ProjectDetail; tab: string; goTab: (t: string) => void }) {
  const qc = useQueryClient();
  const retry = useMutation({ mutationFn: () => api.retry(pid), onSettled: () => qc.invalidateQueries({ queryKey: qk.project(pid) }) });
  if (!state) return null;

  const run = store.runs.at(-1);
  const ep = run?.epochs.at(-1);
  const cur = store.curation;
  const err = store.errors.at(-1);
  const go = (t: string, label: string, icon?: ReactNode, variant: "primary" | "option" | "secondary" = "secondary") =>
    tab === t ? null : (
      <Button size="sm" variant={variant} icon={icon} onClick={() => goTab(t)}>
        {label}
        <ArrowRight size={13} className="-mr-0.5" />
      </Button>
    );

  let tone: "live" | "attention" | "good" | "bad" | "quiet" = "live";
  let text: ReactNode = null;
  let action: ReactNode = null;
  let progress: number | null = null;

  switch (state) {
    case "created":
    case "analyzing":
      text = <>The agent is reading the code to find one decision a small trained model could take over.</>;
      action = go("proposal", "Watch the proposal");
      break;
    case "proposed":
      tone = "attention";
      text = (
        <>
          <b className="font-semibold">Your move.</b> The agent has a proposal. Review it, then accept it or ask for changes.
        </>
      );
      action = go("proposal", "Review proposal", <ThumbsUp size={13} />, "option");
      break;
    case "accepted":
    case "curating":
      text = cur ? <>Building the dataset: {PHASE_TEXT[cur.phase] ?? cur.phase}{cur.total ? <span className="text-muted tnum"> ({cur.done} of {cur.total})</span> : null}</> : <>Building the dataset.</>;
      progress = cur && cur.total ? cur.done / cur.total : null;
      action = go("data", "Watch data");
      break;
    case "labelling": {
      const n = detail?.labels?.needs_label ?? 0;
      text = n ? <>The labelling agent is labelling {n} real rows; disagreements go to the adjudicator.</> : <>Settling the labels.</>;
      action = go("label", "Watch labels");
      break;
    }
    case "label_review": {
      tone = "attention";
      const n = detail?.labels?.needs_label ?? 0;
      text = n ? (
        <>
          <b className="font-semibold">{n} real rows wait for your review.</b> Label them, send them to the teacher, or exclude them.
        </>
      ) : (
        <>
          <b className="font-semibold">Every real row is labelled.</b> Continue when you're done reviewing.
        </>
      );
      action = <LabelGateActions pid={pid} detail={detail} onLabel={tab === "label" ? undefined : () => goTab("label")} />;
      break;
    }
    case "selecting":
      text = <>Recording the experiment's encoder for training.</>;
      action = go("training", "Watch selection");
      break;
    case "training":
      text = ep ? <>Training: epoch {ep.epoch} of {ep.epochs}.</> : <>Training is starting.</>;
      progress = ep ? ep.epoch / ep.epochs : null;
      action = go("training", "Watch training");
      break;
    case "evaluating":
    case "reviewing":
      text = state === "evaluating" ? <>Scoring the trained head on held-out test data.</> : <>Checking the results against the pass bar.</>;
      action = go("evaluation", "See evaluation");
      break;
    case "packaging":
      text = <>Packaging the model with integration instructions.</>;
      action = go("package", "See package");
      break;
    case "delivered":
      tone = "good";
      text = (
        <>
          <b className="font-semibold">Ready to integrate.</b> {store.manifest ? <>Package <code className="font-mono text-[12px]">{store.manifest.name}</code> passed the gate.</> : "The package passed the gate."}
        </>
      );
      action = go("package", "Open package", <PartyPopper size={13} />, "primary");
      break;
    case "errored":
      tone = "bad";
      text = (
        <>
          <b className="font-semibold">{err ? `${err.stage} failed.` : "A stage failed."}</b> {err?.message ? <span className="text-ink-2">{err.message}</span> : null}
        </>
      );
      action = (
        <>
          {tab !== "pipeline" && (
            <Button size="sm" variant="ghost" onClick={() => goTab("pipeline")}>
              Details
            </Button>
          )}
          {detail?.allowed_human_events.includes("retry") && (
            <Button size="sm" variant="primary" icon={<RotateCcw size={13} />} onClick={() => retry.mutate()} disabled={retry.isPending}>
              Retry
            </Button>
          )}
        </>
      );
      break;
    case "no_fit":
      tone = "quiet";
      text = <>The agent didn't find a decision worth training a model for. Ask for changes on the proposal, or start a new experiment.</>;
      action = go("proposal", "See why");
      break;
    case "cancelled":
      tone = "quiet";
      text = <>This run was cancelled.</>;
      break;
  }

  const dot = { live: "var(--accent)", attention: "var(--option)", good: "var(--good-mark)", bad: "var(--bad-mark)", quiet: "var(--muted)" }[tone];
  const Icon = tone === "quiet" ? CircleSlash : null;
  return (
    <div
      role="status"
      className={cn(
        "relative flex items-center gap-3 px-4 min-h-11 py-1.5 border-b border-line-soft text-[13px] overflow-hidden",
        tone === "attention" && "bg-option-soft",
        tone === "bad" && "bg-bad-soft",
        tone === "good" && "bg-good-soft",
      )}
    >
      {Icon ? (
        <Icon size={14} className="text-muted shrink-0" />
      ) : (
        <span className={cn("w-2 h-2 rounded-full shrink-0", (tone === "live" || tone === "attention") && "pulse-dot")} style={{ background: dot, boxShadow: `0 0 10px ${dot}` }} />
      )}
      <div className="min-w-0 flex-1 truncate text-ink">{text}</div>
      <div className="flex items-center gap-2 shrink-0">{action}</div>
      {progress != null && (
        <span aria-hidden className="absolute left-0 bottom-0 h-[2px] transition-[width] duration-700" style={{ width: `${Math.max(0.02, Math.min(1, progress)) * 100}%`, background: "var(--film)", boxShadow: "0 0 8px rgba(var(--glow),.8)" }} />
      )}
    </div>
  );
}
