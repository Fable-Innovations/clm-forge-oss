import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { NavLink, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, FolderGit2, MessageSquareText, MessagesSquare, RotateCcw, Square, WifiOff } from "lucide-react";
import { api } from "../api/client";
import { qk, useProject } from "../api/queries";
import { useProjectEvents } from "../hooks/useProjectEvents";
import { StateBadge } from "../components/StateBadge";
import { providerLabel } from "../components/ProviderPicker";
import { Button, Spinner } from "../components/ui";
import { EncoderSelect, encoderSourceText } from "../components/EncoderSelect";
import { labelOrder, makeColorOf } from "../lib/colors";
import { cn } from "../lib/format";
import { usePersisted } from "../lib/persisted";
import { Ctx, type WorkspaceCtx } from "../workspace/context";
import { ChatPanel } from "../workspace/ChatPanel";
import { JourneyTabs, NowStrip, TABS } from "../workspace/Journey";
import { PipelineTab } from "../workspace/PipelineTab";
import { ProposalTab } from "../workspace/ProposalTab";
import { DataTab } from "../workspace/DataTab";
import { TrainingTab } from "../workspace/TrainingTab";
import { EvaluationTab } from "../workspace/EvaluationTab";
import { PlaygroundTab } from "../workspace/PlaygroundTab";
import { PackageTab } from "../workspace/PackageTab";
import { LabelTab } from "../workspace/LabelTab";

const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

export function WorkspacePage() {
  const { pid = "", tab = "pipeline" } = useParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const { data: detail, isError, error } = useProject(pid);
  const { store, status } = useProjectEvents(pid);

  const task = useMemo(() => {
    if (detail?.task_schema) return detail.task_schema;
    const p = store.proposal ?? detail?.proposal;
    if (!p || !p.opportunities.length) return null;
    const id = detail?.accepted_opportunity_id ?? p.recommended_id;
    return (p.opportunities.find((o) => o.id === id) ?? p.opportunities[0]).task;
  }, [detail, store.proposal]);

  const labels = useMemo(() => {
    const fromSnap = store.runs.at(-1)?.snapshots[0]?.similarity.col_labels ?? store.report?.confusion.labels ?? [];
    return labelOrder(task, fromSnap);
  }, [task, store.runs, store.report]);

  const colorOf = useMemo(() => makeColorOf(labels), [labels]);
  const labelText = useCallback(
    (l: string) => {
      if (task?.type === "score" && Array.isArray(task.criteria)) {
        const c = task.criteria[Number(l)];
        return c ? `${l} · ${c.split(/\s+[—-]\s+/)[0]}` : l;
      }
      return l;
    },
    [task],
  );
  const goTab = useCallback((t: string) => nav(`/experiments/${pid}/${t}`), [nav, pid]);

  const ctx: WorkspaceCtx = { pid, store, detail, task, labels, colorOf, labelText, goTab };
  const state = store.state ?? detail?.state ?? null;
  // label_override is always on offer (recorded from the Labels tab); it is not a "your move" action
  const allowed = (detail?.allowed_human_events ?? []).filter((e) => e !== "label_override");

  const act = useMutation({
    mutationFn: (ev: "cancel" | "retry") => (ev === "cancel" ? api.cancel(pid) : api.retry(pid)),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.project(pid) }),
  });
  const setEncoder = useMutation({
    mutationFn: (key: string) => api.setProjectEncoder(pid, key),
    onSuccess: (d) => qc.setQueryData(qk.project(pid), d),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.encoders }),
  });

  // chat sidebar: collapsible to a rail, resizable by dragging its edge; both remembered
  const [chatOpen, setChatOpen] = usePersisted("chat:open", true);
  const [chatW, setChatW] = usePersisted("chat:width", 400);
  const [dragging, setDragging] = useState(false);
  const seenMsgs = useRef(0);
  const msgCount = store.chat.filter((c) => c.kind === "msg" && c.role === "assistant").length;
  if (chatOpen) seenMsgs.current = msgCount;
  const unread = !chatOpen && msgCount > seenMsgs.current;

  const startDrag = (e: React.PointerEvent) => {
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = chatW;
    setDragging(true);
    const move = (ev: PointerEvent) => setChatW(Math.round(Math.max(300, Math.min(Math.min(720, window.innerWidth * 0.55), w0 + ev.clientX - x0))));
    const up = () => {
      setDragging(false);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // keys: 1–8 jump between tabs (not on the Labels tab, where digits label rows), ⌘\ toggles the chat
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
        e.preventDefault();
        setChatOpen((o) => !o);
        return;
      }
      if (typing || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || document.querySelector("dialog[open], [role=dialog]")) return;
      if (tab === "label") return;
      const hit = TABS.find((x) => x.key === e.key);
      if (hit) goTab(hit.id);
    };
    const onToggle = () => setChatOpen((o) => !o);
    window.addEventListener("keydown", onKey);
    window.addEventListener("forge:toggle-chat", onToggle);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("forge:toggle-chat", onToggle);
    };
  }, [goTab, setChatOpen, tab]);

  if (isError)
    return (
      <div className="p-10 text-center text-ink-2">
        Couldn't load experiment: {(error as Error).message}.{" "}
        <NavLink to="/" className="text-accent underline">
          Back to experiments
        </NavLink>
      </div>
    );

  return (
    <Ctx.Provider value={ctx}>
      <div className="h-full grid grid-rows-[auto_1fr]">
        <div className="glass flex items-center gap-3 px-4 h-12 border-b border-line-soft min-w-0">
          <NavLink to="/" className="p-1 -ml-1 rounded text-muted hover:text-ink hover:bg-sunken" aria-label="Back to experiments">
            <ArrowLeft size={16} />
          </NavLink>
          <div className="min-w-0 flex items-baseline gap-2.5">
            <h1 className="font-display text-[15px] font-medium tracking-[-0.02em] truncate">{detail?.name ?? "…"}</h1>
            <span className="hidden lg:flex items-center gap-1 text-[12px] text-muted min-w-0">
              {detail?.repo ? <FolderGit2 size={12} className="shrink-0" /> : <MessageSquareText size={12} className="shrink-0" />}
              <span className={cn("truncate max-w-[360px]", detail?.repo && "font-mono text-[11.5px]")}>{detail?.repo ?? detail?.task}</span>
            </span>
          </div>
          <StateBadge state={state} />
          {status !== "open" && (
            <span className="inline-flex items-center gap-1 text-[12px] text-warn" title="Event stream disconnected; reconnecting with backoff">
              {status === "connecting" ? <Spinner size={12} /> : <WifiOff size={13} />}
              {status === "connecting" ? "Connecting" : "Reconnecting"}
            </span>
          )}
          <div className="flex-1" />
          {detail && (
            <span
              className="hidden lg:inline-flex items-center gap-1.5 text-[12px] text-muted"
              title={setEncoder.isError ? (setEncoder.error as Error).message : `The encoder this experiment trains on, from its next training stage. ${encoderSourceText(detail.encoder?.source)}`}
            >
              Encoder
              <EncoderSelect
                aria-label="Experiment encoder"
                className="w-[240px] h-7 text-[12px]"
                value={detail.encoder?.key ?? null}
                placeholder="Pick this experiment's encoder"
                disabled={setEncoder.isPending}
                onChange={(k) => setEncoder.mutate(k)}
              />
              {setEncoder.isError && <span className="text-bad max-w-[220px] truncate">{(setEncoder.error as Error).message}</span>}
            </span>
          )}
          {detail && (
            <span className="hidden xl:inline text-[12px] text-muted">
              {detail.model} on {providerLabel(detail.provider)}
              {detail.attempts > 0 && <> · attempt {detail.attempts}</>}
            </span>
          )}
          {allowed.includes("retry") && (
            <Button size="sm" variant="primary" icon={<RotateCcw size={13} />} onClick={() => act.mutate("retry")} disabled={act.isPending}>
              Retry
            </Button>
          )}
          {allowed.includes("cancel") && (
            <Button
              size="sm"
              variant="danger"
              icon={<Square size={11} />}
              disabled={act.isPending}
              onClick={() => {
                if (confirm("Cancel this experiment? Running jobs stop and the experiment becomes final.")) act.mutate("cancel");
              }}
            >
              Cancel run
            </Button>
          )}
        </div>
        <div className={cn("min-h-0 flex", dragging && "select-none cursor-col-resize")}>
          {chatOpen ? (
            <aside className="glass relative min-h-0 border-r border-line-soft shrink-0" style={{ width: chatW }} aria-label="Agent chat">
              <ChatPanel onCollapse={() => setChatOpen(false)} />
              <div
                role="separator"
                aria-orientation="vertical"
                aria-label="Resize chat"
                tabIndex={0}
                onPointerDown={startDrag}
                onDoubleClick={() => setChatW(400)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowLeft") setChatW((w) => Math.max(300, w - 24));
                  if (e.key === "ArrowRight") setChatW((w) => Math.min(720, w + 24));
                }}
                title="Drag to resize, double-click to reset"
                className="group absolute top-0 -right-[4px] w-[8px] h-full cursor-col-resize z-10 focus:outline-none"
              >
                <span className={cn("absolute left-[3px] top-0 h-full w-[2px] transition-colors", dragging ? "bg-accent shadow-[0_0_10px_rgba(var(--glow),.9)]" : "group-hover:bg-accent/60 group-focus-visible:bg-accent")} />
              </div>
            </aside>
          ) : (
            <aside className="glass min-h-0 w-12 shrink-0 border-r border-line-soft flex flex-col items-center py-3 gap-2" aria-label="Agent chat (collapsed)">
              <button
                onClick={() => setChatOpen(true)}
                title={`Open chat (${isMac ? "⌘" : "Ctrl+"}\\)`}
                aria-label="Open chat"
                className="proximity relative p-2 rounded-[10px] text-ink-2 hover:text-ink cursor-pointer"
              >
                <MessagesSquare size={18} />
                {unread && <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-accent pulse-dot" />}
              </button>
              <span className="[writing-mode:vertical-rl] rotate-180 text-[11.5px] text-muted mt-1 select-none">Agent chat</span>
            </aside>
          )}
          <section className="min-h-0 min-w-0 flex-1 flex flex-col">
            <JourneyTabs pid={pid} active={tab} state={state} store={store} />
            <NowStrip pid={pid} state={state} store={store} detail={detail} tab={tab} goTab={goTab} />
            <div className="flex-1 min-h-0 overflow-y-auto" key={tab}>
              {tab === "pipeline" && <PipelineTab />}
              {tab === "proposal" && <ProposalTab />}
              {tab === "data" && <DataTab />}
              {tab === "label" && <LabelTab />}
              {tab === "training" && <TrainingTab />}
              {tab === "evaluation" && <EvaluationTab />}
              {tab === "playground" && <PlaygroundTab />}
              {tab === "package" && <PackageTab />}
            </div>
          </section>
        </div>
      </div>
    </Ctx.Provider>
  );
}
