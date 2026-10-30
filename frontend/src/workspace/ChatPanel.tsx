import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, ArrowUp, ChevronRight, ChevronsLeft, FileSearch, FileText, FolderSearch, Search, ShieldAlert, Sparkles, Wrench } from "lucide-react";
import { api } from "../api/client";
import type { ToolEv } from "../api/types";
import type { ChatItem } from "../store/projectStore";
import { Markdown } from "../components/Markdown";
import { ActorIcon, STATE_LABEL } from "../components/StateBadge";
import { clock, cn } from "../lib/format";
import { useWs } from "./context";

function toolIcon(name: string) {
  const n = name.toLowerCase();
  if (n === "read") return <FileText size={12} />;
  if (n === "grep") return <Search size={12} />;
  if (n === "glob") return <FolderSearch size={12} />;
  if (n === "think") return <Sparkles size={12} />;
  if (n.includes("search")) return <FileSearch size={12} />;
  return <Wrench size={12} />;
}

function ToolGroup({ steps, live }: { steps: ToolEv[]; live: boolean }) {
  const [open, setOpen] = useState(false);
  const last = steps[steps.length - 1];
  const counts = steps.reduce<Record<string, number>>((a, s) => ((a[s.name] = (a[s.name] ?? 0) + 1), a), {});
  return (
    <div className="ml-7 my-1">
      <button
        onClick={() => setOpen((o) => !o)}
        className="group w-full flex items-center gap-2 text-left text-[12px] text-ink-2 hover:text-ink py-1 cursor-pointer"
        aria-expanded={open}
      >
        <ChevronRight size={13} className={cn("shrink-0 transition-transform text-muted", open && "rotate-90")} />
        <span className="font-medium">
          {live ? "Exploring," : "Explored"} {steps.length} step{steps.length > 1 ? "s" : ""}
        </span>
        <span className="text-muted truncate">
          {Object.entries(counts)
            .map(([k, v]) => `${v} ${k.toLowerCase()}`)
            .join(", ")}
        </span>
      </button>
      {!open && live && (
        <div className="ml-5 flex items-center gap-1.5 text-[11.5px] text-muted truncate">
          {toolIcon(last.name)}
          <span className="font-mono truncate">{last.input}</span>
        </div>
      )}
      {open && (
        <ol className="ml-[6px] border-l border-line pl-3 space-y-1.5 py-1">
          {steps.map((s) => (
            <li key={s.seq} className="text-[11.5px] leading-snug">
              <div className="flex items-center gap-1.5 text-ink-2">
                <span className="text-muted">{toolIcon(s.name)}</span>
                <span className="font-medium">{s.name}</span>
                <span className="font-mono text-ink truncate">{s.input}</span>
              </div>
              {s.output && <div className="text-muted pl-[18px] break-words">{s.output}</div>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function StateChip({ item }: { item: Extract<ChatItem, { kind: "state" }> }) {
  const e = item.ev;
  return (
    <div className="flex items-center gap-2 my-2 text-[11.5px]" title={e.reason}>
      <span className="h-px flex-1 bg-line" />
      <span className="inline-flex items-center gap-1.5 px-2 h-5.5 rounded-full border border-line bg-panel text-ink-2 max-w-[85%]">
        <ActorIcon actor={e.actor} size={11} className="text-muted shrink-0" />
        <span className="truncate">
          <span className="text-muted">{e.event}</span> → <span className="font-medium text-ink">{STATE_LABEL[e.to] ?? e.to}</span>
        </span>
      </span>
      <span className="h-px flex-1 bg-line" />
    </div>
  );
}

/** Consecutive stage changes fold into one line: "6 stage changes: Curating data → Delivered". */
function StateRun({ items }: { items: Extract<ChatItem, { kind: "state" }>[] }) {
  const [open, setOpen] = useState(false);
  if (items.length < 3) return <>{items.map((c) => <StateChip key={`s${c.seq}`} item={c} />)}</>;
  const first = items[0].ev;
  const last = items[items.length - 1].ev;
  return (
    <div className="my-2">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full flex items-center gap-2 text-[11.5px] cursor-pointer group"
        title={open ? "Fold stage changes" : "Show each stage change"}
      >
        <span className="h-px flex-1 bg-line" />
        <span className="inline-flex items-center gap-1.5 px-2 h-5.5 rounded-full border border-line bg-panel text-ink-2 group-hover:text-ink group-hover:border-accent/40">
          <ChevronRight size={11} className={cn("text-muted transition-transform", open && "rotate-90")} />
          {items.length} stage changes:
          <span className="text-muted">{STATE_LABEL[first.from as keyof typeof STATE_LABEL] ?? first.from}</span> →{" "}
          <span className="font-medium text-ink">{STATE_LABEL[last.to] ?? last.to}</span>
        </span>
        <span className="h-px flex-1 bg-line" />
      </button>
      {open && items.map((c) => <StateChip key={`s${c.seq}`} item={c} />)}
    </div>
  );
}

export function ChatPanel({ onCollapse }: { onCollapse?: () => void }) {
  const { pid, store, detail } = useWs();
  const [text, setText] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const send = useMutation({
    mutationFn: (m: string) => api.chat(pid, m),
    onSuccess: () => setText(""),
  });

  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [store.chat]);

  const streaming = store.chat.some((c) => c.kind === "msg" && c.streaming);
  const placeholder =
    store.state === "proposed"
      ? "Ask about the proposal, or ask for changes…"
      : store.state === "delivered"
        ? "Ask about the result or the integration plan…"
        : "Message the agent…";
  const final = detail && ["delivered", "no_fit", "cancelled"].includes(detail.state);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between gap-2 pl-4 pr-2 h-10 border-b border-line-soft shrink-0">
        <span className="text-[12.5px] font-medium text-ink-2">Agent chat</span>
        {onCollapse && (
          <button onClick={onCollapse} aria-label="Collapse chat" title={`Collapse chat (${/Mac/.test(navigator.platform) ? "⌘" : "Ctrl+"}\\)`} className="p-1.5 rounded-[7px] text-muted hover:text-ink hover:bg-sunken cursor-pointer">
            <ChevronsLeft size={15} />
          </button>
        )}
      </div>
      <div
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
        }}
        className="flex-1 overflow-y-auto px-4 py-4"
      >
        {store.chat.length === 0 && (
          <div className="text-[13px] text-muted py-8 text-center">The agent's notes, tool steps and decisions appear here.</div>
        )}
        {store.chat.map((c, i) => {
          if (c.kind === "state") {
            // render a whole run of consecutive stage changes at its first item
            if (store.chat[i - 1]?.kind === "state") return null;
            const run: Extract<ChatItem, { kind: "state" }>[] = [];
            for (let k = i; k < store.chat.length && store.chat[k].kind === "state"; k++) run.push(store.chat[k] as Extract<ChatItem, { kind: "state" }>);
            return <StateRun key={`sr${c.seq}`} items={run} />;
          }
          if (c.kind === "tools") {
            const live = i === store.chat.length - 1 && store.state === "analyzing";
            return <ToolGroup key={c.key} steps={c.steps} live={live} />;
          }
          if (c.kind === "rejected")
            return (
              <div key={`r${c.seq}`} className="my-2 ml-7 rounded-[6px] border border-bad/30 bg-bad-soft px-2.5 py-2 text-[12px]">
                <div className="flex items-center gap-1.5 font-medium text-bad">
                  <ShieldAlert size={13} /> Rules layer rejected “{c.ev.event}” from {c.ev.actor}
                </div>
                <div className="text-ink-2 mt-0.5">{c.ev.error}</div>
              </div>
            );
          if (c.kind === "error")
            return (
              <div key={`e${c.seq}`} className="my-2 ml-7 rounded-[6px] border border-bad/30 bg-bad-soft px-2.5 py-2 text-[12px]">
                <div className="flex items-center gap-1.5 font-medium text-bad">
                  <AlertTriangle size={13} /> {c.ev.stage} failed
                </div>
                <div className="text-ink-2 mt-0.5 break-words">{c.ev.message}</div>
              </div>
            );
          if (c.role === "user")
            return (
              <div key={c.id} className="flex justify-end my-3">
                <div className="max-w-[86%] bg-accent-soft text-ink rounded-[10px] rounded-br-[3px] px-3 py-2 text-[13.5px] whitespace-pre-wrap break-words">
                  {c.content}
                </div>
              </div>
            );
          return (
            <div key={c.id} className="flex gap-2.5 my-3">
              <div className="w-[20px] h-[20px] mt-0.5 shrink-0 rounded-full grid place-items-center bg-sunken border border-line text-ink-2">
                <ActorIcon actor="agent" size={11} />
              </div>
              <div className="min-w-0 flex-1">
                <div className={cn("chat-md", c.streaming && "caret")}>
                  <Markdown className="chat-md">{c.content}</Markdown>
                </div>
                {!c.streaming && <div className="text-[10.5px] text-muted mt-1 tnum">{clock(c.ts)}</div>}
              </div>
            </div>
          );
        })}
      </div>
      <form
        className="p-3 border-t border-line"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim() && !send.isPending) send.mutate(text.trim());
        }}
      >
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (text.trim() && !send.isPending) send.mutate(text.trim());
              }
            }}
            rows={2}
            disabled={!!final && detail?.state === "cancelled"}
            placeholder={placeholder}
            className="w-full resize-none bg-raised border border-line rounded-[8px] pl-3 pr-11 py-2 text-[13.5px] focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 placeholder:text-muted/80"
          />
          <button
            type="submit"
            aria-label="Send"
            disabled={!text.trim() || send.isPending}
            className="absolute right-2 bottom-3 w-7 h-7 grid place-items-center rounded-[6px] bg-accent text-accent-ink disabled:opacity-35 cursor-pointer"
          >
            <ArrowUp size={15} />
          </button>
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted mt-1.5 px-0.5">
          <span>{streaming ? "Agent is writing…" : "Enter to send · Shift+Enter for a new line"}</span>
          {send.isError && <span className="text-bad">{(send.error as Error).message}</span>}
        </div>
      </form>
    </div>
  );
}
