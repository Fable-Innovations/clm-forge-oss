import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useLocation, useNavigate } from "react-router-dom";
import { ChevronsDownUp, ChevronsUpDown, CornerDownLeft, FlaskConical, FolderGit2, Layers, MessagesSquare, Moon, Plus, Search, Settings2, Sun } from "lucide-react";
import { useProjects } from "../api/queries";
import { StateBadge } from "./StateBadge";
import { cn } from "../lib/format";

const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
const Kbd = ({ children }: { children: ReactNode }) => (
  <kbd className="inline-flex items-center h-5 min-w-5 justify-center px-1.5 rounded-[5px] border border-line text-[11px] text-muted font-sans">{children}</kbd>
);

type Cmd = { id: string; title: string; hint?: string; icon: ReactNode; group: string; aside?: ReactNode; run: () => void };

import { TABS } from "../workspace/Journey";

/** Fuzzy subsequence score: every query char must appear in order; tighter, earlier matches score higher. */
function score(q: string, text: string): number {
  if (!q) return 1;
  const t = text.toLowerCase();
  let ti = 0;
  let s = 0;
  let run = 0;
  for (const ch of q.toLowerCase()) {
    const at = t.indexOf(ch, ti);
    if (at < 0) return 0;
    run = at === ti ? run + 1 : 0;
    s += 1 + run * 2 - Math.min(at - ti, 4) * 0.2 + (at === 0 || t[at - 1] === " " ? 2 : 0);
    ti = at + 1;
  }
  return s;
}

export function CommandPalette({ open, onClose, onToggleTheme, dark }: { open: boolean; onClose: () => void; onToggleTheme: () => void; dark: boolean }) {
  const nav = useNavigate();
  const loc = useLocation();
  const { data: projects } = useProjects();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQ("");
      setSel(0);
      requestAnimationFrame(() => input.current?.focus());
    }
  }, [open]);

  const pid = loc.pathname.match(/^\/experiments\/([^/]+)/)?.[1];

  const cmds = useMemo<Cmd[]>(() => {
    const go = (to: string) => () => nav(to);
    const out: Cmd[] = [];
    if (pid) {
      for (const t of TABS) out.push({ id: `tab-${t.id}`, group: "This experiment", title: `Go to ${t.label}`, icon: <Layers size={15} />, aside: <Kbd>{t.key}</Kbd>, run: go(`/experiments/${pid}/${t.id}`) });
      out.push(
        { id: "fold", group: "This experiment", title: "Collapse all panels", icon: <ChevronsDownUp size={15} />, run: () => window.dispatchEvent(new CustomEvent("forge:collapse-all", { detail: true })) },
        { id: "unfold", group: "This experiment", title: "Expand all panels", icon: <ChevronsUpDown size={15} />, run: () => window.dispatchEvent(new CustomEvent("forge:collapse-all", { detail: false })) },
      );
      out.push({ id: "chat", group: "This experiment", title: "Show or hide the agent chat", icon: <MessagesSquare size={15} />, aside: <Kbd>{isMac ? "⌘\\" : "Ctrl \\"}</Kbd>, run: () => window.dispatchEvent(new Event("forge:toggle-chat")) });
    }
    for (const p of projects ?? []) {
      out.push({
        id: `p-${p.id}`,
        group: "Experiments",
        title: p.name,
        hint: p.repo ?? p.task ?? undefined,
        icon: <FolderGit2 size={15} />,
        aside: <StateBadge state={p.state} />,
        run: go(`/experiments/${p.id}/pipeline`),
      });
    }
    out.push(
      { id: "new", group: "Actions", title: "New experiment", icon: <Plus size={15} />, run: () => nav("/?new=1") },
      { id: "projects", group: "Actions", title: "All experiments", icon: <FolderGit2 size={15} />, run: go("/") },
      { id: "play", group: "Actions", title: "Open playground", icon: <FlaskConical size={15} />, run: go("/playground") },
      { id: "settings", group: "Actions", title: "Open settings", icon: <Settings2 size={15} />, run: go("/settings") },
      { id: "theme", group: "Actions", title: dark ? "Switch to light theme" : "Switch to dark theme", icon: dark ? <Sun size={15} /> : <Moon size={15} />, run: onToggleTheme },
    );
    return out;
  }, [projects, pid, nav, dark, onToggleTheme]);

  const results = useMemo(() => {
    if (!q.trim()) return cmds;
    return cmds
      .map((c) => ({ c, s: Math.max(score(q, c.title), score(q, c.hint ?? "") * 0.6) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s)
      .map((x) => x.c);
  }, [cmds, q]);

  useEffect(() => setSel(0), [q]);
  useEffect(() => {
    list.current?.querySelector<HTMLElement>(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  if (!open) return null;

  const run = (c: Cmd | undefined) => {
    if (!c) return;
    onClose();
    c.run();
  };

  let lastGroup = "";
  return createPortal(
    <div className="fixed inset-0 z-[900] flex items-start justify-center pt-[14vh] px-4" onMouseDown={onClose}>
      <div className="absolute inset-0 bg-[color-mix(in_oklab,var(--bg)_55%,transparent)] backdrop-blur-[6px]" />
      <div
        role="dialog"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
        className="glass film-edge summon relative w-full max-w-[600px] rounded-[18px] overflow-visible"
        style={{ boxShadow: "var(--shadow-pop)" }}
      >
        <div className="flex items-center gap-3 px-4 h-14 border-b border-line">
          <Search size={17} className="text-accent shrink-0" />
          <input
            ref={input}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") (e.preventDefault(), setSel((i) => Math.min(results.length - 1, i + 1)));
              else if (e.key === "ArrowUp") (e.preventDefault(), setSel((i) => Math.max(0, i - 1)));
              else if (e.key === "Enter") (e.preventDefault(), run(results[sel]));
              else if (e.key === "Escape") (e.preventDefault(), onClose());
            }}
            placeholder="Jump to an experiment, tab or action"
            aria-label="Search commands"
            className="flex-1 bg-transparent text-[15px] text-ink placeholder:text-muted focus:outline-none"
          />
          <kbd className="text-[11px] text-muted border border-line rounded-[5px] px-1.5 h-5 inline-flex items-center">esc</kbd>
        </div>
        <div ref={list} className="max-h-[52vh] overflow-y-auto p-2" role="listbox">
          {results.length === 0 && <div className="px-3 py-8 text-center text-[13px] text-muted">Nothing matches “{q}”.</div>}
          {results.map((c, i) => {
            const head = c.group !== lastGroup ? (lastGroup = c.group) : null;
            return (
              <div key={c.id}>
                {head && <div className="px-3 pt-2.5 pb-1 text-[11.5px] text-muted">{head}</div>}
                <button
                  data-i={i}
                  role="option"
                  aria-selected={i === sel}
                  onMouseMove={() => setSel(i)}
                  onClick={() => run(c)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 h-10 rounded-[10px] text-left cursor-pointer transition-[background-color,box-shadow]",
                    i === sel ? "bg-accent-soft text-ink shadow-[inset_0_0_0_1px_rgba(var(--glow),0.35),0_0_22px_-8px_rgba(var(--glow),0.8)]" : "text-ink-2",
                  )}
                >
                  <span className={i === sel ? "text-accent" : "text-muted"}>{c.icon}</span>
                  <span className="text-[13.5px] font-medium truncate">{c.title}</span>
                  {c.hint && <span className="text-[12px] text-muted truncate min-w-0 flex-1">{c.hint}</span>}
                  {!c.hint && <span className="flex-1" />}
                  {c.aside}
                  {i === sel && <CornerDownLeft size={13} className="text-muted shrink-0" />}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>,
    document.body,
  );
}
