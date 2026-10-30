import { useMemo, useRef, useState } from "react";
import type { ProjectState, StateEv } from "../api/types";
import { ActorIcon, STATE_LABEL } from "../components/StateBadge";
import { clock } from "../lib/format";

const W = 440;
const NW = 132;
const NH = 30;
const MX = 176; // main column centre
const SX = 362; // side column centre
const Y0 = 22;
const DY = 58;

const MAIN: ProjectState[] = ["created", "analyzing", "proposed", "accepted", "curating", "labelling", "selecting", "training", "evaluating", "reviewing", "packaging", "delivered"];
const POS: Record<string, { x: number; y: number }> = {};
MAIN.forEach((s, i) => (POS[s] = { x: MX, y: Y0 + i * DY + NH / 2 }));
POS.no_fit = { x: SX, y: Y0 + 4.5 * DY + NH / 2 };
POS.label_review = { x: SX, y: Y0 + 5.6 * DY + NH / 2 };
POS.errored = { x: SX, y: Y0 + 8 * DY + NH / 2 + 6 };
POS.cancelled = { x: SX, y: Y0 + 11 * DY + NH / 2 };
const H = Y0 + (MAIN.length - 1) * DY + NH + 22;

type Edge = { key: string; event: string; from: string; to: string; actor: "agent" | "human" | "system"; d: string; lx: number; ly: number; anchor?: "start" | "end" | "middle"; dynamic?: boolean };

const r = (s: string) => ({ ...POS[s], l: POS[s].x - NW / 2, rr: POS[s].x + NW / 2, t: POS[s].y - NH / 2, b: POS[s].y + NH / 2 });

function down(from: string, to: string, event: string, actor: Edge["actor"]): Edge {
  const a = r(from);
  const b = r(to);
  return { key: `${from}-${event}-${to}`, event, from, to, actor, d: `M${a.x},${a.b} L${b.x},${b.t - 1}`, lx: a.x + 9, ly: (a.b + b.t) / 2 + 4, anchor: "start" };
}
function backLeft(from: string, to: string, event: string, actor: Edge["actor"], off: number): Edge {
  const a = r(from);
  const b = r(to);
  const x = a.l - off;
  return {
    key: `${from}-${event}-${to}`,
    event,
    from,
    to,
    actor,
    d: `M${a.l},${a.y} C${x},${a.y} ${x},${b.y} ${b.l - 1},${b.y}`,
    lx: x - 4 + off * 0.25,
    ly: (a.y + b.y) / 2 + 4,
    anchor: "end",
  };
}
function side(from: string, to: string, event: string, actor: Edge["actor"], dynamic = false): Edge {
  const a = r(from);
  const b = r(to);
  const sx = a.rr;
  const tx = b.l - 1;
  const mx = (sx + tx) / 2;
  return {
    key: `${from}-${event}-${to}`,
    event,
    from,
    to,
    actor,
    d: `M${sx},${a.y} C${mx},${a.y} ${mx},${b.y} ${tx},${b.y}`,
    lx: dynamic ? mx : sx + 6,
    ly: dynamic ? (a.y + b.y) / 2 - 4 : a.y - 5,
    anchor: dynamic ? "middle" : "start",
    dynamic,
  };
}
function sideBack(from: string, to: string, event: string, actor: Edge["actor"]): Edge {
  // from side column back into main column
  const a = r(from);
  const b = r(to);
  const sx = a.l;
  const tx = b.rr + 1;
  const mx = (sx + tx) / 2;
  return { key: `${from}-${event}-${to}`, event, from, to, actor, d: `M${sx},${a.y + 6} C${mx},${a.y + 6} ${mx},${b.y + 6} ${tx},${b.y + 6}`, lx: mx, ly: (a.y + b.y) / 2 + 14, anchor: "middle", dynamic: true };
}

const STATIC: Edge[] = [
  down("created", "analyzing", "start", "system"),
  down("analyzing", "proposed", "propose", "agent"),
  down("proposed", "accepted", "accept", "human"),
  down("accepted", "curating", "begin_curation", "system"),
  down("curating", "labelling", "curated", "system"),
  down("labelling", "selecting", "labels_settled", "system"),
  side("labelling", "label_review", "needs_review", "system"),
  sideBack("label_review", "labelling", "review_done", "human"),
  down("selecting", "training", "selected", "system"),
  down("training", "evaluating", "trained", "system"),
  down("evaluating", "reviewing", "evaluated", "system"),
  down("reviewing", "packaging", "ship", "agent"),
  down("packaging", "delivered", "packaged", "system"),
  backLeft("proposed", "analyzing", "revise", "human", 42),
  backLeft("reviewing", "curating", "recurate", "agent", 62),
  side("analyzing", "no_fit", "declare_no_fit", "agent"),
  side("proposed", "no_fit", "declare_no_fit", "agent"),
  side("reviewing", "no_fit", "give_up", "agent"),
];

export function StateDiagram({ transitions, current }: { transitions: StateEv[]; current: ProjectState | null }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ edge: Edge; x: number; y: number } | null>(null);

  const { edges, taken, visited } = useMemo(() => {
    const taken = new Map<string, StateEv[]>();
    const visited = new Set<string>(["created"]);
    const edges = [...STATIC];
    const known = new Set(edges.map((e) => e.key));
    for (const t of transitions) {
      visited.add(t.from);
      visited.add(t.to);
      if (t.from === t.to) continue; // label_override: recorded, no state change, no edge
      let key = `${t.from}-${t.event}-${t.to}`;
      if (!known.has(key)) {
        if (!POS[t.from] || !POS[t.to]) continue;
        const inSide = (s: string) => s === "no_fit" || s === "errored" || s === "cancelled" || s === "label_review";
        const e = inSide(t.from) && !inSide(t.to) ? sideBack(t.from, t.to, t.event, t.actor) : side(t.from, t.to, t.event, t.actor, true);
        key = e.key;
        edges.push(e);
        known.add(key);
      }
      taken.set(key, [...(taken.get(key) ?? []), t]);
    }
    return { edges, taken, visited };
  }, [transitions]);

  const lastKey = transitions.length ? `${transitions[transitions.length - 1].from}-${transitions[transitions.length - 1].event}-${transitions[transitions.length - 1].to}` : null;

  const nodeStyle = (s: string) => {
    const isCur = s === current;
    const v = visited.has(s);
    if (isCur) {
      const fill = s === "delivered" ? "var(--good)" : s === "errored" ? "var(--bad)" : s === "no_fit" || s === "cancelled" ? "var(--ink-2)" : s === "proposed" || s === "label_review" ? "var(--option)" : "var(--accent)";
      return { fill, stroke: fill, text: s === "proposed" || s === "label_review" || s === "delivered" || s === "errored" ? "#fff" : "var(--accent-ink)", weight: 600 };
    }
    if (v) return { fill: "var(--accent-soft)", stroke: "color-mix(in oklab, var(--accent) 45%, transparent)", text: "var(--ink)", weight: 500 };
    return { fill: "var(--panel)", stroke: "var(--line)", text: "var(--muted)", weight: 450 };
  };

  return (
    <div ref={wrap} className="relative" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block" role="img" aria-label="Experiment state machine with the path taken highlighted">
        <defs>
          {(["base", "taken", "last"] as const).map((k) => (
            <marker key={k} id={`arr-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,1 L9,5 L0,9 z" fill={k === "base" ? "var(--axis)" : k === "taken" ? "color-mix(in oklab, var(--accent) 70%, var(--panel))" : "var(--accent)"} />
            </marker>
          ))}
        </defs>
        {/* any-state hints */}
        <text x={SX} y={POS.errored.y + NH / 2 + 14} textAnchor="middle" fontSize="10" fill="var(--muted)">
          fail · from any active state
        </text>
        <text x={SX} y={POS.cancelled.y + NH / 2 + 14} textAnchor="middle" fontSize="10" fill="var(--muted)">
          cancel · human, any non-final
        </text>
        {edges.map((e) => {
          const t = taken.get(e.key);
          const isLast = e.key === lastKey;
          if (e.dynamic && !t) return null;
          const stroke = isLast ? "var(--accent)" : t ? "color-mix(in oklab, var(--accent) 70%, var(--panel))" : "var(--axis)";
          return (
            <g key={e.key}>
              <path d={e.d} fill="none" stroke={stroke} strokeWidth={t ? 2 : 1} strokeDasharray={t ? undefined : "3 3"} markerEnd={`url(#arr-${isLast ? "last" : t ? "taken" : "base"})`} />
              <path
                d={e.d}
                fill="none"
                stroke="transparent"
                strokeWidth={14}
                style={{ cursor: "default" }}
                onMouseMove={(ev) => {
                  const b = wrap.current!.getBoundingClientRect();
                  setHover({ edge: e, x: ev.clientX - b.left, y: ev.clientY - b.top });
                }}
              />
              <text
                x={e.lx}
                y={e.ly}
                textAnchor={e.anchor}
                fontSize="10.5"
                fill={t ? "var(--ink-2)" : "var(--muted)"}
                fontWeight={t ? 550 : 400}
                pointerEvents="none"
              >
                {e.event}
                {t && t.length > 1 ? ` ×${t.length}` : ""}
              </text>
            </g>
          );
        })}
        {Object.keys(POS).map((s) => {
          const p = POS[s];
          const st = nodeStyle(s);
          const isCur = s === current;
          return (
            <g key={s}>
              {isCur && !["delivered", "no_fit", "cancelled"].includes(s) && (
                <rect x={p.x - NW / 2} y={p.y - NH / 2} width={NW} height={NH} rx={8} fill="none" stroke={st.fill} strokeWidth={2} className="pulse-ring" />
              )}
              <rect x={p.x - NW / 2} y={p.y - NH / 2} width={NW} height={NH} rx={8} fill={st.fill} stroke={st.stroke} strokeWidth={1} />
              <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="12" fontWeight={st.weight} fill={st.text}>
                {STATE_LABEL[s as ProjectState] ?? s}
              </text>
            </g>
          );
        })}
      </svg>
      {hover && (
        <div
          className="absolute z-20 pointer-events-none w-[280px] bg-raised border border-line rounded-[8px] p-2.5 text-[12px]"
          style={{ left: Math.min(hover.x + 14, (wrap.current?.clientWidth ?? 400) - 290), top: hover.y + 12, boxShadow: "var(--shadow-pop)" }}
        >
          <div className="flex items-center gap-1.5 font-semibold text-ink">
            {hover.edge.event}
            <span className="font-normal text-muted">
              {STATE_LABEL[hover.edge.from as ProjectState]} → {STATE_LABEL[hover.edge.to as ProjectState]}
            </span>
          </div>
          {(taken.get(hover.edge.key) ?? []).length ? (
            <ul className="mt-1.5 space-y-1.5">
              {taken.get(hover.edge.key)!.map((t) => (
                <li key={t.seq} className="leading-snug">
                  <span className="inline-flex items-center gap-1 text-ink-2">
                    <ActorIcon actor={t.actor} size={11} /> {t.actor} <span className="text-muted tnum">{clock(t.ts)}</span>
                  </span>
                  <div className="text-ink">{t.reason}</div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="mt-1 text-muted flex items-center gap-1">
              <ActorIcon actor={hover.edge.actor} size={11} /> fired by {hover.edge.actor} · not taken yet
            </div>
          )}
        </div>
      )}
    </div>
  );
}
