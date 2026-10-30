import { useFocusView } from "../components/ui";
import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { SnapshotPoint } from "../api/types";
import type { AlignedSnapshot } from "./align";
import { cssVar } from "../lib/colors";
import { useThemeVersion } from "../lib/theme";

const resolve = (c: string) => {
  const m = c.match(/^var\((--[^)]+)\)$/);
  return m ? cssVar(m[1]) : c;
};
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

export function EmbeddingScatter({
  snaps,
  index,
  extent,
  colorOf,
  labelText,
  focus,
  onFocus,
}: {
  snaps: AlignedSnapshot[];
  index: number;
  extent: number;
  colorOf: (l: string) => string;
  labelText: (l: string) => string;
  focus: string | null;
  onFocus?: (l: string | null) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(560);
  const inFocus = useFocusView();
  const h = Math.round(inFocus ? Math.max(300, Math.min(w * 0.74, window.innerHeight - 300)) : Math.min(520, Math.max(300, w * 0.74)));
  const tv = useThemeVersion();
  const drawn = useRef<{ x: number; y: number }[]>([]);
  const anim = useRef<number | null>(null);
  const [hover, setHover] = useState<{ p: SnapshotPoint; x: number; y: number } | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pal = useMemo(() => {
    void tv;
    return { panel: cssVar("--panel-solid"), ink: cssVar("--ink"), muted: cssVar("--muted"), grid: cssVar("--grid"), axis: cssVar("--axis") };
  }, [tv]);
  const colorCache = useMemo(() => {
    void tv;
    const m = new Map<string, string>();
    return (l: string) => {
      let c = m.get(l);
      if (!c) {
        c = resolve(colorOf(l));
        m.set(l, c);
      }
      return c;
    };
  }, [colorOf, tv]);

  const snap = snaps[Math.max(0, Math.min(snaps.length - 1, index))];
  const pad = 26;
  const sx = (x: number) => pad + ((x + extent) / (2 * extent)) * (w - 2 * pad);
  const sy = (y: number) => h - pad - ((y + extent) / (2 * extent)) * (h - 2 * pad);

  // option trails up to current epoch
  const trails = useMemo(() => {
    const t = new Map<string, { x: number; y: number }[]>();
    for (let i = 0; i <= index && i < snaps.length; i++) {
      for (const p of snaps[i].points) {
        if (p.kind !== "option") continue;
        const arr = t.get(p.label) ?? [];
        arr.push({ x: p.x, y: p.y });
        t.set(p.label, arr);
      }
    }
    return t;
  }, [snaps, index]);

  const draw = (pos: { x: number; y: number }[]) => {
    const c = canvas.current;
    if (!c || !snap) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    // origin cross
    g.strokeStyle = pal.grid;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(sx(0), pad / 2);
    g.lineTo(sx(0), h - pad / 2);
    g.moveTo(pad / 2, sy(0));
    g.lineTo(w - pad / 2, sy(0));
    g.stroke();

    const pts = snap.points;
    // trails
    g.lineWidth = 1.5;
    g.setLineDash([2, 3]);
    trails.forEach((tr, label) => {
      if (tr.length < 2) return;
      g.strokeStyle = colorCache(label);
      g.globalAlpha = focus && focus !== label ? 0.1 : 0.55;
      g.beginPath();
      tr.forEach((p, i) => (i ? g.lineTo(sx(p.x), sy(p.y)) : g.moveTo(sx(p.x), sy(p.y))));
      g.stroke();
    });
    g.setLineDash([]);
    g.globalAlpha = 1;

    // states
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (p.kind !== "state") continue;
      const q = pos[i] ?? p;
      const X = sx(q.x);
      const Y = sy(q.y);
      g.globalAlpha = focus && focus !== p.label ? 0.1 : 0.92;
      g.beginPath();
      g.arc(X, Y, 5, 0, Math.PI * 2);
      g.fillStyle = pal.panel;
      g.fill();
      g.beginPath();
      g.arc(X, Y, 3.6, 0, Math.PI * 2);
      g.fillStyle = colorCache(p.label);
      g.fill();
    }
    g.globalAlpha = 1;
    // options on top
    g.font = "600 12px 'Instrument Sans Variable', system-ui, sans-serif";
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (p.kind !== "option") continue;
      const q = pos[i] ?? p;
      const X = sx(q.x);
      const Y = sy(q.y);
      const dim = focus && focus !== p.label;
      g.globalAlpha = dim ? 0.25 : 1;
      // diamond marker
      const r = 9;
      g.beginPath();
      g.moveTo(X, Y - r - 2.5);
      g.lineTo(X + r + 2.5, Y);
      g.lineTo(X, Y + r + 2.5);
      g.lineTo(X - r - 2.5, Y);
      g.closePath();
      g.fillStyle = pal.panel;
      g.fill();
      g.beginPath();
      g.moveTo(X, Y - r);
      g.lineTo(X + r, Y);
      g.lineTo(X, Y + r);
      g.lineTo(X - r, Y);
      g.closePath();
      g.fillStyle = colorCache(p.label);
      g.fill();
      g.lineWidth = 1.5;
      g.strokeStyle = pal.ink;
      g.stroke();
      const txt = labelText(p.label);
      const tw = g.measureText(txt).width;
      const right = X + 14 + tw < w - 4;
      const tx = right ? X + 14 : X - 14 - tw;
      g.lineWidth = 4;
      g.strokeStyle = pal.panel;
      g.lineJoin = "round";
      g.strokeText(txt, tx, Y + 4);
      g.fillStyle = pal.ink;
      g.fillText(txt, tx, Y + 4);
    }
    g.globalAlpha = 1;
  };

  // tween when snapshot changes
  useEffect(() => {
    if (!snap) return;
    const target = snap.points.map((p) => ({ x: p.x, y: p.y }));
    const from = drawn.current.length === target.length ? drawn.current.slice() : target;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (anim.current) cancelAnimationFrame(anim.current);
    const t0 = performance.now();
    const D = reduce ? 0 : 460;
    const step = (now: number) => {
      const t = D ? Math.min(1, (now - t0) / D) : 1;
      const e = ease(t);
      const cur = target.map((q, i) => ({ x: from[i].x + (q.x - from[i].x) * e, y: from[i].y + (q.y - from[i].y) * e }));
      drawn.current = cur;
      draw(cur);
      if (t < 1) anim.current = requestAnimationFrame(step);
      else anim.current = null;
    };
    anim.current = requestAnimationFrame(step);
    return () => {
      if (anim.current) cancelAnimationFrame(anim.current);
    };
  }, [snap, w, h, extent, pal, focus, trails, colorCache]);

  const onMove = (e: ReactMouseEvent) => {
    if (!snap) return;
    const b = canvas.current!.getBoundingClientRect();
    const mx = e.clientX - b.left;
    const my = e.clientY - b.top;
    let best: { p: SnapshotPoint; d: number; x: number; y: number } | null = null;
    snap.points.forEach((p, i) => {
      const q = drawn.current[i] ?? p;
      const X = sx(q.x);
      const Y = sy(q.y);
      const d = Math.hypot(X - mx, Y - my) - (p.kind === "option" ? 6 : 0);
      if (d < 10 && (!best || d < best.d)) best = { p, d, x: X, y: Y };
    });
    const bb = best as { p: SnapshotPoint; d: number; x: number; y: number } | null;
    setHover(bb ? { p: bb.p, x: bb.x, y: bb.y } : null);
  };

  return (
    <div ref={wrap} className="relative w-full">
      <canvas
        ref={canvas}
        style={{ width: w, height: h }}
        className="block rounded-[6px] bg-panel"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={() => onFocus?.(hover ? (focus === hover.p.label ? null : hover.p.label) : null)}
        role="img"
        aria-label="2-D PCA of projected validation states and option embeddings"
      />
      {hover && (
        <div
          className="absolute z-10 pointer-events-none max-w-[300px] bg-raised border border-line rounded-[8px] px-2.5 py-2 text-[12px]"
          style={{ left: Math.min(hover.x + 12, w - 310), top: Math.min(hover.y + 12, h - 40), boxShadow: "var(--shadow-pop)" }}
        >
          <div className="flex items-center gap-1.5 font-medium">
            <span className="w-2 h-2 rounded-[2px]" style={{ background: colorOf(hover.p.label) }} />
            {hover.p.kind === "option" ? `Option: ${labelText(hover.p.label)}` : `Gold label: ${labelText(hover.p.label)}`}
          </div>
          {hover.p.kind === "state" && <div className="text-ink-2 mt-1 whitespace-pre-wrap line-clamp-5">{hover.p.text}</div>}
        </div>
      )}
    </div>
  );
}
