import { useFocusView } from "../components/ui";
import { useEffect, useMemo, useRef, useState } from "react";
import { makeRamp } from "../lib/ramp";
import { useThemeVersion } from "../lib/theme";

export interface HeatmapProps {
  values: number[][];
  rowLabels: string[];
  colLabels: string[];
  domain: [number, number];
  colorOf: (l: string) => string;
  labelText?: (l: string) => string;
  format?: (v: number) => string;
  rowTitle?: string;
  colTitle?: string;
  cellTitle?: (r: string, c: string, v: number) => string;
  highlightDiagonal?: boolean;
  cell?: number;
  cellText?: (i: number, j: number, v: number) => string;
  onCellClick?: (r: string, c: string) => void;
  selected?: [string, string] | null;
}

/** Matrix heatmap on a continuous sequential ramp; fills transition so epoch changes animate. */
export function Heatmap({
  values,
  rowLabels,
  colLabels,
  domain,
  colorOf,
  labelText = (l) => l,
  format = (v) => v.toFixed(2),
  rowTitle,
  colTitle,
  cellTitle,
  highlightDiagonal = true,
  cell: maxCell = 58,
  cellText,
  onCellClick,
  selected,
}: HeatmapProps) {
  const tv = useThemeVersion();
  const ramp = useMemo(() => makeRamp(), [tv]);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [bw, setBw] = useState(400);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setBw(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const focus = useFocusView();
  const cap = focus ? Math.max(maxCell, Math.floor((window.innerHeight - 280) / Math.max(1, rowLabels.length))) : maxCell;
  const LW = bw < 360 ? 74 : 96;
  const fit = Math.floor((bw - LW) / Math.max(1, colLabels.length));
  const cell = Math.max(30, Math.min(cap, fit < 46 ? Math.floor((bw - LW - 40) / Math.max(1, colLabels.length)) : fit));
  const small = cell < 46;
  const TH = small ? 58 : 34;
  const gap = 2;
  const W = LW + colLabels.length * cell + (small ? 40 : 0);
  const H = TH + rowLabels.length * cell + (colTitle ? 0 : 0);
  const norm = (v: number) => (v - domain[0]) / Math.max(1e-9, domain[1] - domain[0]);
  const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  return (
    <div className="relative" ref={box}>
      <svg width={W} height={H + 4} viewBox={`0 0 ${W} ${H + 4}`} className="block max-w-full h-auto" role="img" aria-label={`${rowTitle ?? "rows"} by ${colTitle ?? "columns"} heatmap`} onMouseLeave={() => setHover(null)}>
        {colLabels.map((c, j) => {
          const cx = LW + j * cell + cell / 2;
          const active = hover?.[1] === j;
          return (
            <g key={c}>
              <rect x={cx - Math.min(14, cell / 2 - 4)} y={TH - 9} width={Math.min(28, cell - 8)} height={3} rx={1.5} fill={colorOf(c)} />
              {small ? (
                <text x={cx - 2} y={TH - 14} transform={`rotate(-38 ${cx - 2} ${TH - 14})`} fontSize="10.5" fill={active ? "var(--ink)" : "var(--ink-2)"} fontWeight={active ? 600 : 450}>
                  {trunc(labelText(c), 11)}
                </text>
              ) : (
                <text x={cx} y={TH - 14} textAnchor="middle" fontSize="10.5" fill={active ? "var(--ink)" : "var(--ink-2)"} fontWeight={active ? 600 : 450}>
                  {trunc(labelText(c), Math.floor(cell / 5.7))}
                </text>
              )}
            </g>
          );
        })}
        {rowLabels.map((r, i) => (
          <g key={r}>
            <rect x={2} y={TH + i * cell + cell / 2 - 4} width={8} height={8} rx={2} fill={colorOf(r)} />
            <text x={14} y={TH + i * cell + cell / 2 + 4} fontSize="11" fill={hover?.[0] === i ? "var(--ink)" : "var(--ink-2)"} fontWeight={hover?.[0] === i ? 600 : 450}>
              {trunc(labelText(r), LW < 90 ? 9 : 12)}
            </text>
            {colLabels.map((c, j) => {
              const v = values[i]?.[j] ?? 0;
              const t = norm(v);
              const diag = highlightDiagonal && r === c;
              return (
                <g key={c} onMouseEnter={() => setHover([i, j])} onClick={onCellClick ? () => onCellClick(r, c) : undefined} style={onCellClick ? { cursor: "pointer" } : undefined}>
                  <rect
                    x={LW + j * cell + gap / 2}
                    y={TH + i * cell + gap / 2}
                    width={cell - gap}
                    height={cell - gap}
                    rx={3}
                    fill={ramp.color(t)}
                    style={{ transition: "fill 420ms ease" }}
                  />
                  {diag && <rect x={LW + j * cell + gap / 2 + 1} y={TH + i * cell + gap / 2 + 1} width={cell - gap - 2} height={cell - gap - 2} rx={2.5} fill="none" stroke="var(--ink)" strokeOpacity={0.55} strokeWidth={1.25} />}
                  {((hover && hover[0] === i && hover[1] === j) || (selected && selected[0] === r && selected[1] === c)) && (
                    <rect x={LW + j * cell + 0.5} y={TH + i * cell + 0.5} width={cell - 1} height={cell - 1} rx={3.5} fill="none" stroke="var(--accent)" strokeWidth={2} />
                  )}
                  <text x={LW + j * cell + cell / 2} y={TH + i * cell + cell / 2 + 4} textAnchor="middle" fontSize={small ? 10 : 11} className="tnum" fill={ramp.dark(t) ? "#fff" : "#141820"} fontWeight={diag ? 600 : 400} pointerEvents="none">
                    {cellText ? cellText(i, j, v) : format(v)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <div className="flex items-center justify-between mt-1.5 text-[11px] text-muted gap-3">
        <span className="truncate">
          {hover
            ? cellTitle
              ? cellTitle(rowLabels[hover[0]], colLabels[hover[1]], values[hover[0]]?.[hover[1]] ?? 0)
              : `${labelText(rowLabels[hover[0]])} → ${labelText(colLabels[hover[1]])}: ${format(values[hover[0]]?.[hover[1]] ?? 0)}`
            : `${rowTitle ?? "rows"} × ${colTitle ?? "columns"}`}
        </span>
        <span className="flex items-center gap-1.5 shrink-0">
          <span className="tnum">{format(domain[0])}</span>
          <span className="w-16 h-2 rounded-sm" style={{ background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => ramp.color(t)).join(",")})` }} />
          <span className="tnum">{format(domain[1])}</span>
        </span>
      </div>
    </div>
  );
}
