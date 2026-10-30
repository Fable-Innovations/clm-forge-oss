import { useState } from "react";
import type { ReliabilityBin } from "../api/types";

/** Reliability diagram: per-bin accuracy bars vs the perfect-calibration diagonal, with bin counts underneath. */
export function Reliability({ bins, ece }: { bins: ReliabilityBin[]; ece: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 340;
  const H = 260;
  const L = 36;
  const B = 58;
  const T = 10;
  const R = 10;
  const pw = W - L - R;
  const ph = H - T - B;
  const x = (v: number) => L + v * pw;
  const y = (v: number) => T + (1 - v) * ph;
  const maxC = Math.max(1, ...bins.map((b) => b.count));
  const hb = hover != null ? bins[hover] : null;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block" role="img" aria-label={`Reliability diagram, ECE ${ece.toFixed(3)}`} onMouseLeave={() => setHover(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={L - 6} y={y(t) + 3.5} textAnchor="end" fontSize="10" fill="var(--muted)">
              {Math.round(t * 100)}%
            </text>
            <text x={x(t)} y={T + ph + 14} textAnchor="middle" fontSize="10" fill="var(--muted)">
              {t.toFixed(t === 0 || t === 1 ? 0 : 2)}
            </text>
          </g>
        ))}
        {bins.map((b, i) => {
          const bw = (b.bin_hi - b.bin_lo) * pw;
          const x0 = x(b.bin_lo) + 1;
          const has = b.count > 0;
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={x0} y={T} width={bw - 2} height={ph + B - 6} fill="transparent" />
              {has && (
                <>
                  {/* gap between confidence and accuracy */}
                  <rect
                    x={x0 + 2}
                    y={Math.min(y(b.accuracy), y(b.confidence))}
                    width={bw - 6}
                    height={Math.abs(y(b.accuracy) - y(b.confidence))}
                    fill="var(--bad-mark)"
                    opacity={0.18}
                  />
                  <rect x={x0 + 2} y={y(b.accuracy)} width={bw - 6} height={Math.max(0, y(0) - y(b.accuracy))} rx={2} fill="var(--accent)" opacity={hover === i ? 1 : 0.85} />
                </>
              )}
              <rect x={x0 + 2} y={T + ph + 22 + (1 - b.count / maxC) * 22} width={bw - 6} height={(b.count / maxC) * 22} rx={1.5} fill="var(--axis)" />
            </g>
          );
        })}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--ink-2)" strokeDasharray="4 3" />
        <line x1={L} x2={W - R} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
        <text x={L + pw / 2} y={H - 2} textAnchor="middle" fontSize="10" fill="var(--muted)">
          predicted confidence · grey = number of test examples per bin
        </text>
      </svg>
      <div className="text-[11.5px] text-ink-2 h-4 mt-1 tnum">
        {hb ? (
          hb.count ? (
            <>
              confidence {hb.bin_lo.toFixed(1)}–{hb.bin_hi.toFixed(1)}: {hb.count} examples, mean confidence {(hb.confidence * 100).toFixed(0)}%, accuracy {(hb.accuracy * 100).toFixed(0)}%
            </>
          ) : (
            <>no test examples in {hb.bin_lo.toFixed(1)}–{hb.bin_hi.toFixed(1)}</>
          )
        ) : (
          <span className="text-muted">Bars on the dashed line = perfectly calibrated. Red = gap.</span>
        )}
      </div>
    </div>
  );
}
