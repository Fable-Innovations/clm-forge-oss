import { useFocusView } from "../components/ui";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface Series {
  key: string;
  name: string;
  color: string;
  dashed?: boolean;
}

type Row = Record<string, number | null | undefined> & { epoch: number };

function Tip({ active, payload, label, series, fmt }: { active?: boolean; payload?: { dataKey?: string | number; value?: number }[]; label?: number | string; series: Series[]; fmt: (v: number) => string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-raised border border-line rounded-[6px] px-2 py-1.5 text-[11.5px]" style={{ boxShadow: "var(--shadow-pop)" }}>
      <div className="text-muted mb-0.5">epoch {label}</div>
      {series.map((s) => {
        const p = payload.find((x) => x.dataKey === s.key);
        if (!p || p.value == null) return null;
        return (
          <div key={s.key} className="flex items-center gap-1.5">
            <span className="w-2.5 h-[2px] inline-block" style={{ background: s.color }} />
            <span className="text-ink-2">{s.name}</span>
            <span className="ml-auto pl-3 tnum text-ink">{fmt(p.value)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function MiniChart({
  data,
  series,
  height = 170,
  yDomain,
  yFmt = (v) => v.toFixed(2),
  refY,
  refYLabel,
  scrubEpoch,
  onScrub,
  xMax,
  yScale,
}: {
  data: Row[];
  series: Series[];
  height?: number;
  yDomain?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"];
  yFmt?: (v: number) => string;
  refY?: number;
  refYLabel?: string;
  scrubEpoch?: number | null;
  onScrub?: (e: number) => void;
  xMax?: number;
  yScale?: "linear" | "log";
}) {
  const focus = useFocusView();
  return (
    <div style={{ height: focus ? Math.max(height, window.innerHeight - 260) : height }} className="w-full select-none">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 8, right: 14, bottom: 0, left: 0 }}
          onClick={(s: unknown) => {
            const st = s as { activeLabel?: number | string } | null;
            if (st && st.activeLabel != null && onScrub) onScrub(Number(st.activeLabel));
          }}
          style={{ cursor: onScrub ? "pointer" : undefined }}
        >
          <CartesianGrid vertical={false} strokeDasharray="0" />
          <XAxis dataKey="epoch" type="number" domain={[0, xMax ?? "dataMax"]} allowDecimals={false} tickLine={false} tickMargin={4} fontSize={11} height={20} />
          <YAxis domain={yDomain ?? ["auto", "auto"]} scale={yScale ?? "linear"} tickFormatter={yFmt} width={44} tickLine={false} axisLine={false} fontSize={11} tickCount={4} />
          <Tooltip content={(p) => <Tip {...(p as object)} series={series} fmt={yFmt} />} isAnimationActive={false} cursor={{ stroke: "var(--axis)", strokeWidth: 1 }} />
          {refY != null && (
            <ReferenceLine y={refY} stroke="var(--ink-2)" strokeDasharray="4 3" strokeWidth={1} label={refYLabel ? { value: refYLabel, position: "insideBottomRight", fontSize: 10.5, fill: "var(--ink-2)" } : undefined} />
          )}
          {scrubEpoch != null && <ReferenceLine x={scrubEpoch} stroke="var(--ink-2)" strokeOpacity={0.6} strokeWidth={1} />}
          {series.map((s) => (
            <Line
              key={s.key}
              dataKey={s.key}
              name={s.name}
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={s.dashed ? "5 3" : undefined}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--panel)" }}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
