import { int } from "../lib/format";

/** Horizontal bar list: thin bars with value at the tip, label in text ink, colour swatch carries identity. */
export function HBars({
  data,
  colorOf,
  max,
  labelWidth = 92,
  format = int,
}: {
  data: { key: string; label?: string; value: number }[];
  colorOf: (k: string) => string;
  max?: number;
  labelWidth?: number;
  format?: (n: number) => string;
}) {
  const m = max ?? Math.max(1, ...data.map((d) => d.value));
  return (
    <div className="space-y-[7px]" role="list">
      {data.map((d) => (
        <div key={d.key} role="listitem" className="grid items-center gap-2 text-[12.5px]" style={{ gridTemplateColumns: `${labelWidth}px 1fr` }} title={`${d.label ?? d.key}: ${format(d.value)}`}>
          <span className="truncate text-ink-2 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-[2px] shrink-0" style={{ background: colorOf(d.key) }} />
            {d.label ?? d.key}
          </span>
          <span className="flex items-center gap-2 min-w-0">
            <span className="h-[12px] rounded-r-[3px] transition-[width] duration-500" style={{ width: `${(d.value / m) * 100}%`, minWidth: d.value ? 2 : 0, background: colorOf(d.key), maxWidth: "calc(100% - 44px)" }} />
            <span className="tnum text-[12px] text-ink shrink-0">{format(d.value)}</span>
          </span>
        </div>
      ))}
    </div>
  );
}

/** One stacked bar with 2px surface gaps and legend underneath. */
export function StackBar({ data, colorOf }: { data: { key: string; value: number; label?: string }[]; colorOf: (k: string) => string }) {
  const total = data.reduce((a, d) => a + d.value, 0) || 1;
  return (
    <div>
      <div className="flex h-[14px] gap-[2px]">
        {data.map((d, i) => (
          <span
            key={d.key}
            title={`${d.label ?? d.key}: ${int(d.value)} (${((d.value / total) * 100).toFixed(0)}%)`}
            className="h-full transition-[flex-grow] duration-500"
            style={{
              flexGrow: d.value,
              flexBasis: 0,
              background: colorOf(d.key),
              borderRadius: `${i === 0 ? 3 : 0}px ${i === data.length - 1 ? 3 : 0}px ${i === data.length - 1 ? 3 : 0}px ${i === 0 ? 3 : 0}px`,
            }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[12px]">
        {data.map((d) => (
          <span key={d.key} className="inline-flex items-center gap-1.5 text-ink-2">
            <span className="w-2 h-2 rounded-[2px]" style={{ background: colorOf(d.key) }} />
            {d.label ?? d.key} <span className="text-ink tnum">{int(d.value)}</span>
            <span className="text-muted tnum">{((d.value / total) * 100).toFixed(0)}%</span>
          </span>
        ))}
      </div>
    </div>
  );
}
