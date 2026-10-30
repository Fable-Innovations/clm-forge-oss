import { cn } from "../lib/format";

/** Probability distribution over options; the argmax is emphasised, gold optionally marked. */
export function ProbBars({
  probs,
  order,
  colorOf,
  labelText = (l) => l,
  gold,
  compact,
  labelWidth = 96,
}: {
  probs: Record<string, number>;
  order?: string[];
  colorOf: (l: string) => string;
  labelText?: (l: string) => string;
  gold?: string;
  compact?: boolean;
  labelWidth?: number;
}) {
  const keys = order?.length ? order.filter((k) => k in probs).concat(Object.keys(probs).filter((k) => !order.includes(k))) : Object.keys(probs);
  const top = keys.reduce((a, k) => (probs[k] > (probs[a] ?? -1) ? k : a), keys[0]);
  return (
    <div className={cn(compact ? "space-y-[3px]" : "space-y-1.5")}>
      {keys.map((k) => {
        const v = probs[k] ?? 0;
        return (
          <div key={k} className="grid items-center gap-2" style={{ gridTemplateColumns: `${labelWidth}px 1fr 44px` }} title={`${labelText(k)}: ${(v * 100).toFixed(1)}%`}>
            <span className={cn("truncate flex items-center gap-1.5", compact ? "text-[11.5px]" : "text-[12.5px]", k === top ? "text-ink font-medium" : "text-ink-2")}>
              <span className="w-2 h-2 rounded-[2px] shrink-0" style={{ background: colorOf(k) }} />
              {labelText(k)}
              {gold === k && <span className="text-[10px] text-good font-semibold">gold</span>}
            </span>
            <span className={cn("relative rounded-[3px] bg-sunken", compact ? "h-[8px]" : "h-[12px]")}>
              <span className="absolute inset-y-0 left-0 rounded-[3px] transition-[width] duration-300" style={{ width: `${v * 100}%`, background: colorOf(k), opacity: k === top ? 1 : 0.55 }} />
            </span>
            <span className={cn("tnum text-right", compact ? "text-[11px]" : "text-[12px]", k === top ? "text-ink font-semibold" : "text-ink-2")}>{(v * 100).toFixed(v < 0.1 ? 1 : 0)}%</span>
          </div>
        );
      })}
    </div>
  );
}
