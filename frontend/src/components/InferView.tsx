import type { InferResult, TaskSchema } from "../api/types";
import { ProbBars } from "../viz/ProbBars";
import { ms } from "../lib/format";
import { Spinner } from "./ui";

export function probsOf(r: InferResult): Record<string, number> {
  const a = r.answer;
  if (a.type === "noul") return { false: 1 - a.noul, true: a.noul };
  return a.probabilities;
}

export function AnswerHeadline({ r, task, labelText }: { r: InferResult; task: TaskSchema | null; labelText: (l: string) => string }) {
  const a = r.answer;
  if (a.type === "noul")
    return (
      <div>
        <span className="text-[22px] font-semibold">{a.noul >= 0.5 ? "yes" : "no"}</span>
        <span className="text-[13px] text-ink-2 ml-2 tnum">P(true) = {a.noul.toFixed(3)}</span>
      </div>
    );
  if (a.type === "choice")
    return (
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-[22px] font-semibold">{labelText(a.choice)}</span>
        <span className="text-[13px] text-ink-2 tnum">confidence {a.confidence.toFixed(2)}</span>
        {task?.type === "choice" && task.criteria && !Array.isArray(task.criteria) && (
          <span className="text-[12px] text-muted basis-full truncate">{task.criteria[a.choice]}</span>
        )}
      </div>
    );
  return (
    <div className="flex items-baseline gap-2 flex-wrap">
      <span className="text-[22px] font-semibold tnum">{a.score.toFixed(2)}</span>
      <span className="text-[13px] text-ink-2">expected level · confidence {a.confidence.toFixed(2)}</span>
      <span className="text-[12px] text-muted basis-full truncate">nearest: {a.legend[String(Math.round(a.score))]}</span>
    </div>
  );
}

export function LatencyBreakdown({ r }: { r: InferResult }) {
  const other = Math.max(0, r.latency_ms - r.embed_ms - r.head_ms);
  const total = r.latency_ms || 1;
  const parts = [
    { k: "encoder", v: r.embed_ms, c: "var(--c1)" },
    { k: "head", v: r.head_ms, c: "var(--c3)" },
    { k: "other", v: other, c: "var(--axis)" },
  ];
  return (
    <div>
      <div className="flex items-baseline justify-between text-[12px] mb-1">
        <span className="text-muted">latency</span>
        <span className="tnum font-semibold">{ms(r.latency_ms)}</span>
      </div>
      <div className="flex h-[8px] gap-[2px]">
        {parts.map((p) => (
          <span key={p.k} style={{ flexGrow: Math.max(p.v, total * 0.012), flexBasis: 0, background: p.c }} className="first:rounded-l-[2px] last:rounded-r-[2px]" title={`${p.k} ${ms(p.v)}`} />
        ))}
      </div>
      <div className="flex gap-3 text-[11px] text-ink-2 mt-1 tnum">
        {parts.map((p) => (
          <span key={p.k} className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-[2px]" style={{ background: p.c }} />
            {p.k} {ms(p.v)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function InferCard({
  title,
  sub,
  result,
  loading,
  error,
  task,
  order,
  colorOf,
  labelText,
}: {
  title: string;
  sub: string;
  result: InferResult | undefined;
  loading: boolean;
  error?: string | null;
  task: TaskSchema | null;
  order: string[];
  colorOf: (l: string) => string;
  labelText: (l: string) => string;
}) {
  return (
    <div className="bg-panel border border-line rounded-[10px] p-4 min-w-0">
      <div className="flex items-start justify-between gap-2 mb-3">
        <div>
          <h3 className="text-[13.5px] font-semibold">{title}</h3>
          <p className="text-[11.5px] text-muted">{sub}</p>
        </div>
        {loading && <Spinner />}
      </div>
      {error ? (
        <p className="text-[12.5px] text-muted">{error}</p>
      ) : result ? (
        <div className="space-y-4">
          <AnswerHeadline r={result} task={task} labelText={labelText} />
          <ProbBars probs={probsOf(result)} order={order} colorOf={colorOf} labelText={labelText} />
          <LatencyBreakdown r={result} />
        </div>
      ) : (
        <p className="text-[12.5px] text-muted">Type a state to score it.</p>
      )}
    </div>
  );
}
