import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, CircleX, X } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { EvalReport, ModelScore } from "../api/types";
import { Chip, Empty, InfoTip, Panel, Segmented, Spinner } from "../components/ui";
import { Heatmap } from "../viz/Heatmap";
import { Reliability } from "../viz/Reliability";
import { ProbBars } from "../viz/ProbBars";
import { cn, ms, pct } from "../lib/format";
import { useWs } from "./context";

const MODEL_META: Record<string, { name: string; color: string; desc: string }> = {
  finetuned: { name: "Fine-tuned head", color: "var(--accent)", desc: "this experiment's trained head" },
  reference: { name: "Zero-shot CLM", color: "var(--ink-2)", desc: "reference CLM-v0.1 head, no training" },
  majority: { name: "Majority class", color: "var(--muted)", desc: "always the most common label" },
  teacher: { name: "Teacher LLM", color: "var(--option)", desc: "the LLM that labelled the data" },
};

function GateBanner({ r }: { r: EvalReport }) {
  const ok = r.gate_passed;
  return (
    <div className={cn("rounded-[10px] border px-5 py-4 flex gap-4 items-start", ok ? "bg-good-soft border-good/30" : "bg-bad-soft border-bad/30")}>
      {ok ? <CheckCircle2 size={26} className="text-good shrink-0 mt-0.5" /> : <CircleX size={26} className="text-bad shrink-0 mt-0.5" />}
      <div className="min-w-0 flex-1">
        <h2 className={cn("text-[17px] font-semibold", ok ? "text-good" : "text-bad")}>{ok ? "Gate passed: the head can ship" : "Gate failed: the head can't ship"}
          <InfoTip className="ml-2 align-[1px]">
            The ship gate is an automatic pass/fail check on the held-out test split, data the model never saw while training. It passes only if all three checks below pass: the fine-tuned head reaches the pass bar, beats the untrained zero-shot head, and beats always guessing the most common label.
          </InfoTip>
        </h2>
        <ul className="mt-1.5 space-y-0.5">
          {r.gate_reasons.map((g) => {
            const pass = /^\s*(✓|PASS:)/.test(g);
            const fail = /^\s*(✗|FAIL:)/.test(g);
            const text = g.replace(/^\s*(✓|✗|PASS:|FAIL:)\s*/, "");
            return (
              <li key={g} className="flex items-center gap-2 text-[13px] text-ink">
                {pass ? <CheckCircle2 size={14} className="text-good shrink-0" /> : fail ? <X size={14} className="text-bad shrink-0" /> : <span className="w-[14px]" />}
                {text}
              </li>
            );
          })}
        </ul>
      </div>
      <div className="text-right shrink-0">
        <div className="text-[12px] text-ink-2 inline-flex items-center gap-1">
          test {r.metric === "macro_f1" ? "macro F1" : r.metric}
          <InfoTip>
            {r.metric === "macro_f1"
              ? "The fine-tuned head's macro F1 on the test split: F1 is computed for each label separately and then averaged, so rare labels count as much as common ones. 1.0 is perfect."
              : "The fine-tuned head's accuracy on the test split: the share of test examples it labelled correctly. 1.0 is perfect."}{" "}
            “bar” is the pass threshold this score must reach, and n is the number of test examples it was measured on. With a small n, the score can move a lot from one run to the next.
          </InfoTip>
        </div>
        <div className="font-display text-[30px] font-light tnum leading-none mt-1 tracking-[-0.03em]">{(r.models.find((m) => m.name === "finetuned")?.[r.metric === "macro_f1" ? "macro_f1" : "accuracy"] ?? 0).toFixed(3)}</div>
        <div className="text-[11.5px] text-ink-2 mt-1 tnum">bar {r.threshold.toFixed(2)} · n = {r.n_test}</div>
      </div>
    </div>
  );
}

function ModelBars({ models, metric, threshold }: { models: ModelScore[]; metric: "accuracy" | "macro_f1"; threshold: number }) {
  const order = ["finetuned", "teacher", "reference", "majority"];
  const ms_ = models.slice().sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
  return (
    <div className="relative">
      <div className="space-y-3">
        {ms_.map((m) => {
          const meta = MODEL_META[m.name] ?? { name: m.name, color: "var(--muted)", desc: "" };
          const v = m[metric];
          return (
            <div key={m.name} className="grid grid-cols-[150px_1fr] gap-3 items-center" title={m.note ?? meta.desc}>
              <div className="min-w-0">
                <div className="text-[13px] font-medium truncate flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-[2px]" style={{ background: meta.color }} />
                  {meta.name}
                </div>
                <div className="text-[11px] text-muted truncate">{m.note ?? meta.desc} · n={m.n}</div>
              </div>
              <div className="relative h-[18px]">
                <div className="absolute inset-y-[3px] left-0 rounded-r-[4px] transition-[width] duration-500" style={{ width: `calc((100% - 52px) * ${v})`, background: meta.color }} />
                <span className="absolute top-0 text-[12.5px] font-semibold tnum" style={{ left: `calc((100% - 52px) * ${v} + 6px)` }}>
                  {v.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="absolute top-[-6px] bottom-[-6px] pointer-events-none" style={{ left: `calc(162px + (100% - 214px) * ${threshold})` }}>
        <div className="h-full border-l-[1.5px] border-dashed border-ink-2" />
        <div className="absolute -bottom-4 -translate-x-1/2 text-[10.5px] text-ink-2 whitespace-nowrap">bar {threshold.toFixed(2)}</div>
      </div>
    </div>
  );
}

function LatencyPanel({ r, encoderName }: { r: EvalReport; encoderName: string | null }) {
  const teacher = r.models.find((m) => m.name === "teacher");
  const parsed = teacher?.note?.match(/([\d,]+(?:\.\d+)?)\s*ms/);
  const llm = parsed ? Number(parsed[1].replace(/,/g, "")) : 1000;
  const llmAssumed = !parsed;
  const L = r.latency;
  const full = L.embed_ms_p50 + L.head_ms_p50;
  const rows = [
    { k: "LLM call", v: llm, p95: null as number | null, color: "var(--option)", note: llmAssumed ? "typical hosted LLM call (assumed)" : "teacher LLM, measured p50" },
    { k: "CLM, uncached", v: full, p95: L.embed_ms_p95 + L.head_ms_p95, color: "var(--accent)", note: "encoder + head" },
    { k: "Encoder only", v: L.embed_ms_p50, p95: L.embed_ms_p95, color: "var(--c1)", note: `${encoderName ?? "encoder"} forward pass` },
    { k: "Head only", v: L.head_ms_p50, p95: L.head_ms_p95, color: "var(--c3)", note: "projection + cosine" },
    { k: "CLM, cached state", v: L.warm_cached_ms_p50, p95: null, color: "var(--c3)", note: "embedding already cached" },
  ];
  const lo = 0.01;
  const hi = 10000;
  const lx = (v: number) => (Math.log10(Math.max(lo, v)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  const ticks = [0.01, 0.1, 1, 10, 100, 1000, 10000];
  const speedup = llm / full;
  return (
    <Panel expandable title="Latency per decision" info={<>How long one classification takes on this machine, shown on a log scale where each gridline is 10× the previous one. p50 is the typical (median) time and p95 is a slow case that 1 in 20 calls exceeds (thin tick mark). <b>Uncached</b> covers both running the text through the encoder and the trained head. <b>Cached</b> means the text's embedding was already computed. The big number compares this with an LLM call.</>} sub="p50 on this machine, log scale. The encoder dominates; the trained head itself is sub-millisecond.">
      <div className="grid grid-cols-1 lg:grid-cols-[200px_1fr] gap-5 items-center">
        <div>
          <div className="font-display film-text text-[44px] font-light tnum leading-none tracking-[-0.04em]">{speedup >= 10 ? Math.round(speedup) : speedup.toFixed(1)}×</div>
          <div className="text-[13px] text-ink-2 mt-1.5 inline-flex items-center gap-1">
            faster than {llmAssumed ? "a typical" : "the teacher"} LLM call
            <InfoTip>
              The LLM call's median time divided by the uncached CLM time (encoder + head). {llmAssumed ? "No teacher timing was measured, so a typical 1 s hosted LLM call is assumed." : "The LLM time is the teacher model's measured median for a single call."}
            </InfoTip>
          </div>
          <div className="text-[12px] text-muted mt-1 tnum">
            {ms(llm)} → {ms(full)}
          </div>
        </div>
        <div>
          <div className="space-y-2">
            {rows.map((x) => (
              <div key={x.k} className="grid grid-cols-[124px_1fr] gap-3 items-center" title={x.note}>
                <div className="min-w-0">
                  <div className="text-[12.5px] font-medium truncate">{x.k}</div>
                  <div className="text-[10.5px] text-muted truncate">{x.note}</div>
                </div>
                <div className="relative h-[16px]">
                  <div className="absolute inset-y-[3px] left-0 rounded-r-[3px]" style={{ width: `${lx(x.v) * 86}%`, background: x.color }} />
                  {x.p95 != null && <div className="absolute top-[1px] bottom-[1px] w-[2px] bg-ink-2/60 rounded" style={{ left: `${lx(x.p95) * 86}%` }} title={`p95 ${ms(x.p95)}`} />}
                  <span className="absolute top-[0px] text-[12px] tnum font-semibold" style={{ left: `calc(${lx(Math.max(x.v, x.p95 ?? 0)) * 86}% + 6px)` }}>
                    {ms(x.v)}
                    {x.p95 != null && <span className="text-muted font-normal"> p95 {ms(x.p95)}</span>}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-[124px_1fr] gap-3 mt-1">
            <span />
            <div className="relative h-4 border-t border-axis" style={{ borderColor: "var(--axis)" }}>
              {ticks.map((t) => (
                <span key={t} className="absolute top-0.5 -translate-x-1/2 text-[10px] text-muted tnum" style={{ left: `${lx(t) * 86}%` }}>
                  {t >= 1000 ? `${t / 1000}s` : `${t}ms`}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

export function EvaluationTab() {
  const { pid, store, detail, colorOf, labelText, labels } = useWs();
  const q = useQuery({ queryKey: qk.eval(pid), queryFn: () => api.eval(pid), retry: false, enabled: !!store.report || store.state === "reviewing" || store.state === "packaging" || store.state === "delivered" });
  const r = store.report ?? q.data ?? null;
  const [metric, setMetric] = useState<"accuracy" | "macro_f1">("accuracy");
  const [cell, setCell] = useState<[string, string] | null>(null);
  const [norm, setNorm] = useState<"row" | "count">("row");

  const confusion = useMemo(() => {
    if (!r) return null;
    const { labels: ls, matrix } = r.confusion;
    const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0) || 1);
    const max = Math.max(1, ...matrix.flat());
    return { ls, matrix, values: matrix.map((row, i) => row.map((v) => (norm === "row" ? v / rowSums[i] : v / max))) };
  }, [r, norm]);

  const errors = useMemo(() => (r ? r.errors.filter((e) => !cell || (e.gold === cell[0] && e.pred === cell[1])) : []), [r, cell]);

  if (!r) {
    return (
      <Empty icon={store.state === "evaluating" ? <Spinner size={22} /> : undefined} title={store.state === "evaluating" ? "Evaluating on the held-out test split" : "No evaluation yet"}>
        The fine-tuned head is compared with the zero-shot CLM head, a majority-class baseline and the teacher LLM. The gate needs all three conditions to pass.
      </Empty>
    );
  }

  const order = labels.length ? labels : r.confusion.labels;

  return (
    <div className="p-4 space-y-4">
      <GateBanner r={r} />
      <div className="grid gap-4 grid-cols-1 xl:grid-cols-[1.25fr_1fr]">
        <Panel
          expandable title="Against the baselines"
          info={<>Compares the trained head with three reference points on the same test data. <b>Zero-shot CLM</b> is the same model before any training, <b>Majority class</b> always guesses the most common label, and <b>Teacher LLM</b> is the large model that labelled the training data, scored on a smaller sample of at most 50 rows. The dashed line is the pass bar. The fine-tuned bar should clear the line and the bars of both the zero-shot and majority baselines.</>}
          sub="The fine-tuned head must clear the bar and beat both the zero-shot head and always-guess-the-majority."
          aside={
            <Segmented
              value={metric}
              onChange={setMetric}
              options={[
                { value: "accuracy", label: "Accuracy" },
                { value: "macro_f1", label: "Macro F1" },
              ]}
            />
          }
        >
          <p className="text-[11.5px] text-muted -mt-1 mb-2 flex items-center gap-1">
            Switch metric
            <InfoTip>
              <b>Accuracy</b> is the share of examples labelled correctly. <b>Macro F1</b> averages a per-label score that balances precision and recall, so a model that ignores a rare label is penalised even when its accuracy looks high. When labels are imbalanced, trust macro F1 more.
            </InfoTip>
          </p>
          <div className="pt-2 pb-5">
            <ModelBars models={r.models} metric={metric} threshold={r.threshold} />
          </div>
        </Panel>
        <Panel expandable title="Per-label scores" sub="Fine-tuned head on the test split." info="How well the trained head does on each label separately. A low row tells you which category it struggles with, even when the overall score looks fine.">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-muted text-[11.5px] text-left">
                <th className="font-medium py-1">Label</th>
                <th className="font-medium py-1 text-right"><span className="inline-flex items-center gap-1">Precision <InfoTip>Of the examples the model gave this label, the share that truly had it. Low precision means it uses this label too often (false alarms).</InfoTip></span></th>
                <th className="font-medium py-1 text-right"><span className="inline-flex items-center gap-1">Recall <InfoTip>Of the examples that truly have this label, the share the model caught. Low recall means it misses this label and calls those examples something else.</InfoTip></span></th>
                <th className="font-medium py-1 pl-3 w-[38%]"><span className="inline-flex items-center gap-1">F1 <InfoTip>A single 0–1 score that combines precision and recall and is only high when both are high. Amber bars fall below the pass bar.</InfoTip></span></th>
                <th className="font-medium py-1 text-right"><span className="inline-flex items-center gap-1">n <InfoTip>How many test examples have this label. Scores on labels with few examples are noisy.</InfoTip></span></th>
              </tr>
            </thead>
            <tbody>
              {r.per_label.map((l) => (
                <tr key={l.label} className="border-t border-line-soft">
                  <td className="py-1.5">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-[2px]" style={{ background: colorOf(l.label) }} />
                      {labelText(l.label)}
                    </span>
                  </td>
                  <td className="py-1.5 text-right tnum">{l.precision.toFixed(2)}</td>
                  <td className="py-1.5 text-right tnum">{l.recall.toFixed(2)}</td>
                  <td className="py-1.5 pl-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-[8px] bg-sunken rounded-[2px] relative">
                        <div className="absolute inset-y-0 left-0 rounded-[2px]" style={{ width: `${l.f1 * 100}%`, background: l.f1 < r.threshold ? "var(--warn-mark)" : "var(--accent)" }} />
                      </div>
                      <span className="tnum w-9 text-right font-medium">{l.f1.toFixed(2)}</span>
                    </div>
                  </td>
                  <td className="py-1.5 text-right tnum text-muted">{l.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-[11px] text-muted mt-2">Amber F1 bars fall below the pass bar.</p>
        </Panel>
      </div>

      <LatencyPanel r={r} encoderName={detail?.encoder?.name ?? null} />

      <div className="grid gap-4 grid-cols-1 xl:grid-cols-2">
        <Panel
          expandable title="Confusion matrix"
          info={<>Each row is the true label and each column is what the model predicted. The diagonal (top-left to bottom-right) holds the correct answers, and every other cell is a specific kind of mistake. A strong model has a bright diagonal and dim everything else. <b>Row %</b> shows each row as shares of that label's examples, and <b>Counts</b> shows raw numbers.</>}
          sub="Rows are gold labels, columns predictions. Click a cell to see those errors below."
          aside={
            <Segmented
              value={norm}
              onChange={setNorm}
              options={[
                { value: "row", label: "Row %" },
                { value: "count", label: "Counts" },
              ]}
            />
          }
        >
          {confusion && (
            <div className="max-w-[440px]">
            <Heatmap
              values={confusion.values}
              rowLabels={confusion.ls}
              colLabels={confusion.ls}
              domain={[0, 1]}
              colorOf={colorOf}
              labelText={labelText}
              rowTitle="gold"
              colTitle="predicted"
              cellText={(i, j, v) => (norm === "row" ? (confusion.matrix[i][j] ? `${Math.round(v * 100)}%` : "·") : String(confusion.matrix[i][j]))}
              format={(v) => (norm === "row" ? `${Math.round(v * 100)}%` : String(Math.round(v * Math.max(1, ...confusion.matrix.flat()))))}
              cellTitle={(g, p) => {
                const i = confusion.ls.indexOf(g);
                const j = confusion.ls.indexOf(p);
                return `gold ${labelText(g)}, predicted ${labelText(p)}: ${confusion.matrix[i][j]} of ${confusion.matrix[i].reduce((a, b) => a + b, 0)}`;
              }}
              onCellClick={(g, p) => setCell((c) => (c && c[0] === g && c[1] === p) || g === p ? null : [g, p])}
              selected={cell}
            />
            </div>
          )}
        </Panel>
        <Panel expandable title="Calibration" info={<>Checks whether the model's confidence can be trusted. Test predictions are grouped by how confident the model was, and each group's actual accuracy is plotted. Points on the diagonal mean “80% sure” really is right about 80% of the time. <b>ECE</b> (expected calibration error) is the average gap: under 0.05 is good, 0.05–0.10 is fair, and above 0.10 means the confidence numbers are misleading.</>} sub="When the head says 80%, is it right 80% of the time? This is what makes a confidence-threshold fallback safe." aside={<Chip tone={r.ece < 0.05 ? "good" : r.ece < 0.1 ? "warn" : "bad"}>ECE {r.ece.toFixed(3)}</Chip>}>
          <div className="max-w-[440px]"><Reliability bins={r.reliability} ece={r.ece} /></div>
        </Panel>
      </div>

      <Panel
        expandable title="Error browser"
        info="Real test examples the model got wrong, starting with the ones where it gave the true label the least probability (up to 25). The bars show how it split its probability across labels. Use this to spot mislabelled data, ambiguous categories or a pattern the model hasn't learned. Click a cell in the confusion matrix to filter."
        sub={cell ? `Gold ${labelText(cell[0])}, predicted ${labelText(cell[1])}` : `The ${r.errors.length} least-confident-wrong test examples (up to 25).`}
        aside={
          cell ? (
            <button onClick={() => setCell(null)} className="inline-flex items-center gap-1 text-[12px] text-accent cursor-pointer">
              <X size={12} /> Clear filter
            </button>
          ) : null
        }
      >
        {errors.length === 0 ? (
          <p className="text-[13px] text-muted py-4">{r.errors.length ? "No errors in this cell." : "No test errors. Suspiciously perfect: check the test split isn't leaking."}</p>
        ) : (
          <ul className="grid gap-3 grid-cols-1 2xl:grid-cols-2">
            {errors.map((e) => (
              <li key={e.id} className="border border-line-soft rounded-[8px] p-3 grid grid-cols-1 md:grid-cols-[1fr_240px] gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 text-[12px] mb-1.5 flex-wrap">
                    <span className="font-mono text-[11px] text-muted">{e.id}</span>
                    <Chip tone="good">gold {labelText(e.gold)}</Chip>
                    <Chip tone="bad">predicted {labelText(e.pred)}</Chip>
                  </div>
                  <p className="text-[12.5px] whitespace-pre-wrap break-words line-clamp-6 text-ink">{e.state}</p>
                </div>
                <ProbBars probs={e.probabilities} order={order} colorOf={colorOf} labelText={labelText} gold={e.gold} compact labelWidth={92} />
              </li>
            ))}
          </ul>
        )}
      </Panel>
      <p className="text-[11.5px] text-muted">
        Test split n = {r.n_test}. Teacher scored on a smaller sample to bound cost. Accuracy {pct(r.models.find((m) => m.name === "finetuned")?.accuracy ?? 0)} fine-tuned vs {pct(r.models.find((m) => m.name === "reference")?.accuracy ?? 0)} zero-shot.
      </p>
    </div>
  );
}
