import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ListOrdered, Sparkles } from "lucide-react";
import { api } from "../api/client";
import { Button, Empty, Field, Input, Panel, Segmented, Textarea } from "../components/ui";
import { InferCard } from "../components/InferView";
import { useDebounced } from "../hooks/useDebounced";
import { useWs } from "./context";

export function PlaygroundTab() {
  const { pid, task, store, labels, colorOf, labelText } = useWs();
  const [mode, setMode] = useState<"answer" | "rank">("answer");
  const [state, setState] = useState("");
  useEffect(() => {
    if (task && !state) setState(task.state_example);
  }, [task]);
  const dState = useDebounced(state.trim(), 450);
  const trained = !!store.report || ["reviewing", "packaging", "delivered"].includes(store.state ?? "");

  const ft = useQuery({ queryKey: ["infer", pid, "finetuned", dState], queryFn: () => api.infer(pid, dState, "finetuned"), enabled: !!dState && trained && mode === "answer", retry: false, staleTime: Infinity });
  const ref = useQuery({ queryKey: ["infer", pid, "reference", dState], queryFn: () => api.infer(pid, dState, "reference"), enabled: !!dState && !!task && mode === "answer", retry: false, staleTime: Infinity });

  const [cands, setCands] = useState("");
  const [instr, setInstr] = useState("");
  useEffect(() => {
    if (task && !cands && task.type === "choice" && task.criteria && !Array.isArray(task.criteria)) setCands(Object.values(task.criteria).join("\n"));
    if (task && !instr) setInstr(task.instructions);
  }, [task]);
  const rank = useMutation({ mutationFn: () => api.rank(pid, state.trim(), cands.split("\n").map((c) => c.trim()).filter(Boolean), instr.trim() || undefined) });

  const samples = useMemo(() => {
    const xs: { label: string; text: string }[] = [];
    if (task) xs.push({ label: "Schema example", text: task.state_example });
    for (const e of store.report?.errors.slice(0, 3) ?? []) xs.push({ label: `Test error: ${labelText(e.gold)} → ${labelText(e.pred)}`, text: e.state });
    return xs;
  }, [task, store.report, labelText]);

  if (!task) return <Empty title="No task yet">The playground opens once there is a proposal with a task schema.</Empty>;

  return (
    <div className="p-4 space-y-4">
      <Panel
        title="Try the head on any state"
        sub={<>Question asked: “{task.instructions}”</>}
        aside={
          <Segmented
            value={mode}
            onChange={setMode}
            options={[
              { value: "answer", label: <span className="inline-flex items-center gap-1"><Sparkles size={12} /> Answer</span> },
              { value: "rank", label: <span className="inline-flex items-center gap-1"><ListOrdered size={12} /> Rank</span> },
            ]}
          />
        }
      >
        <Textarea rows={6} value={state} onChange={(e) => setState(e.target.value)} className="font-mono text-[12.5px]" placeholder={task.state_description} aria-label="State text" />
        <div className="flex flex-wrap gap-1.5 mt-2">
          {samples.map((s) => (
            <button key={s.label + s.text.slice(0, 10)} onClick={() => setState(s.text)} className="text-[11.5px] px-2 h-6 rounded-[4px] border border-line bg-raised text-ink-2 hover:text-ink hover:border-ink-2/40 cursor-pointer truncate max-w-[280px]">
              {s.label}
            </button>
          ))}
        </div>
      </Panel>

      {mode === "answer" ? (
        <div className="grid gap-4 grid-cols-1 lg:grid-cols-2">
          <InferCard
            title="Fine-tuned head"
            sub="This experiment's trained head"
            result={ft.data}
            loading={ft.isFetching}
            error={!trained ? "Available once training and evaluation finish." : ft.isError ? (ft.error as Error).message : null}
            task={task}
            order={labels}
            colorOf={colorOf}
            labelText={labelText}
          />
          <InferCard
            title="Zero-shot reference"
            sub="CLM-v0.1 head, no training on this task"
            result={ref.data}
            loading={ref.isFetching}
            error={ref.isError ? (ref.error as Error).message : null}
            task={task}
            order={labels}
            colorOf={colorOf}
            labelText={labelText}
          />
        </div>
      ) : (
        <Panel title="Rank free-form candidates" sub="Score any list of candidate texts against the state, one per line. Uses the fine-tuned head's projections.">
          <div className="grid gap-3 grid-cols-1 lg:grid-cols-[1fr_1fr]">
            <div className="space-y-3">
              <Field label="Instructions (optional)">
                <Input value={instr} onChange={(e) => setInstr(e.target.value)} />
              </Field>
              <Field label="Candidates">
                <Textarea rows={7} wrap="off" value={cands} onChange={(e) => setCands(e.target.value)} className="text-[12.5px]" />
              </Field>
              <Button variant="primary" onClick={() => rank.mutate()} disabled={!state.trim() || !cands.trim() || rank.isPending}>
                {rank.isPending ? "Ranking…" : "Rank candidates"}
              </Button>
              {rank.isError && <p className="text-[12px] text-bad">{(rank.error as Error).message}</p>}
            </div>
            <div>
              {rank.data ? (
                <ol className="space-y-2">
                  {rank.data.map((r) => (
                    <li key={r.rank} className="grid grid-cols-[22px_1fr_52px] gap-2 items-center">
                      <span className="text-[12px] text-muted tnum text-right">{r.rank}</span>
                      <div className="min-w-0">
                        <div className="text-[12.5px] truncate" title={r.candidate}>{r.candidate}</div>
                        <div className="h-[6px] rounded-[2px] bg-sunken mt-1 relative">
                          <div className="absolute inset-y-0 left-0 rounded-[2px] bg-accent" style={{ width: `${r.prob * 100}%`, opacity: r.rank === 1 ? 1 : 0.55 }} />
                        </div>
                      </div>
                      <span className="text-[12px] tnum text-right font-medium">{(r.prob * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-[12.5px] text-muted pt-6">Ranked candidates with probabilities appear here.</p>
              )}
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
