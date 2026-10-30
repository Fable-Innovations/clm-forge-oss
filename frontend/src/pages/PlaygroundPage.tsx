import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { TaskSchema } from "../api/types";
import { Panel, Textarea } from "../components/ui";
import { InferCard } from "../components/InferView";
import { TaskSchemaEditor, validateTask } from "../workspace/TaskSchemaEditor";
import { useDebounced } from "../hooks/useDebounced";
import { labelOrder, makeColorOf } from "../lib/colors";

const DEFAULT_TASK: TaskSchema = {
  name: "sentiment",
  question_id: "sentiment",
  type: "choice",
  instructions: "What is the customer's overall sentiment?",
  criteria: { positive: "Happy, satisfied, thankful", neutral: "Factual, no clear feeling", negative: "Frustrated, angry or disappointed" },
  state_description: "A customer message.",
  state_example: "I've been waiting two weeks for a reply and nobody has helped. Really disappointed.",
};

export function PlaygroundPage() {
  const [task, setTask] = useState<TaskSchema>(DEFAULT_TASK);
  const [state, setState] = useState(DEFAULT_TASK.state_example);
  const dTask = useDebounced(task, 500);
  const dState = useDebounced(state.trim(), 450);
  const errs = validateTask(dTask);
  const q = useQuery({
    queryKey: ["playground", dTask, dState],
    queryFn: () => api.playgroundInfer(dState, dTask),
    enabled: !!dState && errs.length === 0,
    retry: false,
  });
  const labels = useMemo(() => labelOrder(dTask), [dTask]);
  const colorOf = useMemo(() => makeColorOf(labels), [labels]);
  const labelText = (l: string) => (dTask.type === "score" && Array.isArray(dTask.criteria) ? `${l} · ${String(dTask.criteria[Number(l)] ?? "").split(/\s+[—-]\s+/)[0]}` : l);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[1240px] mx-auto px-6 py-8">
        <h1 className="font-display text-[30px] font-normal tracking-[-0.03em] leading-none">Playground</h1>
        <p className="text-[13.5px] text-ink-2 mt-1 mb-6 max-w-[72ch]">
          Ask the zero-shot reference head any closed question, no experiment or training needed. Useful for checking whether a task is plausible before starting an experiment, and for wording option descriptions.
        </p>
        <div className="grid gap-4 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start">
          <Panel title="Question">
            <TaskSchemaEditor value={task} onChange={setTask} />
          </Panel>
          <div className="space-y-4 lg:sticky lg:top-4">
            <Panel title="State" sub={task.state_description}>
              <Textarea rows={6} value={state} onChange={(e) => setState(e.target.value)} className="font-mono text-[12.5px]" aria-label="State text" />
            </Panel>
            <InferCard
              title="Zero-shot reference head"
              sub="CLM-v0.1-8B, frozen encoder + reference projection heads"
              result={q.data}
              loading={q.isFetching}
              error={errs.length ? errs[0] : q.isError ? (q.error as Error).message : null}
              task={dTask}
              order={labels}
              colorOf={colorOf}
              labelText={labelText}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
