import { useEffect, useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, CircleSlash, FileCode2, Minus, PencilLine, Plus, ThumbsUp } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { Opportunity, TaskSchema } from "../api/types";
import { Button, Chip, Dialog, Empty, Field, Input, Panel, Spinner, Textarea } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Markdown } from "../components/Markdown";
import { DataPromptPanel } from "../components/DataPromptPanel";
import { cn } from "../lib/format";
import { ProposalEncoder } from "./ProposalEncoder";
import { TaskSchemaEditor, validateTask } from "./TaskSchemaEditor";
import { useWs } from "./context";

function FitMeter({ v }: { v: number }) {
  const tone = v >= 0.8 ? "var(--good-mark)" : v >= 0.6 ? "var(--accent)" : "var(--warn-mark)";
  return (
    <div className="w-[120px]" title={`fit score ${v.toFixed(2)}`}>
      <div className="flex items-baseline justify-between text-[11.5px] text-muted mb-1">
        <span>fit</span>
        <span className="text-ink text-[15px] font-semibold tnum">{v.toFixed(2)}</span>
      </div>
      <div className="flex gap-[2px]">
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className="h-[6px] flex-1 rounded-[1px]" style={{ background: i < Math.round(v * 10) ? tone : "var(--line)" }} />
        ))}
      </div>
    </div>
  );
}

function OpportunityCard({
  o,
  rank,
  recommended,
  selected,
  accepted,
  onSelect,
  selectable,
}: {
  o: Opportunity;
  rank: number;
  recommended: boolean;
  selected: boolean;
  accepted: boolean;
  onSelect: () => void;
  selectable: boolean;
}) {
  const [open, setOpen] = useState(rank === 1);
  const labels = o.task.type === "choice" && o.task.criteria && !Array.isArray(o.task.criteria) ? Object.keys(o.task.criteria) : o.task.type === "score" && Array.isArray(o.task.criteria) ? o.task.criteria.map((c) => c.split(/\s+[—-]\s+/)[0]) : ["true", "false"];
  return (
    <article
      className={cn(
        "bg-panel border rounded-[10px] transition-colors min-w-0",
        selected ? "border-accent shadow-[0_0_0_1px_var(--accent)]" : accepted ? "border-good" : "border-line",
      )}
    >
      <header className="flex items-start gap-3 p-4 pb-3">
        <span className="text-[12px] text-muted tnum pt-0.5 w-4 shrink-0">#{rank}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-[15px] font-semibold leading-snug">{o.title}</h3>
            {recommended && <Chip tone="option">Recommended</Chip>}
            {accepted && (
              <Chip tone="good">
                <Check size={11} /> Accepted
              </Chip>
            )}
          </div>
          <div className="flex items-center gap-1.5 flex-wrap mt-1.5 text-[12px] text-ink-2">
            <span className="font-mono text-[11.5px] text-muted">{o.task.type}</span>
            <span className="text-muted">over</span>
            {labels.map((l) => (
              <span key={l} className="px-1.5 h-5 inline-flex items-center rounded-[4px] bg-sunken border border-line-soft text-[11.5px]">
                {l}
              </span>
            ))}
          </div>
        </div>
        <FitMeter v={o.fit_score} />
      </header>
      <div className="px-4 pl-11 pb-3 grid grid-cols-[minmax(0,1fr)] gap-3">
        <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[13px]">
          <span className="text-muted">Today</span>
          <Markdown className="text-[13px]">{o.current_approach}</Markdown>
          <span className="text-muted">Why it fits</span>
          <span className="text-ink-2">{o.rationale}</span>
          <span className="text-muted">Success bar</span>
          <span>
            test <span className="font-medium">{o.success_metric === "macro_f1" ? "macro F1" : "accuracy"}</span> ≥ <span className="font-semibold tnum">{o.success_threshold.toFixed(2)}</span>, beating the zero-shot head and the majority baseline · {o.target_examples} examples
          </span>
        </div>
        <button onClick={() => setOpen((x) => !x)} className="justify-self-start inline-flex items-center gap-1 text-[12.5px] font-medium text-accent cursor-pointer" aria-expanded={open}>
          <ChevronDown size={14} className={cn("transition-transform", !open && "-rotate-90")} />
          {open ? "Hide details" : `Files, benefits and risks (${o.files.length} files)`}
        </button>
        {open && (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-4">
            <div className="space-y-2.5 min-w-0">
              {o.files.map((f) => (
                <div key={f.path + (f.start_line ?? "")}>
                  <div className="flex items-center gap-1.5 text-[12px] mb-1">
                    <FileCode2 size={13} className="text-muted" />
                    <span className="font-mono text-ink">{f.path}</span>
                    {f.start_line != null && (
                      <span className="font-mono text-muted">
                        :{f.start_line}
                        {f.end_line != null && f.end_line !== f.start_line ? `–${f.end_line}` : ""}
                      </span>
                    )}
                  </div>
                  {f.snippet && <CodeBlock code={f.snippet} startLine={f.start_line ?? 1} />}
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="text-[12.5px] font-semibold mb-1.5">Benefits</h4>
                <ul className="space-y-1">
                  {o.expected_benefits.map((b) => (
                    <li key={b} className="flex gap-2 text-[13px]">
                      <Plus size={13} className="text-good mt-[3px] shrink-0" />
                      {b}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-[12.5px] font-semibold mb-1.5">Risks</h4>
                <ul className="space-y-1">
                  {o.risks.map((b) => (
                    <li key={b} className="flex gap-2 text-[13px]">
                      <Minus size={13} className="text-bad mt-[3px] shrink-0" />
                      {b}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div>
              <h4 className="text-[12.5px] font-semibold mb-1">Where the training data comes from</h4>
              <Markdown className="text-[13px] text-ink-2">{o.data_strategy}</Markdown>
            </div>
            <div>
              <h4 className="text-[12.5px] font-semibold mb-1">Integration sketch</h4>
              <Markdown className="text-[13px] text-ink-2">{o.integration_sketch}</Markdown>
            </div>
          </div>
        )}
      </div>
      {selectable && (
        <footer className="px-4 pl-11 pb-4">
          <Button size="sm" variant={selected ? "primary" : "secondary"} onClick={onSelect} icon={selected ? <Check size={13} /> : undefined}>
            {selected ? "Selected for training" : "Select this one"}
          </Button>
        </footer>
      )}
    </article>
  );
}

export function ProposalTab() {
  const { pid, store, detail, goTab } = useWs();
  const proposal = store.proposal ?? detail?.proposal ?? null;
  const proposed = store.state === "proposed" && (detail?.allowed_human_events ?? []).includes("accept");
  const qc = useQueryClient();
  const [sel, setSel] = useState<string | null>(null);
  const [draft, setDraft] = useState<TaskSchema | null>(null);
  const [target, setTarget] = useState(600);
  const [threshold, setThreshold] = useState(0.85);
  const [revOpen, setRevOpen] = useState(false);
  const [feedback, setFeedback] = useState("");

  const selected = useMemo(() => proposal?.opportunities.find((o) => o.id === sel) ?? null, [proposal, sel]);

  // reset selection when a new revision arrives
  useEffect(() => {
    if (!proposal) return;
    const id = detail?.accepted_opportunity_id ?? proposal.recommended_id ?? proposal.opportunities[0]?.id ?? null;
    setSel(id);
  }, [proposal, detail?.accepted_opportunity_id]);
  useEffect(() => {
    if (!selected) return;
    setDraft(structuredClone(selected.task));
    setTarget(selected.target_examples);
    setThreshold(selected.success_threshold);
  }, [selected]);

  // real rows without a label are labelled by the labelling agent (the default policy); the Labels tab can
  // switch a project to a human review policy later
  const accept = useMutation({
    mutationFn: () => {
      const changedTask = draft && selected && JSON.stringify(draft) !== JSON.stringify(selected.task);
      const overrides: { task?: TaskSchema; target_examples?: number; success_threshold?: number } = {};
      if (changedTask && draft) overrides.task = draft;
      if (selected && target !== selected.target_examples) overrides.target_examples = target;
      if (selected && threshold !== selected.success_threshold) overrides.success_threshold = threshold;
      return api.accept(pid, { opportunity_id: sel!, overrides: Object.keys(overrides).length ? overrides : undefined });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.project(pid) });
      goTab("pipeline");
    },
  });
  const startAccept = () => accept.mutate();
  const revise = useMutation({
    mutationFn: () => api.revise(pid, feedback.trim()),
    onSuccess: () => {
      setRevOpen(false);
      setFeedback("");
      qc.invalidateQueries({ queryKey: qk.project(pid) });
    },
  });

  if (!proposal) {
    return (
      <Empty icon={store.state === "analyzing" ? <Spinner size={22} /> : undefined} title={store.state === "analyzing" ? "The agent is analysing the code" : "No proposal yet"}>
        {store.state === "analyzing" ? "Tool steps stream into the chat on the left. The proposal lands here when the agent is done." : "Analysis hasn't produced a proposal."}
      </Empty>
    );
  }

  if (proposal.verdict === "no_fit") {
    return (
      <div className="p-4 max-w-[860px]">
        <Panel>
          <div className="flex gap-4 items-start pt-2">
            <div className="w-10 h-10 rounded-full grid place-items-center bg-sunken text-ink-2 shrink-0">
              <CircleSlash size={20} />
            </div>
            <div className="min-w-0">
              <h2 className="text-[17px] font-semibold">No good fit for a CLM here</h2>
              <p className="text-[13.5px] text-ink-2 mt-1.5 max-w-[70ch]">{proposal.summary}</p>
              <h3 className="text-[13px] font-semibold mt-4 mb-1.5">Why not</h3>
              <ol className="space-y-1.5 list-decimal pl-5 text-[13.5px]">
                {proposal.no_fit_reasons.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ol>
              <p className="text-[12.5px] text-muted mt-4">A CLM answers closed questions — pick one of N options, a yes/no, or a score on ordered levels — over text it is given. Describe a decision like that in a new project, or chat with the agent.</p>
            </div>
          </div>
        </Panel>
      </div>
    );
  }

  const errs = draft ? validateTask(draft) : [];
  const accepted = detail?.accepted_opportunity_id;

  return (
    <div className="p-4 grid gap-4 grid-cols-1 min-[1400px]:grid-cols-[minmax(0,1fr)_420px] items-start">
      <div className="space-y-3 min-w-0">
        <Panel
          title={
            <span className="inline-flex items-center gap-2">
              Proposal{proposal.revision > 0 && <Chip>revision {proposal.revision}</Chip>}
            </span>
          }
        >
          <Markdown>{proposal.summary}</Markdown>
        </Panel>
        {proposal.opportunities.map((o, i) => (
          <OpportunityCard
            key={o.id}
            o={o}
            rank={i + 1}
            recommended={o.id === proposal.recommended_id}
            selected={proposed && o.id === sel}
            accepted={o.id === accepted}
            selectable={proposed}
            onSelect={() => setSel(o.id)}
          />
        ))}
      </div>

      {draft && selected && (
        <div className="min-[1400px]:sticky min-[1400px]:top-4">
          <Panel
            title={proposed ? "Review the task, then accept" : "Task schema"}
            sub={proposed ? <>Training starts on accept. Edits here are sent as overrides for “{selected.title}”.</> : "Locked: the proposal has been decided."}
          >
            <TaskSchemaEditor value={draft} onChange={setDraft} readOnly={!proposed} />
            <div className="grid grid-cols-[1fr_1fr] gap-2.5 mt-3 pt-3 border-t border-line-soft">
              <Field label="Examples to curate" hint="120 to 3,000">
                <Input type="number" min={120} max={3000} step={20} value={target} disabled={!proposed} onChange={(e) => setTarget(Math.max(120, Math.min(3000, Number(e.target.value) || 0)))} />
              </Field>
              <Field label={`Pass threshold (${selected.success_metric === "macro_f1" ? "macro F1" : "accuracy"})`} hint="Absolute floor on the test split">
                <div className="flex items-center gap-2 h-8.5">
                  <input type="range" min={0.5} max={0.99} step={0.01} value={threshold} disabled={!proposed} onChange={(e) => setThreshold(Number(e.target.value))} className="flex-1 accent-[var(--accent)]" aria-label="Pass threshold" />
                  <span className="tnum text-[13px] font-semibold w-9 text-right">{threshold.toFixed(2)}</span>
                </div>
              </Field>
            </div>
            <ProposalEncoder pid={pid} detail={detail} opportunityId={selected.id} editable={proposed} />
            {proposed && (
              <>
                {errs.length > 0 && (
                  <ul className="mt-3 text-[12px] text-bad space-y-0.5">
                    {errs.map((e) => (
                      <li key={e}>{e}</li>
                    ))}
                  </ul>
                )}
                {accept.isError && <p className="mt-2 text-[12px] text-bad">{(accept.error as Error).message}</p>}
                <div className="flex gap-2 mt-4">
                  <Button variant="primary" icon={<ThumbsUp size={14} />} disabled={!!errs.length || accept.isPending} onClick={startAccept} className="flex-1">
                    {accept.isPending ? "Accepting…" : "Accept and start training"}
                  </Button>
                  <Button icon={<PencilLine size={14} />} onClick={() => setRevOpen(true)}>
                    Request changes
                  </Button>
                </div>
              </>
            )}
          </Panel>
          <div className="mt-4"><DataPromptPanel task={draft} opp={selected} repo={detail?.repo} accepted={!proposed} /></div>
        </div>
      )}

      <Dialog open={revOpen} onClose={() => setRevOpen(false)} title="Request changes">
        <p className="text-[13px] text-ink-2 mb-3">The agent re-analyses with your feedback and returns revision {proposal.revision + 1}.</p>
        <Textarea rows={4} autoFocus value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="e.g. Prioritise the urgency scorer; the router is being replaced next quarter." />
        {revise.isError && <p className="mt-2 text-[12px] text-bad">{(revise.error as Error).message}</p>}
        <div className="flex justify-end gap-2 mt-3">
          <Button variant="ghost" onClick={() => setRevOpen(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!feedback.trim() || revise.isPending} onClick={() => revise.mutate()}>
            Send feedback
          </Button>
        </div>
      </Dialog>
    </div>
  );
}

