import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronLeft, ChevronRight, Lock, Search, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { CurationPhase, Example, Split } from "../api/types";
import { Chip, Empty, Input, Panel, Select, Spinner } from "../components/ui";
import { HBars, StackBar } from "../viz/HBars";
import { cn, int } from "../lib/format";
import { LabelSourceChip } from "../components/LabelQueue";
import { DataPromptPanel } from "../components/DataPromptPanel";
import { useWs } from "./context";

const PHASES: { id: CurationPhase; label: string; hint: string }[] = [
  { id: "importing", label: "Import", hint: "real rows from exported tables in clm_data/" },
  { id: "mining", label: "Mine", hint: "real states from fixtures, prompts, sample data" },
  { id: "generating", label: "Generate", hint: "parallel workers, one per label: synthetic states + hard negatives" },
  { id: "reviewing", label: "Review", hint: "orchestrator checks a sample of each worker's output; weak labels are regenerated with its feedback" },
  { id: "labelling", label: "Label", hint: "independent teacher relabel; disagreements dropped" },
  { id: "dedup", label: "Dedup", hint: "exact + embedding cosine > 0.95" },
  { id: "splitting", label: "Split", hint: "70/15/15 stratified, real rows preferred for test" },
];

const SOURCE_COLORS: Record<string, string> = { imported: "var(--c4)", mined: "var(--c1)", synthetic: "var(--c3)", hard_negative: "var(--c2)", user: "var(--c7)" };
const SPLIT_COLORS: Record<string, string> = { train: "var(--s5)", val: "var(--s3)", test: "var(--option)" };
const PAGE = 25;

function CurationProgress() {
  const { store, labelText } = useWs();
  const c = store.curation;
  const active = store.state === "curating";
  const curIdx = c ? (c.phase === "done" ? PHASES.length : PHASES.findIndex((p) => p.id === c.phase)) : -1;
  return (
    <Panel
      title="Curation"
      sub={c ? (c.phase === "done" ? `Done: ${int(c.total)} examples written to dataset.jsonl` : `Phase ${curIdx + 1} of ${PHASES.length}`) : active ? "Starting…" : "Not started"}
      aside={store.embed && store.embed.done < store.embed.total ? <Chip tone="accent">embedding {int(store.embed.done)}/{int(store.embed.total)}</Chip> : null}
    >
      <ol className="grid grid-cols-4 lg:grid-cols-7 gap-2">
        {PHASES.map((p, i) => {
          const ph = c?.phases[p.id];
          const done = i < curIdx;
          const cur = i === curIdx;
          const frac = done ? 1 : ph && ph.total ? ph.done / ph.total : 0;
          return (
            <li key={p.id} title={p.hint} className="min-w-0">
              <div className="flex items-center gap-1.5 text-[12.5px] font-medium">
                <span
                  className={cn("w-4 h-4 rounded-full grid place-items-center text-[10px] shrink-0", cur && active && "pulse-dot")}
                  style={{ background: done ? "var(--accent)" : cur ? "var(--accent-soft)" : "var(--sunken)", color: done ? "var(--accent-ink)" : "var(--ink-2)", border: cur ? "1px solid var(--accent)" : "1px solid var(--line)" }}
                >
                  {done ? <Check size={10} /> : i + 1}
                </span>
                <span className={cn(!done && !cur && "text-muted")}>{p.label}</span>
              </div>
              <div className="h-[4px] rounded-full mt-2" style={{ background: "var(--line)" }}>
                <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${frac * 100}%`, background: "var(--accent)" }} />
              </div>
              <div className="text-[11px] text-muted mt-1 tnum truncate">{ph ? `${int(ph.done)} / ${int(ph.total)}` : done ? "done" : "—"}</div>
            </li>
          );
        })}
      </ol>
      {c?.phase === "generating" && c.workers && Object.keys(c.workers).length > 0 && (
        <div className="mt-3 pt-3 border-t border-line-soft">
          <div className="text-[11.5px] text-muted mb-1.5">{Object.keys(c.workers).length} generator workers in parallel</div>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(c.workers).map(([lab, w]) => (
              <Chip key={lab} tone={w.done >= w.total ? "good" : "accent"} title={`${w.done} of ${w.total} batches, ${w.generated} texts`}>
                {labelText(lab)} <span className="tnum">{w.done}/{w.total}</span>
              </Chip>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}

function ExampleRow({ ex, locked, onPatch, onDelete, labels, colorOf, labelText }: { ex: Example; locked: boolean; onPatch: (b: Partial<Example>) => void; onDelete: () => void; labels: string[]; colorOf: (l: string) => string; labelText: (l: string) => string }) {
  const [open, setOpen] = useState(false);
  return (
    <tr className="border-b border-line-soft align-top hover:bg-sunken/50">
      <td className="py-2 pl-4 pr-2 font-mono text-[11px] text-muted whitespace-nowrap">{ex.id}</td>
      <td className="py-2 px-2 text-[12.5px] min-w-0">
        <button onClick={() => setOpen((o) => !o)} className={cn("text-left whitespace-pre-wrap break-words cursor-pointer w-full", !open && "line-clamp-2")} title={open ? "Collapse" : "Expand"}>
          {ex.state}
        </button>
        {open && ex.rationale && <div className="text-[11.5px] text-muted mt-1">Labeller: {ex.rationale}</div>}
      </td>
      <td className="py-2 px-2">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-[2px] shrink-0" style={{ background: colorOf(ex.label) }} />
          <select
            aria-label="Label"
            disabled={locked}
            value={ex.label}
            onChange={(e) => onPatch({ label: e.target.value })}
            className="bg-transparent text-[12.5px] border border-transparent hover:border-line rounded px-1 py-0.5 cursor-pointer disabled:cursor-default max-w-[130px]"
          >
            {(labels.includes(ex.label) ? labels : [ex.label, ...labels]).map((l) => (
              <option key={l} value={l}>
                {labelText(l)}
              </option>
            ))}
          </select>
        </div>
      </td>
      <td className="py-2 px-2 text-[12px] text-ink-2 whitespace-nowrap">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ background: SOURCE_COLORS[ex.source] }} />
          {ex.source.replace("_", " ")}
          <LabelSourceChip source={ex.label_source} className="ml-1" />
        </span>
      </td>
      <td className="py-2 px-2">
        <select
          aria-label="Split"
          disabled={locked}
          value={ex.split}
          onChange={(e) => onPatch({ split: e.target.value as Split })}
          className="bg-transparent text-[12.5px] border border-transparent hover:border-line rounded px-1 py-0.5 cursor-pointer disabled:cursor-default"
        >
          <option>train</option>
          <option>val</option>
          <option>test</option>
        </select>
      </td>
      <td className="py-2 pr-3 pl-1 text-right">
        <button aria-label="Delete example" disabled={locked} onClick={onDelete} className="p-1 rounded text-muted hover:text-bad hover:bg-bad-soft disabled:opacity-30 cursor-pointer">
          <Trash2 size={13} />
        </button>
      </td>
    </tr>
  );
}

export function DataTab() {
  const { pid, store, detail, task, labels, colorOf, labelText } = useWs();
  const acceptedOpp = detail?.proposal?.opportunities.find((o) => o.id === detail.accepted_opportunity_id) ?? null;
  const [q, setQ] = useState("");
  const [dq, setDq] = useState("");
  const [split, setSplit] = useState("");
  const [label, setLabel] = useState("");
  const [source, setSource] = useState("");
  const [page, setPage] = useState(0);
  const qc = useQueryClient();
  useEffect(() => {
    const t = setTimeout(() => setDq(q), 250);
    return () => clearTimeout(t);
  }, [q]);
  useEffect(() => setPage(0), [dq, split, label, source]);

  const locked = store.state === "curating" || store.state === "selecting" || store.state === "training";
  const hasData = !!store.curation && store.curation.phase === "done";
  const ds = useQuery({
    queryKey: [...qk.dataset(pid), { dq, split, label, source, page }],
    queryFn: () => api.dataset(pid, { q: dq, split, label, source, offset: page * PAGE, limit: PAGE }),
    placeholderData: keepPreviousData,
    retry: false,
  });
  const patch = useMutation({
    mutationFn: ({ id, b }: { id: string; b: Partial<Example> }) => api.patchExample(pid, id, b as Partial<Pick<Example, "label" | "split" | "state">>),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.dataset(pid) }),
  });
  const del = useMutation({
    mutationFn: (id: string) => api.deleteExample(pid, id),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.dataset(pid) }),
  });

  const counts = ds.data?.counts && ds.data.total >= 0 && Object.keys(ds.data.counts.by_label).length ? ds.data.counts : store.curation?.counts;
  const labelData = useMemo(() => {
    const ks = labels.length ? labels : Object.keys(counts?.by_label ?? {});
    return ks.map((k) => ({ key: k, label: labelText(k), value: counts?.by_label[k] ?? 0 }));
  }, [labels, counts, labelText]);
  const srcData = ["imported", "mined", "synthetic", "hard_negative", "user"].map((k) => ({ key: k, label: k.replace("_", " "), value: counts?.by_source[k] ?? 0 })).filter((d) => d.value > 0 || (d.key !== "user" && d.key !== "imported"));
  const splitData = ["train", "val", "test"].map((k) => ({ key: k, value: counts?.by_split[k] ?? 0 }));

  if (!store.curation && store.state && ["created", "analyzing", "proposed", "accepted", "no_fit"].includes(store.state)) {
    return (
      <div className="p-4 space-y-4">
        <Empty title="No data yet">Curation starts after you accept a proposal. Imported, mined, synthetic and hard-negative examples will show up here as they're produced.</Empty>
        {task && <div className="max-w-[760px] mx-auto"><DataPromptPanel task={task} opp={acceptedOpp} repo={detail?.repo} accepted={store.state === "accepted"} /></div>}
      </div>
    );
  }

  const total = ds.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));

  return (
    <div className="p-4 space-y-4">
      <CurationProgress />
      <div className="grid gap-4 grid-cols-1 xl:grid-cols-[1.2fr_1fr_1fr]">
        <Panel title="By label" sub="Balanced classes keep the head from learning the prior.">
          {counts ? <HBars data={labelData} colorOf={colorOf} /> : <p className="text-[12.5px] text-muted">Waiting for counts…</p>}
        </Panel>
        <Panel title="By source" sub="Hard negatives sit on confusable label boundaries.">
          {counts ? <HBars data={srcData} colorOf={(k) => SOURCE_COLORS[k]} labelWidth={104} /> : <p className="text-[12.5px] text-muted">Waiting…</p>}
        </Panel>
        <Panel title="By split" sub="Test is held out from training and model selection.">
          {counts && splitData.some((d) => d.value) ? <StackBar data={splitData} colorOf={(k) => SPLIT_COLORS[k]} /> : <p className="text-[12.5px] text-muted">Split happens at the end of curation.</p>}
        </Panel>
      </div>

      <Panel
        expandable title="Examples"
        sub={locked ? undefined : "Click a state to expand it. Label and split edits are saved immediately and mark the row as a user edit."}
        aside={
          locked ? (
            <Chip tone="neutral">
              <Lock size={11} /> read-only while {store.state}
            </Chip>
          ) : null
        }
        bodyClass="p-0"
      >
        <div className="flex flex-wrap items-center gap-2 px-4 pb-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted" />
            <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search state text or id" className="pl-8" />
          </div>
          <Select value={label} onChange={(e) => setLabel(e.target.value)} className="w-[140px]" aria-label="Filter by label">
            <option value="">All labels</option>
            {labels.map((l) => (
              <option key={l} value={l}>
                {labelText(l)}
              </option>
            ))}
          </Select>
          <Select value={source} onChange={(e) => setSource(e.target.value)} className="w-[140px]" aria-label="Filter by source">
            <option value="">All sources</option>
            <option value="imported">imported</option>
            <option value="mined">mined</option>
            <option value="synthetic">synthetic</option>
            <option value="hard_negative">hard negative</option>
            <option value="user">user</option>
          </Select>
          <Select value={split} onChange={(e) => setSplit(e.target.value)} className="w-[110px]" aria-label="Filter by split">
            <option value="">All splits</option>
            <option value="train">train</option>
            <option value="val">val</option>
            <option value="test">test</option>
          </Select>
        </div>
        <div className="overflow-x-auto border-t border-line">
          <table className="w-full table-fixed">
            <colgroup>
              <col className="w-[96px]" />
              <col />
              <col className="w-[150px]" />
              <col className="w-[124px]" />
              <col className="w-[80px]" />
              <col className="w-[40px]" />
            </colgroup>
            <thead>
              <tr className="text-left text-[12px] text-muted border-b border-line">
                <th className="font-medium py-2 pl-4 pr-2">id</th>
                <th className="font-medium py-2 px-2">State</th>
                <th className="font-medium py-2 px-2">Label</th>
                <th className="font-medium py-2 px-2">Source</th>
                <th className="font-medium py-2 px-2">Split</th>
                <th />
              </tr>
            </thead>
            <tbody className={cn(ds.isFetching && "opacity-70")}>
              {ds.isLoading && (
                <tr>
                  <td colSpan={6} className="p-8 text-center text-muted text-[13px]">
                    <span className="inline-flex items-center gap-2"><Spinner /> Loading…</span>
                  </td>
                </tr>
              )}
              {ds.isError && (
                <tr>
                  <td colSpan={6} className="p-8 text-center text-muted text-[13px]">
                    {hasData ? `Couldn't load the dataset: ${(ds.error as Error).message}` : "The dataset is written when curation finishes."}
                  </td>
                </tr>
              )}
              {ds.data?.items.length === 0 && (
                <tr>
                  <td colSpan={6} className="p-8 text-center text-muted text-[13px]">
                    {total === 0 && !dq && !label && !source && !split ? "The dataset is written when curation finishes." : "No examples match these filters."}
                  </td>
                </tr>
              )}
              {ds.data?.items.map((ex) => (
                <ExampleRow
                  key={ex.id}
                  ex={ex}
                  locked={locked}
                  labels={labels}
                  colorOf={colorOf}
                  labelText={labelText}
                  onPatch={(b) => patch.mutate({ id: ex.id, b })}
                  onDelete={() => del.mutate(ex.id)}
                />
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between px-4 py-2.5 text-[12.5px] text-ink-2">
          <span className="tnum">
            {total ? `${int(page * PAGE + 1)}–${int(Math.min(total, (page + 1) * PAGE))} of ${int(total)}` : "0 examples"}
            {(patch.isError || del.isError) && <span className="text-bad ml-3">{((patch.error ?? del.error) as Error).message}</span>}
          </span>
          <span className="flex items-center gap-1">
            <button aria-label="Previous page" disabled={page === 0} onClick={() => setPage((p) => p - 1)} className="p-1 rounded hover:bg-sunken disabled:opacity-30 cursor-pointer">
              <ChevronLeft size={16} />
            </button>
            <span className="tnum px-1">
              {page + 1} / {pages}
            </span>
            <button aria-label="Next page" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)} className="p-1 rounded hover:bg-sunken disabled:opacity-30 cursor-pointer">
              <ChevronRight size={16} />
            </button>
          </span>
        </div>
      </Panel>
      {task && <DataPromptPanel task={task} opp={acceptedOpp} repo={detail?.repo} accepted />}
    </div>
  );
}
