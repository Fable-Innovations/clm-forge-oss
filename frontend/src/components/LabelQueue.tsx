import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Keyboard, Lock, SkipForward, Undo2 } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { LabelSource, QueueFilter, QueueRow, TaskSchema } from "../api/types";
import { Button, Chip, Empty, Meter, Segmented, Select, Spinner } from "./ui";
import { cn } from "../lib/format";

const PAGE = 100;
const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

export const SOURCE_TEXT: Record<LabelSource, string> = { user: "you", teacher: "teacher", agent: "agent", export: "export" };

/** Who labelled a row, as a chip. Null = not labelled yet (or, on a synthetic row, the generator). */
export function LabelSourceChip({ source, className }: { source: LabelSource | null | undefined; className?: string }) {
  if (!source) return null;
  const tone = source === "user" ? "good" : source === "export" ? "accent" : "neutral";
  const title = {
    user: "Labelled by you",
    teacher: "Labelled by the teacher LLM",
    agent: "Labelled by the labelling agent (teacher passes, adjudicated when they disagreed)",
    export: "Label from the exported data",
  }[source];
  return (
    <Chip tone={tone} title={title} className={className}>
      {SOURCE_TEXT[source]}
    </Chip>
  );
}

/** The question this row is asked: its own (heads whose options differ per row) or the task's. */
function rowQuestion(task: TaskSchema | null, row: QueueRow): string {
  return row.question_meta?.question?.instructions || task?.instructions || "";
}

/**
 * The labelling queue of one label set (an experiment's real rows, or a registered head directory): optional review
 * and override of what the labelling agent decided. Keys: 1–9 label, S or → skip, ← previous, ⌘Z / Ctrl+Z or
 * U undo. Shared by the experiment's Labels tab and the Labelling page. While a stage locks the queue,
 * `onOverride` (experiments) records the label as an override that the next labelling stage applies.
 */
export function LabelQueue({ lsid, header, onOverride }: {
  lsid: string;
  header?: ReactNode;
  onOverride?: (rid: string, label: string) => Promise<unknown>;
}) {
  const qc = useQueryClient();
  const [filter, setFilter] = useState<QueueFilter>("needs_label");
  const [head, setHead] = useState("");
  const [i, setI] = useState(0);
  const [focusRid, setFocusRid] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);

  const q = useQuery({
    queryKey: [...qk.labelQueue(lsid), { filter, head }],
    queryFn: () => api.labelQueue(lsid, { state: filter, head: head || undefined, limit: PAGE }),
    placeholderData: keepPreviousData,
  });
  const items = q.data?.items ?? [];
  const heads = useMemo(() => [...new Set(items.map((r) => r.head).filter(Boolean))] as string[], [items]);
  const counts = q.data?.counts;
  const editable = q.data?.editable ?? false;
  const canOverride = !editable && !!onOverride && !!q.data;

  // keep the cursor on the row an undo restored, else inside the list
  useEffect(() => {
    if (!q.data) return;
    if (focusRid) {
      const k = items.findIndex((r) => r.rid === focusRid);
      if (k >= 0) setI(k);
      setFocusRid(null);
    } else if (i >= items.length) setI(Math.max(0, items.length - 1));
  }, [q.data]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setI(0), [filter, head]);

  const row: QueueRow | undefined = items[i];
  const refresh = useCallback(() => {
    qc.invalidateQueries({ queryKey: qk.labelQueue(lsid) });
    qc.invalidateQueries({ queryKey: qk.labelsets });
    if (lsid.startsWith("p:")) {
      const pid = lsid.slice(2);
      qc.invalidateQueries({ queryKey: qk.project(pid) });
      qc.invalidateQueries({ queryKey: qk.dataset(pid) });
    }
  }, [qc, lsid]);

  const act = useMutation({
    mutationFn: (a: { kind: "label"; rid: string; label: string } | { kind: "skip"; rid: string } | { kind: "undo" }) =>
      a.kind === "undo" ? api.labelUndo(lsid) : api.labelRow(lsid, a.rid, a.kind === "skip" ? { skip: true } : { label: a.label }),
    onSuccess: (res, a) => {
      if (a.kind === "undo") {
        if (res.row) setFocusRid(res.row.rid);
        setFlash(res.row ? "Undone" : "Nothing to undo");
      } else if (filter !== "needs_label") setI((k) => Math.min(k + 1, Math.max(0, items.length - 1)));
      // a skipped row sorts last: follow the row that came after it
      else if (a.kind === "skip") setFocusRid(items[i + 1]?.rid ?? items[0]?.rid ?? null);
      // labelling in the needs-label view: the row leaves the list and the next one takes its place
    },
    onSettled: refresh,
  });

  const override = useMutation({
    mutationFn: (a: { rid: string; label: string }) => onOverride!(a.rid, a.label),
    onSuccess: (_r, a) => setFlash(`Override ${a.label} recorded: applied at the next labelling stage`),
    onSettled: refresh,
  });
  const label = useCallback(
    (key: string) => {
      if (!row) return;
      if (editable && !act.isPending) act.mutate({ kind: "label", rid: row.rid, label: key });
      else if (canOverride && !override.isPending) override.mutate({ rid: row.rid, label: key });
    },
    [row, editable, act, canOverride, override],
  );
  const skip = useCallback(() => {
    if (!row) return;
    if (filter === "needs_label" && editable && !act.isPending) act.mutate({ kind: "skip", rid: row.rid });
    else setI((k) => Math.min(k + 1, Math.max(0, items.length - 1)));
  }, [row, filter, editable, act, items.length]);
  const undo = useCallback(() => {
    if (editable && !act.isPending) act.mutate({ kind: "undo" });
  }, [editable, act]);
  const prev = useCallback(() => setI((k) => Math.max(0, k - 1)), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (document.querySelector("dialog[open]")) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undo();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (/^[1-9]$/.test(e.key)) {
        const opt = row?.options[Number(e.key) - 1];
        if (opt) {
          e.preventDefault();
          label(opt.key);
        }
      } else if (e.key === "s" || e.key === "S" || e.key === "ArrowRight") {
        e.preventDefault();
        skip();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        prev();
      } else if (e.key === "u" || e.key === "U") {
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [row, label, skip, undo, prev]);

  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1500);
    return () => clearTimeout(t);
  }, [flash]);

  const done = counts ? counts.total - counts.needs_label - counts.excluded - counts.excluded_by_agent : 0;
  return (
    <div className="space-y-3" aria-label="Labelling queue">
      <div className="flex flex-wrap items-center gap-3">
        {header}
        <Segmented<QueueFilter>
          value={filter}
          onChange={setFilter}
          options={[
            { value: "needs_label", label: `Needs label${counts ? ` · ${counts.needs_label}` : ""}` },
            { value: "agent", label: `Agent${counts ? ` · ${counts.agent}` : ""}` },
            { value: "teacher", label: `Teacher${counts ? ` · ${counts.teacher}` : ""}` },
            { value: "excluded_by_agent", label: `Excluded${counts ? ` · ${counts.excluded_by_agent}` : ""}` },
            { value: "all", label: "All" },
          ]}
        />
        {heads.length > 1 && (
          <Select value={head} onChange={(e) => setHead(e.target.value)} className="w-[180px]" aria-label="Filter by head">
            <option value="">All heads</option>
            {heads.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </Select>
        )}
        <div className="flex-1 min-w-[160px] flex items-center gap-2">
          <Meter value={counts && counts.total ? done / counts.total : 0} className="flex-1" />
          <span className="text-[12px] text-muted tnum whitespace-nowrap">
            {counts ? `${done} of ${counts.total} labelled` : "…"}
            {counts?.excluded ? ` · ${counts.excluded} set aside` : ""}
            {counts?.excluded_by_agent ? ` · ${counts.excluded_by_agent} excluded by the agent` : ""}
          </span>
        </div>
      </div>

      {!editable && q.data && (
        <p className="flex items-center gap-1.5 text-[12.5px] text-muted">
          <Lock size={12} />
          {canOverride
            ? `Locked while the experiment is ${q.data.state}: a label you pick now is recorded as an override and applied at the next labelling stage.`
            : `Read-only while the experiment is ${q.data.state}: a running stage is reading the dataset.`}
        </p>
      )}
      {override.isError && <p className="text-[12.5px] text-bad">{(override.error as Error).message}</p>}
      {q.isError && <p className="text-[12.5px] text-bad">{(q.error as Error).message}</p>}
      {act.isError && <p className="text-[12.5px] text-bad">{(act.error as Error).message}</p>}

      {q.isLoading ? (
        <div className="p-8 text-center text-muted text-[13px]">
          <span className="inline-flex items-center gap-2">
            <Spinner /> Loading…
          </span>
        </div>
      ) : !row ? (
        <Empty title={filter === "needs_label" ? "Nothing needs a label" : "No rows here"}>
          {filter === "needs_label" ? "Every real row in this set has a disposition. Review the agent's labels under Agent." : "Try another filter."}
        </Empty>
      ) : (
        <div className="grid gap-3 grid-cols-1 lg:grid-cols-[1.3fr_1fr] min-h-0">
          <section aria-label="State" className="min-w-0 border border-line rounded-[12px] bg-sunken/60">
            <header className="flex items-center gap-2 px-3 h-9 border-b border-line text-[12px] text-muted">
              <span className="tnum">
                Row {i + 1} of {q.data?.total ?? items.length}
              </span>
              <Chip>{row.source}</Chip>
              <Chip>{row.split}</Chip>
              {row.skipped && <Chip tone="warn">skipped</Chip>}
              {row.excluded && <Chip tone="warn">excluded</Chip>}
              <span className="flex-1" />
              <span className="font-mono text-[11px] truncate max-w-[200px]" title={row.rid}>
                {row.rid}
              </span>
            </header>
            <pre data-testid="label-state" className="font-mono text-[12px] leading-[1.5] whitespace-pre-wrap break-words p-3 max-h-[60vh] overflow-y-auto">
              {row.state}
            </pre>
          </section>
          <section aria-label="Question" className="min-w-0 space-y-3">
            <p className="text-[13.5px] leading-snug whitespace-pre-wrap">{rowQuestion(q.data?.task ?? null, row)}</p>
            <div className="flex flex-wrap items-center gap-1.5">
              {row.label_state === "labelled" ? (
                <>
                  <span className="text-[12px] text-muted">Labelled</span>
                  <Chip tone="good">{row.label}</Chip>
                  <span className="text-[12px] text-muted">by</span>
                  <LabelSourceChip source={row.label_source} />
                  {row.adjudicated && <Chip tone="option" title="The teacher passes disagreed; the adjudicator picked this label">adjudicated</Chip>}
                </>
              ) : row.label_state === "excluded_by_agent" ? (
                <>
                  <Chip tone="warn">excluded by the agent</Chip>
                  {row.exclusion_reason && <span className="text-[12px] text-ink-2">{row.exclusion_reason}</span>}
                </>
              ) : (
                <Chip tone="warn">needs a label</Chip>
              )}
              {row.suggested_label != null && row.suggested_label !== row.label && (
                <Chip tone="option" title={row.teacher_votes.length ? row.teacher_votes.map((v) => `${v.pass}: ${v.label ?? "?"}`).join(", ") : undefined}>
                  teacher suggests {row.suggested_label}
                  {row.teacher_conf != null && ` · ${Math.round(row.teacher_conf * 100)}%`}
                </Chip>
              )}
            </div>
            <ol className="space-y-1.5" aria-label="Options">
              {row.options.map((o, k) => {
                const current = row.label === o.key;
                const suggested = row.suggested_label === o.key;
                return (
                  <li key={o.key}>
                    <button
                      onClick={() => label(o.key)}
                      disabled={!(editable || canOverride) || act.isPending || override.isPending}
                      aria-pressed={current}
                      className={cn(
                        "w-full text-left flex items-start gap-2.5 px-3 py-2 rounded-[10px] border transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-60",
                        current ? "border-good bg-good-soft" : suggested ? "border-option/50 bg-option-soft/60" : "border-line hover:border-accent/60 hover:bg-sunken",
                      )}
                    >
                      {k < 9 && <kbd className="shrink-0 w-5 h-5 grid place-items-center rounded-[5px] border border-line text-[11px] font-mono text-muted bg-raised">{k + 1}</kbd>}
                      <span className="min-w-0">
                        <span className="block text-[13px] font-medium">{o.key}</span>
                        {o.description && <span className="block text-[12px] text-ink-2 leading-snug">{o.description}</span>}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="ghost" icon={<ArrowLeft size={13} />} onClick={prev} disabled={i === 0}>
                Previous
              </Button>
              <Button size="sm" variant="secondary" icon={filter === "needs_label" ? <SkipForward size={13} /> : <ArrowRight size={13} />} onClick={skip} disabled={act.isPending}>
                {filter === "needs_label" ? "Skip" : "Next"}
              </Button>
              <Button size="sm" variant="ghost" icon={<Undo2 size={13} />} onClick={undo} disabled={!editable || act.isPending}>
                Undo
              </Button>
              {flash && <span className="text-[12px] text-muted">{flash}</span>}
              <span className="flex-1" />
              <span className="inline-flex items-center gap-1 text-[11.5px] text-muted">
                <Keyboard size={12} /> 1–9 label · S/→ skip · ← back · {isMac ? "⌘Z" : "Ctrl+Z"}/U undo
              </span>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
