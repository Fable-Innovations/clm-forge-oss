import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { JournalEntry } from "../api/types";
import { Chip, Empty, Select, Spinner } from "../components/ui";
import { clock } from "../lib/format";

const TONE: Record<string, "neutral" | "accent" | "good" | "warn" | "bad" | "option"> = {
  need: "option",
  remedy: "option",
  promote: "good",
  run_done: "good",
  reject: "warn",
  defer: "warn",
  carry: "warn",
  export_contradicted: "warn",
  run_failed: "bad",
  head_failed: "bad",
  clean: "warn",
  gate_policy: "warn",
  run_done_with_failures: "warn",
  interrupted: "warn",
  encoder_changed: "accent",
  target_edited: "neutral",
  target_created: "accent",
  target_disabled: "warn",
};

/** The counts behind an entry, flattened to `key value` pairs. */
export function countsText(c: Record<string, unknown>): string {
  return Object.entries(c)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${k} ${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
    .join(" · ");
}

/**
 * Every autopilot decision with its reason and counts, newest first: needs, remedies, generation, rounds,
 * deferrals, gate and promotion. Filterable by head and action; scoped to a target or a run.
 */
export function ActivityJournal({
  target,
  run,
  limit = 300,
  refetchMs = 5_000,
}: {
  target?: string;
  run?: string;
  limit?: number;
  /** Poll interval; false stops polling (a finished run's journal no longer changes). */
  refetchMs?: number | false;
}) {
  const [head, setHead] = useState("");
  const [action, setAction] = useState("");
  const q = useQuery({
    queryKey: qk.journal({ target, run, head: head || undefined, action: action || undefined }),
    queryFn: () => api.journal({ target, run, head: head || undefined, action: action || undefined, limit }),
    refetchInterval: refetchMs,
  });
  // the filter menus come from the unfiltered journal, so a filter never hides its own alternatives
  const all = useQuery({ queryKey: qk.journal({ target, run }), queryFn: () => api.journal({ target, run, limit }), refetchInterval: refetchMs === false ? false : 15_000 });
  const heads = useMemo(() => [...new Set((all.data ?? []).map((e) => e.head).filter(Boolean))].sort() as string[], [all.data]);
  const actions = useMemo(() => [...new Set((all.data ?? []).map((e) => e.action))].sort(), [all.data]);
  return (
    <div className="space-y-2" aria-label="Activity journal">
      <div className="flex flex-wrap items-center gap-2">
        <Select aria-label="Filter by head" value={head} onChange={(e) => setHead(e.target.value)} className="w-[200px]">
          <option value="">All heads</option>
          {heads.map((h) => (
            <option key={h} value={h}>
              {h}
            </option>
          ))}
        </Select>
        <Select aria-label="Filter by action" value={action} onChange={(e) => setAction(e.target.value)} className="w-[170px]">
          <option value="">All actions</option>
          {actions.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </Select>
        {q.isFetching && <Spinner size={12} />}
      </div>
      {q.isError ? (
        <p className="text-[12.5px] text-bad">{(q.error as Error).message}</p>
      ) : !q.data ? null : !q.data.length ? (
        <Empty title="Nothing journalled yet">Runs record every need they detect, what they did about it and why.</Empty>
      ) : (
        <ol className="divide-y divide-line-soft">
          {q.data.map((e) => (
            <JournalRow key={e.id} e={e} />
          ))}
        </ol>
      )}
    </div>
  );
}

function JournalRow({ e }: { e: JournalEntry }) {
  const counts = countsText(e.counts);
  return (
    <li className="py-1.5 grid grid-cols-[64px_minmax(0,1fr)] gap-x-3 text-[12.5px]" data-testid="journal-entry">
      <span className="text-muted tnum">{clock(e.ts)}</span>
      <span className="min-w-0">
        <span className="inline-flex flex-wrap items-center gap-1.5">
          <Chip tone={TONE[e.action] ?? "neutral"}>{e.action}</Chip>
          {e.head && <span className="font-mono text-[11.5px] text-ink-2">{e.head}</span>}
          {e.round != null && <span className="text-muted">round {e.round}</span>}
          <span className="text-ink">{e.reason}</span>
        </span>
        {counts && <span className="block text-[11.5px] text-muted font-mono truncate" title={counts}>{counts}</span>}
      </span>
    </li>
  );
}
