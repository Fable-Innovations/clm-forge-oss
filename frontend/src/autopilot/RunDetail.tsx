import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { isLive, qk, RUN_POLL_MS } from "../api/queries";
import type { HeadRunRecord, JournalEntry } from "../api/types";
import { Chip, Spinner } from "../components/ui";
import { clock, num } from "../lib/format";
import { countsText } from "./ActivityJournal";
import { RUN_TONE } from "./runState";

const OUTCOME_TONE: Record<string, "good" | "warn" | "neutral" | "bad"> = { promoted: "good", rejected: "warn", skipped: "neutral", failed: "bad" };

/** One run: per head, its needs and remedies, the rounds with control deltas, and the promotion decision. */
export function RunDetail({ runId }: { runId: string }) {
  const run = useQuery({
    queryKey: qk.autopilotRun(runId),
    queryFn: () => api.autopilotRun(runId),
    refetchInterval: (q) => (isLive(q.state.data?.state) ? RUN_POLL_MS : false),
  });
  const live = isLive(run.data?.state);
  const journal = useQuery({
    queryKey: [...qk.journal({ run: runId }), "all"],
    queryFn: () => api.journal({ run: runId, limit: 2000 }),
    refetchInterval: live ? RUN_POLL_MS : false,
  });
  if (run.isLoading) return <Spinner />;
  if (run.isError) return <p className="text-[12.5px] text-bad">{(run.error as Error).message}</p>;
  const r = run.data!;
  const heads = Object.entries(r.result.heads ?? {});
  const byHead = (h: string, actions: string[]) => (journal.data ?? []).filter((e) => e.head === h && actions.includes(e.action)).reverse();
  return (
    <div className="space-y-3 text-[12.5px]" aria-label={`Run ${r.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <Chip tone={RUN_TONE[r.state]}>{r.state.replace(/_/g, " ")}</Chip>
        <span className="text-ink-2">{r.trigger} run</span>
        <span className="text-muted">
          {clock(r.first_started ?? r.started)} → {r.ended ? clock(r.ended) : "…"}
        </span>
        {r.progress?.current && <span className="text-ink-2 truncate">{r.progress.current.label}</span>}
        {r.progress?.detail && <span className="text-muted font-mono truncate">{r.progress.detail}</span>}
      </div>
      {r.result.error && <p className="text-bad font-mono whitespace-pre-wrap">{r.result.error}</p>}
      {heads.length === 0 && !r.result.error && <p className="text-muted">No head finished yet.</p>}
      {heads.map(([h, rec]) => (
        <HeadBlock key={h} head={h} rec={rec} needs={byHead(h, ["need"])} remedies={byHead(h, ["remedy", "generate", "carry"])} />
      ))}
    </div>
  );
}

function HeadBlock({ head, rec, needs, remedies }: { head: string; rec: HeadRunRecord; needs: JournalEntry[]; remedies: JournalEntry[] }) {
  const d = rec.gate?.decision;
  return (
    <section className="border border-line rounded-[12px] p-3 space-y-2" aria-label={`Head ${head}`}>
      <header className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[12.5px] font-medium">{head}</span>
        <Chip tone={OUTCOME_TONE[rec.outcome ?? ""] ?? "neutral"}>{rec.outcome ?? "in progress"}</Chip>
        {rec.path && <span className="font-mono text-[11.5px] text-muted truncate">{rec.path}</span>}
      </header>
      {rec.why && <p className="text-ink-2">{rec.why}</p>}
      {rec.error && <p className="text-bad font-mono whitespace-pre-wrap">{rec.error}</p>}
      {(needs.length > 0 || remedies.length > 0) && (
        <div className="grid gap-2 md:grid-cols-2">
          <div>
            <h4 className="text-[11.5px] uppercase tracking-wide text-muted mb-1">Needs</h4>
            <ul className="space-y-0.5">
              {needs.map((e) => (
                <li key={e.id}>
                  <Chip tone="option">{String(e.counts.kind ?? "need")}</Chip> {e.reason}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-[11.5px] uppercase tracking-wide text-muted mb-1">Remedies</h4>
            <ul className="space-y-0.5">
              {remedies.map((e) => (
                <li key={e.id}>
                  <span className="text-muted">{e.round != null ? `r${e.round} ` : ""}</span>
                  {e.reason}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {rec.rounds && rec.rounds.length > 0 && (
        <table className="w-full text-left tnum">
          <thead className="text-muted text-[11.5px]">
            <tr>
              <th className="font-normal py-1">Round</th>
              <th className="font-normal">Val macro-F1</th>
              <th className="font-normal">Val gate</th>
              <th className="font-normal">vs controls (val)</th>
              <th className="font-normal">Real cap</th>
            </tr>
          </thead>
          <tbody>
            {rec.rounds.map((x) => (
              <tr key={x.round} className="border-t border-line-soft">
                <td className="py-1">{x.round}</td>
                <td>{num(x.val_macro_f1, 3)}</td>
                <td>{x.val_gate ? <Chip tone="good">passed</Chip> : <Chip tone="warn">failed</Chip>}</td>
                <td className="font-mono text-[11.5px]">
                  {Object.entries(x.control_val_deltas ?? {})
                    .map(([m, v]) => `${m} ${v >= 0 ? "+" : ""}${num(v, 3)}`)
                    .join(" · ") || "—"}
                </td>
                <td>{x.real_cap ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {d && (
        <div className="space-y-1">
          <p>
            <span className="text-muted">Test (read once):</span> macro-F1 {num(d.test_macro_f1, 3)}
            {d.incumbent_test_macro_f1 != null && (
              <>
                {" "}
                vs incumbent {num(d.incumbent_test_macro_f1, 3)} ({d.delta != null && d.delta >= 0 ? "+" : ""}
                {num(d.delta, 3)})
              </>
            )}
            {rec.gate?.reference && <span className="text-muted"> · reference {num(rec.gate.reference.macro_f1, 3)}</span>}
          </p>
          <p className="flex flex-wrap items-center gap-1.5">
            <Chip tone={d.promote ? "good" : "warn"}>{d.promote ? "promoted" : "kept in staging"}</Chip>
            <span className="text-ink-2">{d.reasons.join("; ")}</span>
          </p>
        </div>
      )}
      {rec.carry && Object.keys(rec.carry).length > 0 && <p className="text-muted">Carried to the next run: {countsText(rec.carry)}</p>}
    </section>
  );
}
