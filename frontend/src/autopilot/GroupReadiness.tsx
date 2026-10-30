import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { qk } from "../api/queries";
import { Chip, Empty, Spinner } from "../components/ui";
import { clock, num } from "../lib/format";

/** Call-group readiness on each head set's candidate encoder: a group ships only when every head in it is ready. */
export function GroupReadiness() {
  const q = useQuery({ queryKey: qk.groups, queryFn: api.groups, refetchInterval: 30_000 });
  if (q.isLoading) return <Spinner />;
  if (q.isError) return <p className="text-[12.5px] text-bad">{(q.error as Error).message}</p>;
  const entries = (q.data ?? []).filter((e) => e.report);
  if (!entries.length) return <Empty title="No readiness report yet">A head-set run ends by asking its adapter for the call groups and scoring them.</Empty>;
  return (
    <div className="space-y-4 text-[12.5px]">
      {entries.map((e) => (
        <div key={e.target} className="space-y-2" aria-label={`Groups for ${e.target}`}>
          <p className="text-ink-2">
            <span className="font-mono">{e.target}</span> on <span className="font-mono">{e.report!.encoder}</span>
            <span className="text-muted"> · computed {clock(e.computed_at)}</span>
          </p>
          {e.report!.notice && <p className="text-warn">{e.report!.notice}</p>}
          <table className="w-full text-left tnum">
            <thead className="text-muted text-[11.5px]">
              <tr>
                <th className="font-normal py-1">Group</th>
                <th className="font-normal">Head</th>
                <th className="font-normal">Test macro-F1</th>
                <th className="font-normal">4B</th>
                <th className="font-normal">Ready</th>
              </tr>
            </thead>
            <tbody>
              {e.report!.groups.flatMap((g) =>
                g.heads.map((h, i) => (
                  <tr key={`${g.name}-${h.head}`} className="border-t border-line-soft" title={h.why || undefined}>
                    <td className="py-1">
                      {i === 0 && (
                        <span className="inline-flex items-center gap-1.5">
                          {g.name} <Chip tone={g.ready ? "good" : "warn"}>{g.ready ? "ready" : "not ready"}</Chip>
                        </span>
                      )}
                    </td>
                    <td className="font-mono">{h.head}</td>
                    <td>{num(h.test_macro_f1 ?? null, 3)}</td>
                    <td>{num(h.baseline_4b_macro_f1 ?? null, 3)}</td>
                    <td>{h.ready ? <Chip tone="good">yes</Chip> : <span className="text-muted">{h.why || "no"}</span>}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
