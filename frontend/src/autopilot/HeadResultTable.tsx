import type { HeadResultRow } from "../api/types";
import { Chip } from "../components/ui";
import { num } from "../lib/format";

const DECISION: Record<string, { tone: "good" | "warn" | "neutral" | "bad"; label: string }> = {
  promoted: { tone: "good", label: "Promoted" },
  kept: { tone: "warn", label: "Kept" },
  rejected: { tone: "warn", label: "Rejected (val)" },
  skipped: { tone: "neutral", label: "Skipped" },
  retired: { tone: "neutral", label: "Retired" },
  failed: { tone: "bad", label: "Failed" },
};

/** Per head: control and final macro-F1, against the reference head and the incumbent, the gate, the decision. */
export function HeadResultTable({ rows }: { rows: HeadResultRow[] }) {
  if (!rows.length) return <p className="text-[12.5px] text-muted">No head has started yet.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[12.5px] tnum" aria-label="Head results">
        <thead className="text-muted text-[11.5px]">
          <tr>
            <th className="font-normal py-1 pr-3">Head</th>
            <th className="font-normal pr-3" title="Synthetic-only control, val macro-F1 (chosen round)">Control F1</th>
            <th className="font-normal pr-3" title="Candidate test macro-F1, read once">Final F1</th>
            <th className="font-normal pr-3" title="Reference head / incumbent, test macro-F1">Reference / incumbent F1</th>
            <th className="font-normal pr-3">Gate</th>
            <th className="font-normal pr-3">Decision</th>
            <th className="font-normal">Rounds</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const d = r.decision ? DECISION[r.decision] : null;
            return (
              <tr key={r.head} className="border-t border-line-soft">
                <td className="py-1 pr-3 font-mono text-[12px]">{r.head}</td>
                <td className="pr-3">{num(r.control_f1, 3)}</td>
                <td className="pr-3">{num(r.final_f1, 3)}</td>
                <td className="pr-3">
                  {num(r.reference_f1, 3)} / {num(r.incumbent_f1, 3)}
                </td>
                <td className="pr-3">{r.gate_passed == null ? "—" : r.gate_passed ? <Chip tone="good">passed</Chip> : <Chip tone="warn">failed</Chip>}</td>
                <td className="pr-3">{d ? <Chip tone={d.tone}>{d.label}</Chip> : <span className="text-muted">{r.decision ?? "in progress"}</span>}</td>
                <td>{r.rounds || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
