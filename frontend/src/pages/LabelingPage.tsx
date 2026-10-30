import { NavLink, useParams } from "react-router-dom";
import { ArrowLeft, FlaskConical, Tags } from "lucide-react";
import { useLabelsets } from "../api/queries";
import type { LabelSetSummary } from "../api/types";
import { LabelQueue } from "../components/LabelQueue";
import { Chip, Empty, Meter, Panel, Spinner } from "../components/ui";
import { STATE_LABEL } from "../components/StateBadge";

function SetRow({ s }: { s: LabelSetSummary }) {
  const done = s.counts.total - s.counts.needs_label - s.counts.excluded;
  const to = s.kind === "project" ? `/experiments/${s.id.slice(2)}/label` : `/labelling/${encodeURIComponent(s.id)}`;
  return (
    <li>
      <NavLink to={to} className="proximity flex items-center gap-3 px-4 py-3 border-b border-line-soft hover:bg-sunken/60">
        {s.kind === "project" ? <FlaskConical size={14} className="text-muted shrink-0" /> : <Tags size={14} className="text-muted shrink-0" />}
        <span className="min-w-0 flex-1">
          <span className="block text-[13.5px] font-medium truncate">{s.name}</span>
          <span className="block text-[11.5px] text-muted truncate">
            {s.kind === "project" ? `experiment · ${s.state ? STATE_LABEL[s.state] : ""}` : `head directory · ${s.dir ?? ""}`}
          </span>
        </span>
        {s.counts.needs_label > 0 ? <Chip tone="warn">{s.counts.needs_label} need a label</Chip> : <Chip tone="good">all labelled</Chip>}
        <span className="w-[160px] flex items-center gap-2">
          <Meter value={s.counts.total ? done / s.counts.total : 0} className="flex-1" />
          <span className="text-[11.5px] text-muted tnum">
            {done}/{s.counts.total}
          </span>
        </span>
      </NavLink>
    </li>
  );
}

/**
 * Every labelling queue: experiments with real rows, and registered head directories (Settings
 * `labelset_roots` / `FORGE_LABELSETS`). `/labelling/:lsid` labels one of them.
 */
export function LabelingPage() {
  const { lsid } = useParams();
  const sets = useLabelsets();
  if (lsid) {
    const id = decodeURIComponent(lsid);
    const s = sets.data?.find((x) => x.id === id);
    return (
      <div className="p-4 max-w-[1400px] mx-auto space-y-4">
        <NavLink to="/labelling" className="inline-flex items-center gap-1 text-[12.5px] text-muted hover:text-ink">
          <ArrowLeft size={13} /> All label sets
        </NavLink>
        <Panel title={s?.name ?? id} sub="Your label always wins over the teacher's and is never sent anywhere." collapsible={false}>
          <LabelQueue lsid={id} />
        </Panel>
      </div>
    );
  }
  return (
    <div className="p-4 max-w-[1000px] mx-auto space-y-4">
      <Panel title="Labelling" sub="Real rows that need a label, or a second look at the teacher's." collapsible={false} bodyClass="p-0">
        {sets.isLoading ? (
          <div className="p-8 text-center text-muted text-[13px]">
            <span className="inline-flex items-center gap-2">
              <Spinner /> Loading…
            </span>
          </div>
        ) : sets.isError ? (
          <p className="p-4 text-[12.5px] text-bad">{(sets.error as Error).message}</p>
        ) : !sets.data?.length ? (
          <Empty icon={<Tags size={22} />} title="No label sets yet">
            An experiment gets one when curation finds real rows. Head directories appear here once they have a <code className="font-mono text-[12px]">real_rows.jsonl</code> and their root is
            listed in Settings (labelset_roots) or FORGE_LABELSETS.
          </Empty>
        ) : (
          <ul>
            {sets.data.map((s) => (
              <SetRow key={s.id} s={s} />
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
