import { AlertTriangle, Check, ExternalLink } from "lucide-react";
import type { BaseModelCandidate, BaseModelSelection } from "../api/types";
import { Chip, InfoTip, Meter, Panel, Spinner } from "../components/ui";
import { cn, num, pct } from "../lib/format";

function Metric({ label, value, info }: { label: string; value: React.ReactNode; info?: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-muted flex items-center gap-1">
        {label}
        {info && <InfoTip>{info}</InfoTip>}
      </div>
      <div className="text-[13px] font-medium tnum text-ink">{value}</div>
    </div>
  );
}

function CandidateCard({ c, best, live }: { c: BaseModelCandidate; best: number; live: boolean }) {
  const running = c.status === "running";
  const failed = c.status === "failed";
  return (
    <div className={cn("rounded-[8px] border p-3 min-w-0", c.chosen ? "border-accent bg-accent-soft" : "border-line-soft")}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-[13.5px] font-semibold text-ink">
            {c.url ? (
              <a href={c.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:text-accent">
                {c.name} <ExternalLink size={11} className="text-muted" />
              </a>
            ) : (
              c.name
            )}
          </div>
          {c.detail && <div className="text-[11.5px] text-muted mt-0.5 leading-snug">{c.detail}</div>}
        </div>
        <div className="shrink-0">
          {c.chosen ? (
            <Chip tone="accent">
              <Check size={11} /> Selected
            </Chip>
          ) : running ? (
            <Chip tone="neutral">
              <Spinner size={11} /> {live ? "training" : "running"}
            </Chip>
          ) : failed ? (
            <Chip tone="bad">
              <AlertTriangle size={11} /> failed
            </Chip>
          ) : c.status === "pending" ? (
            <Chip tone="neutral">waiting</Chip>
          ) : null}
        </div>
      </div>
      {failed ? (
        <p className="mt-2 text-[12px] text-bad break-words">{c.error}</p>
      ) : (
        <>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-[22px] font-semibold tnum leading-none">{c.val_acc != null ? pct(c.val_acc) : "—"}</span>
            <span className="text-[11.5px] text-muted">val accuracy{c.best_epoch != null ? ` · best epoch ${c.best_epoch}` : ""}</span>
          </div>
          <Meter className="mt-2" value={c.val_acc ?? 0} color={c.chosen ? "var(--accent)" : "var(--muted)"} marker={best > 0 && !c.chosen ? best : undefined} />
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Metric label="Val soft-CE" value={c.val_soft_ce != null ? num(c.val_soft_ce, 3) : "—"} info="Soft cross-entropy on the validation split: how much probability the model puts on the right label. Lower is better. It breaks ties when two base models have the same accuracy." />
            <Metric label="Embed / text" value={c.embed_ms != null ? `${num(c.embed_ms, 0)} ms` : "—"} info="Time to embed one text with this base model, uncached, on this machine. Almost all of a decision's latency is this step (the head adds well under a millisecond), so it is roughly the cost per call in production." />
            <Metric label="Embedding" value={c.dim != null ? `${c.dim}-d` : "—"} />
            <Metric label="Time" value={c.seconds != null ? `${num(c.seconds, 0)} s` : "—"} />
          </div>
          {c.note && <p className="mt-2 text-[11.5px] text-muted">{c.note}</p>}
        </>
      )}
    </div>
  );
}

/** Base-model selection: each candidate's probe run, scored on val; the winner trains next. */
export function BaseModelPanel({ sel, active }: { sel: BaseModelSelection; active: boolean }) {
  const best = Math.max(0, ...sel.candidates.map((c) => c.val_acc ?? 0));
  const chosen = sel.candidates.find((c) => c.chosen) ?? sel.candidates.find((c) => c.spec === sel.chosen);
  return (
    <Panel
      title="Base model"
      info={
        <>
          The base model is the frozen language model that turns each text into a vector (its last-token embedding). Only the small CLM head on top is trained, so the base model decides how separable your labels are to begin with.
          <br />
          <br />
          Each candidate gets the same head-training run and is scored on the validation split. The test split is never used here, so the final evaluation stays an honest estimate. Qwen3-8B starts from the reference CLM head; Winnow-12B has no reference head, so its head trains from scratch.
        </>
      }
      sub={
        active
          ? "Training a head on each candidate and scoring it on the validation split. The test split is kept untouched for the final evaluation."
          : chosen
            ? `${chosen.name} won on the validation split and is the base model for training. The test split was not used to choose.`
            : "Highest validation accuracy wins; ties go to lower soft-CE, then faster embedding."
      }
      aside={active ? <Spinner size={13} /> : null}
    >
      <div className={cn("grid gap-3", sel.candidates.length > 1 ? "md:grid-cols-2" : "")}>
        {sel.candidates.map((c) => (
          <CandidateCard key={c.spec} c={c} best={best} live={active} />
        ))}
      </div>
    </Panel>
  );
}
