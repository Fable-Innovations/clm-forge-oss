import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Bot, EyeOff, Settings2, Sparkles, Tags } from "lucide-react";
import { api } from "../api/client";
import { projectLsid, qk } from "../api/queries";
import type { LabelCounts, LabelPolicy, ProjectDetail } from "../api/types";
import { LabelQueue } from "../components/LabelQueue";
import { Button, Chip, Empty, Panel, Spinner } from "../components/ui";
import { LabelPolicyDialog, POLICY_TEXT } from "./LabelPolicyDialog";
import { useWs } from "./context";

/**
 * The label review gate's actions (opt-in human policies only): label, send the pending rows to the teacher,
 * or set them aside (exclude). Shown in the status strip, the Overview and the Labels tab while the experiment
 * is in `label_review`. Leaving the gate re-enters the automatic labelling stage, which settles the labels.
 */
export function LabelGateActions({ pid, detail, onLabel }: { pid: string; detail?: ProjectDetail; onLabel?: () => void }) {
  const qc = useQueryClient();
  const pending = detail?.labels?.needs_label ?? 0;
  const refresh = () => {
    qc.invalidateQueries({ queryKey: qk.project(pid) });
    qc.invalidateQueries({ queryKey: qk.labelQueue(projectLsid(pid)) });
  };
  const teacher = useMutation({ mutationFn: () => api.labelTeacher(projectLsid(pid)), onSettled: refresh });
  const review = useMutation({ mutationFn: (mode: "continue" | "exclude") => api.reviewLabels(pid, mode), onSettled: refresh });
  const err = (teacher.error ?? review.error) as Error | null;
  if (detail?.state !== "label_review") return null;
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      {err && <span className="text-[12px] text-bad max-w-[360px] truncate" title={err.message}>{err.message}</span>}
      {pending > 0 ? (
        <>
          {onLabel && (
            <Button size="sm" variant="option" icon={<Tags size={13} />} onClick={onLabel}>
              Label
            </Button>
          )}
          <Button size="sm" variant="secondary" icon={<Bot size={13} />} disabled={teacher.isPending} onClick={() => teacher.mutate()}>
            {teacher.isPending ? "Sending…" : "Send to teacher"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            icon={<EyeOff size={13} />}
            disabled={review.isPending}
            onClick={() => {
              if (confirm(`Train without the ${pending} unlabelled rows? They stay in the queue and join the dataset once someone labels them.`)) review.mutate("exclude");
            }}
          >
            Exclude
          </Button>
        </>
      ) : (
        <Button size="sm" variant="primary" icon={<ArrowRight size={13} />} disabled={review.isPending} onClick={() => review.mutate("continue")}>
          Continue to training
        </Button>
      )}
    </span>
  );
}

/** Who labelled the real rows, as counts: the labelling stage's outcome at a glance. */
export function LabelCountsStrip({ counts }: { counts?: LabelCounts }) {
  if (!counts) return null;
  const items: [string, number, "good" | "accent" | "neutral" | "warn" | "option"][] = [
    ["agent", counts.agent, "accent"],
    ["adjudicated", counts.adjudicated, "option"],
    ["you", counts.user, "good"],
    ["teacher", counts.teacher, "neutral"],
    ["export", counts.export, "neutral"],
    ["excluded by agent", counts.excluded_by_agent, "warn"],
    ["set aside", counts.excluded, "warn"],
    ["waiting", counts.needs_label, "warn"],
  ];
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-label="Label counts">
      {items
        .filter(([, n]) => n > 0)
        .map(([name, n, tone]) => (
          <Chip key={name} tone={tone}>
            <span className="tnum">{n}</span> {name}
          </Chip>
        ))}
      <span className="text-[12px] text-muted tnum">of {counts.total} real rows</span>
    </div>
  );
}

export function LabelTab() {
  const { pid, detail, store } = useWs();
  const qc = useQueryClient();
  const [policyOpen, setPolicyOpen] = useState(false);
  const policy = useMutation({
    mutationFn: (p: LabelPolicy) => api.setLabelPolicy(projectLsid(pid), p),
    onSuccess: () => setPolicyOpen(false),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.project(pid) }),
  });
  const counts = detail?.labels;
  const state = store.state ?? detail?.state;
  const current: LabelPolicy = detail?.label_policy ?? "agent";
  const overrides = detail?.label_overrides ?? [];

  const policyChip = (
    <span className="inline-flex items-center gap-1.5 text-[12.5px] text-ink-2">
      Policy:
      <Chip tone={current === "agent" ? "accent" : "option"}>{POLICY_TEXT[current].title}</Chip>
      <Button size="sm" variant="ghost" icon={<Settings2 size={12} />} onClick={() => setPolicyOpen(true)}>
        Change
      </Button>
    </span>
  );

  return (
    <div className="p-4 space-y-4">
      <Panel
        id="labels-stage"
        title={
          state === "labelling" ? (
            <span className="inline-flex items-center gap-2">
              <Spinner /> The labelling agent is working
            </span>
          ) : state === "label_review" ? (
            `${counts?.needs_label ?? 0} real rows wait for your review`
          ) : (
            "Labelling is automatic"
          )
        }
        sub={
          state === "label_review"
            ? "This experiment opted into human review. Label them below, send them to the teacher, or go on without them."
            : "The teacher labels every real row; disagreements go to an adjudicator, and rows it cannot settle are excluded with a reason. Review or override any label below."
        }
        aside={<LabelGateActions pid={pid} detail={detail} />}
        collapsible={false}
      >
        <div className="space-y-2">
          <LabelCountsStrip counts={counts} />
          {overrides.length > 0 && (
            <p className="inline-flex items-center gap-1.5 text-[12.5px] text-ink-2">
              <Sparkles size={12} /> {overrides.length} label override{overrides.length === 1 ? "" : "s"} recorded; the next labelling stage applies{" "}
              {overrides.length === 1 ? "it" : "them"}.
            </p>
          )}
        </div>
      </Panel>
      {!counts?.total ? (
        <Empty icon={<Tags size={22} />} title="No real rows to label">
          Rows imported from your exported tables or mined from the code show up here. Synthetic examples don't: the generator labels them and the teacher checks them.
          <div className="mt-3">{policyChip}</div>
        </Empty>
      ) : (
        <Panel title="Review and override" sub="Every real row and who labelled it. Your label always wins over the agent's." collapsible={false}>
          <LabelQueue lsid={projectLsid(pid)} header={policyChip} onOverride={(rid, label) => api.overrideLabel(pid, rid, label)} />
        </Panel>
      )}
      <LabelPolicyDialog open={policyOpen} onClose={() => setPolicyOpen(false)} onPick={(p) => policy.mutate(p)} current={current} pending={policy.isPending} confirmLabel="Save policy" />
    </div>
  );
}
