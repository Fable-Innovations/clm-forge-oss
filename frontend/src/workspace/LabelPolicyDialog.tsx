import { useEffect, useState } from "react";
import { Bot, Combine, Sparkles, UserRound } from "lucide-react";
import type { LabelPolicy } from "../api/types";
import { Button, Dialog } from "../components/ui";
import { cn } from "../lib/format";

export const POLICY_TEXT: Record<LabelPolicy, { title: string; body: string }> = {
  agent: {
    title: "The labelling agent labels them",
    body: "The default. The teacher labels each row; disagreements go to an adjudicator, and rows it cannot settle are excluded with a reason. Nothing waits for you; you can still review and override any label in the Labels tab.",
  },
  user: {
    title: "I'll label them",
    body: "Rows without a label wait for your review in the Labels tab. No LLM sees them. Training waits until you label them or choose to go on without them.",
  },
  teacher: {
    title: "The teacher labels them",
    body: "The project's LLM labels them during curation, as before. You can still review and override every one in the Labels tab.",
  },
  mixed: {
    title: "Teacher suggests, I decide",
    body: "The LLM proposes a label for each row; nothing counts until you confirm or change it in the Labels tab.",
  },
};
const ICON: Record<LabelPolicy, typeof Bot> = { agent: Sparkles, user: UserRound, teacher: Bot, mixed: Combine };

/**
 * Who labels real rows that arrive without a label (exported tables without a value in the label column,
 * examples mined from the code). Opened from the policy chip in the Labels tab; the default is the labelling
 * agent, the other policies opt into the human label review gate.
 */
export function LabelPolicyDialog({
  open,
  onClose,
  onPick,
  current,
  pending,
  confirmLabel = "Use this",
}: {
  open: boolean;
  onClose: () => void;
  onPick: (p: LabelPolicy) => void;
  current?: LabelPolicy | null;
  pending?: boolean;
  confirmLabel?: string;
}) {
  const [choice, setChoice] = useState<LabelPolicy>(current ?? "agent");
  useEffect(() => {
    if (open) setChoice(current ?? "agent");
  }, [open, current]);
  return (
    <Dialog open={open} onClose={onClose} title="Who labels real rows without a label?" width={600}>
      <p className="text-[13px] text-ink-2 leading-relaxed mb-3">
        Real data from this repository (exported tables, examples found in the code) may arrive without a label. Synthetic examples are always labelled by the generator and checked by
        the teacher; this choice is only about your real rows.
      </p>
      <div role="radiogroup" aria-label="Label policy" className="space-y-2">
        {(Object.keys(POLICY_TEXT) as LabelPolicy[]).map((p) => {
          const I = ICON[p];
          return (
            <button
              key={p}
              role="radio"
              aria-checked={choice === p}
              onClick={() => setChoice(p)}
              className={cn(
                "w-full text-left flex items-start gap-3 px-3.5 py-3 rounded-[12px] border cursor-pointer transition-colors",
                choice === p ? "border-accent bg-accent-soft" : "border-line hover:border-accent/50 hover:bg-sunken",
              )}
            >
              <I size={16} className="mt-0.5 shrink-0 text-ink-2" />
              <span>
                <span className="block text-[13.5px] font-medium">{POLICY_TEXT[p].title}</span>
                <span className="block text-[12.5px] text-ink-2 leading-snug mt-0.5">{POLICY_TEXT[p].body}</span>
              </span>
            </button>
          );
        })}
      </div>
      <div className="flex justify-end gap-2 mt-4">
        <Button variant="ghost" onClick={onClose}>
          Back
        </Button>
        <Button variant="primary" disabled={pending} onClick={() => onPick(choice)}>
          {pending ? "Saving…" : confirmLabel}
        </Button>
      </div>
    </Dialog>
  );
}
