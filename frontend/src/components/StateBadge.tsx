import { Bot, Cog, User } from "lucide-react";
import type { Actor, ProjectState } from "../api/types";
import { cn } from "../lib/format";

export const STATE_LABEL: Record<ProjectState, string> = {
  created: "Created",
  analyzing: "Analyzing",
  proposed: "Awaiting review",
  accepted: "Accepted",
  curating: "Curating data",
  labelling: "Labelling",
  label_review: "Label review",
  selecting: "Selecting model",
  training: "Training",
  evaluating: "Evaluating",
  reviewing: "Reviewing gate",
  packaging: "Packaging",
  delivered: "Delivered",
  no_fit: "No fit",
  cancelled: "Cancelled",
  errored: "Errored",
};

export const ACTIVE_STATES: ProjectState[] = ["analyzing", "accepted", "curating", "labelling", "selecting", "training", "evaluating", "reviewing", "packaging"];

export function stateTone(s: ProjectState | null | undefined): "accent" | "good" | "warn" | "bad" | "neutral" | "option" {
  if (!s) return "neutral";
  if (s === "delivered") return "good";
  if (s === "proposed" || s === "label_review") return "option";
  if (s === "errored") return "bad";
  if (s === "no_fit" || s === "cancelled") return "neutral";
  if (ACTIVE_STATES.includes(s)) return "accent";
  return "neutral";
}

export function StateBadge({ state, className }: { state: ProjectState | null | undefined; className?: string }) {
  const tone = stateTone(state);
  const active = state ? ACTIVE_STATES.includes(state) : false;
  const cls = {
    accent: "text-accent bg-accent-soft border-accent/25",
    good: "text-good bg-good-soft border-good/25",
    warn: "text-warn bg-warn-soft border-warn/25",
    bad: "text-bad bg-bad-soft border-bad/25",
    neutral: "text-ink-2 bg-sunken border-line",
    option: "text-option bg-option-soft border-option/30",
  }[tone];
  const dot = {
    accent: "var(--accent)",
    good: "var(--good)",
    warn: "var(--warn)",
    bad: "var(--bad)",
    neutral: "var(--muted)",
    option: "var(--option)",
  }[tone];
  return (
    <span className={cn("inline-flex items-center gap-1.5 h-6 pl-2 pr-2.5 rounded-full border text-[12px] font-medium whitespace-nowrap", cls, className)}>
      <span className={cn("w-1.5 h-1.5 rounded-full", active && "pulse-dot")} style={{ background: dot, boxShadow: tone !== "neutral" ? `0 0 6px ${dot}` : undefined }} />
      {state ? STATE_LABEL[state] : "…"}
    </span>
  );
}

export function ActorIcon({ actor, size = 12, className }: { actor: Actor | string; size?: number; className?: string }) {
  const I = actor === "agent" ? Bot : actor === "human" ? User : Cog;
  return <I size={size} className={className} aria-label={actor} />;
}
