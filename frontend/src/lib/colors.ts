import type { TaskSchema } from "../api/types";

/** Fixed-order categorical slots. Colour follows the label (entity), never its rank. */
export const SLOTS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)", "var(--c8)"];

export function labelOrder(task: TaskSchema | null | undefined, fallback: string[] = []): string[] {
  if (!task) return fallback;
  if (task.type === "choice" && task.criteria && !Array.isArray(task.criteria)) return Object.keys(task.criteria);
  if (task.type === "score" && Array.isArray(task.criteria)) return task.criteria.map((_, i) => String(i));
  if (task.type === "noul") return ["false", "true"];
  return fallback;
}

export function makeColorOf(labels: string[]): (label: string) => string {
  const map = new Map<string, string>();
  labels.forEach((l, i) => map.set(l, i < SLOTS.length ? SLOTS[i] : "var(--muted)"));
  return (l: string) => map.get(l) ?? "var(--muted)";
}

/** Read a CSS variable's resolved value (for canvas drawing). */
export function cssVar(name: string, el: Element = document.documentElement): string {
  return getComputedStyle(el).getPropertyValue(name).trim() || "#888";
}

/** Sequential ramp lookup for v in [0,1] (blue, light -> dark in light mode; inverted steps in dark). */
export const SEQ = ["var(--s0)", "var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)"];
export function seqColor(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  return SEQ[Math.min(SEQ.length - 1, Math.floor(t * SEQ.length))];
}
/** Index of the ramp step — >= 5 means use light text on light theme. */
export function seqStep(v: number): number {
  const t = Math.max(0, Math.min(1, v));
  return Math.min(SEQ.length - 1, Math.floor(t * SEQ.length));
}
