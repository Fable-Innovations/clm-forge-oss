/* Test-only: an in-memory labelling queue with the backend's rules (forge.labelling: a user label wins, the labelling
 * agent disposes every pending row, undo walks back user actions). Component tests serve the label-set routes from it.
 * Never imported by app code. */
import type { LabelCounts, LabelOption, QueueFilter, LabelQueuePage, QueueRow, RealRow, TaskSchema } from "../api/types";

function options(task: TaskSchema, row: RealRow): LabelOption[] {
  const q = row.question_meta?.question;
  const type = q?.type ?? task.type;
  const crit = (q && "criteria" in q ? q.criteria : task.criteria) as TaskSchema["criteria"];
  if (type === "choice") return Object.entries((crit as Record<string, string>) ?? {}).map(([key, description]) => ({ key, description }));
  if (type === "score") return ((crit as string[]) ?? []).map((d, i) => ({ key: String(i), description: d }));
  const c = (crit && !Array.isArray(crit) ? crit : {}) as Record<string, string>;
  return [
    { key: "true", description: c.true ?? "yes" },
    { key: "false", description: c.false ?? "no" },
  ];
}

const pending = (r: RealRow) => r.label_state === "needs_label" && !r.excluded;

export class LabelSetStub {
  rows: RealRow[] = [];
  undoStack: { prev: RealRow }[] = [];
  task: TaskSchema;
  constructor(task: TaskSchema) {
    this.task = task;
  }

  add(rows: RealRow[]) {
    const have = new Set(this.rows.map((r) => r.rid));
    for (const r of rows) if (!have.has(r.rid)) this.rows.push(r);
  }
  pending() {
    return this.rows.filter(pending);
  }
  counts(): LabelCounts {
    const lab = this.rows.filter((r) => r.label_state === "labelled");
    return {
      needs_label: this.rows.filter(pending).length,
      user: lab.filter((r) => r.label_source === "user").length,
      teacher: lab.filter((r) => r.label_source === "teacher").length,
      agent: lab.filter((r) => r.label_source === "agent").length,
      adjudicated: lab.filter((r) => r.adjudicated).length,
      export: lab.filter((r) => r.label_source === "export").length,
      excluded: this.rows.filter((r) => r.excluded && r.label_state === "needs_label").length,
      excluded_by_agent: this.rows.filter((r) => r.label_state === "excluded_by_agent").length,
      suggested: this.rows.filter((r) => pending(r) && r.suggested_label != null).length,
      total: this.rows.length,
    };
  }
  out(r: RealRow): QueueRow {
    return { ...r, options: options(this.task, r) };
  }
  page(state: QueueFilter, offset: number, limit: number): Pick<LabelQueuePage, "total" | "items" | "counts" | "task"> {
    let xs = this.rows;
    if (state === "needs_label") xs = xs.filter((r) => r.label_state === "needs_label").sort((a, b) => Number(a.excluded) - Number(b.excluded) || Number(a.skipped) - Number(b.skipped));
    else if (state === "labelled" || state === "excluded_by_agent") xs = xs.filter((r) => r.label_state === state);
    else if (state !== "all") xs = xs.filter((r) => r.label_state === "labelled" && r.label_source === state);
    return { total: xs.length, items: xs.slice(offset, offset + limit).map((r) => this.out(r)), counts: this.counts(), task: this.task };
  }
  private find(rid: string) {
    const i = this.rows.findIndex((r) => r.rid === rid);
    if (i < 0) throw new Error(`no row ${rid} in this queue`);
    return i;
  }
  label(rid: string, label: string) {
    const i = this.find(rid);
    const keys = options(this.task, this.rows[i]).map((o) => o.key);
    if (!keys.includes(label)) throw new Error(`label must be one of ${keys.join(", ")}`);
    this.undoStack.push({ prev: this.rows[i] });
    this.rows[i] = { ...this.rows[i], label, label_state: "labelled", label_source: "user", skipped: false, excluded: false, updated_at: new Date().toISOString() };
    return this.rows[i];
  }
  skip(rid: string) {
    const i = this.find(rid);
    this.undoStack.push({ prev: this.rows[i] });
    this.rows[i] = { ...this.rows[i], skipped: true };
    return this.rows[i];
  }
  undo(): RealRow | null {
    const last = this.undoStack.pop();
    if (!last) return null;
    const i = this.find(last.prev.rid);
    this.rows[i] = last.prev;
    return last.prev;
  }
    /** The labelling agent: agreeing passes label (every 4th row needed the adjudicator), the last row is
   * excluded with a reason. Never touches user or export labels. */
  agentLabel(guess: (r: RealRow) => { label: string; conf: number }) {
    const todo = this.pending();
    const ids = new Set(todo.map((r) => r.rid));
    let k = 0;
    this.rows = this.rows.map((r) => {
      if (!ids.has(r.rid)) return r;
      const i = k++;
      if (i === todo.length - 1 && todo.length > 3)
        return { ...r, label: null, label_state: "excluded_by_agent" as const, exclusion_reason: "passes disagreed (billing, account); adjudicator abstained: the ticket mixes both", adjudicated: true };
      const g = guess(r);
      return { ...r, label: g.label, label_state: "labelled" as const, label_source: "agent" as const, adjudicated: i % 4 === 3, teacher_conf: g.conf, skipped: false };
    });
  }
}

export function realRow(rid: string, state: string, head: string, extra: Partial<RealRow> = {}): RealRow {
  return {
    rid,
    state,
    question_meta: {},
    source: "imported",
    split: "train",
    label: null,
    label_state: "needs_label",
    label_source: null,
    suggested_label: null,
    teacher_conf: null,
    teacher_votes: [],
    head,
    skipped: false,
    excluded: false,
    updated_at: new Date().toISOString(),
    ...extra,
  };
}
