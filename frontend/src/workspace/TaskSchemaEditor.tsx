import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import type { QuestionType, TaskSchema } from "../api/types";
import { Field, Input, Select, Textarea } from "../components/ui";

function convert(t: TaskSchema, type: QuestionType): TaskSchema {
  if (type === t.type) return t;
  let criteria: TaskSchema["criteria"] = null;
  const old = t.criteria;
  if (type === "choice") {
    criteria = Array.isArray(old) ? Object.fromEntries(old.map((c, i) => [`option_${i + 1}`, c])) : old && t.type !== "noul" ? old : { yes: "", no: "" };
  } else if (type === "score") {
    criteria = Array.isArray(old) ? old : old ? Object.values(old) : ["low", "medium", "high"];
  } else {
    criteria = { true: "", false: "" };
  }
  return { ...t, type, criteria };
}

export function slugOk(s: string) {
  return /^[a-z][a-z0-9_]{1,40}$/.test(s);
}

export function validateTask(t: TaskSchema): string[] {
  const errs: string[] = [];
  if (!slugOk(t.name)) errs.push("Name must be a lowercase slug (letters, digits, underscores).");
  if (!t.question_id.trim()) errs.push("Question id is required.");
  if (!t.instructions.trim()) errs.push("Question text is required.");
  if (t.type === "choice") {
    const c = (t.criteria ?? {}) as Record<string, string>;
    const keys = Object.keys(c);
    if (keys.length < 2) errs.push("A choice question needs at least two options.");
    if (keys.some((k) => !k.trim())) errs.push("Option keys can't be empty.");
  }
  if (t.type === "score" && (!Array.isArray(t.criteria) || t.criteria.length < 2)) errs.push("A score question needs at least two levels.");
  return errs;
}

export function TaskSchemaEditor({ value, onChange, readOnly }: { value: TaskSchema; onChange: (t: TaskSchema) => void; readOnly?: boolean }) {
  const set = <K extends keyof TaskSchema>(k: K, v: TaskSchema[K]) => onChange({ ...value, [k]: v });
  const choice = value.type === "choice" && value.criteria && !Array.isArray(value.criteria) ? (value.criteria as Record<string, string>) : null;
  const levels = value.type === "score" && Array.isArray(value.criteria) ? value.criteria : null;
  const entries = choice ? Object.entries(choice) : [];

  const setEntries = (es: [string, string][]) => set("criteria", Object.fromEntries(es));

  return (
    <fieldset disabled={readOnly} className="space-y-3 min-w-0">
      <div className="grid grid-cols-[1.3fr_1fr_120px] gap-2.5">
        <Field label="Name" hint={<>package <span className="font-mono">clm_{value.name || "…"}</span></>}>
          <Input className="font-mono text-[12.5px]" value={value.name} onChange={(e) => set("name", e.target.value)} aria-invalid={!slugOk(value.name)} />
        </Field>
        <Field label="Question id">
          <Input className="font-mono text-[12.5px]" value={value.question_id} onChange={(e) => set("question_id", e.target.value)} />
        </Field>
        <Field label="Type">
          <Select value={value.type} onChange={(e) => onChange(convert(value, e.target.value as QuestionType))}>
            <option value="choice">choice</option>
            <option value="score">score</option>
            <option value="noul">yes / no</option>
          </Select>
        </Field>
      </div>
      <Field label="Question" hint="Appended after the state text on every call.">
        <Input value={value.instructions} onChange={(e) => set("instructions", e.target.value)} />
      </Field>

      <div>
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-[12.5px] font-medium text-ink-2">
            {value.type === "choice" ? "Options" : value.type === "score" ? "Levels, low to high" : "What true and false mean (optional)"}
          </span>
          <span className="text-[11.5px] text-muted">the head scores the state against these texts</span>
        </div>
        {choice && (
          <div className="space-y-1.5">
            {entries.map(([k, d], i) => (
              <div key={i} className="grid grid-cols-[minmax(80px,104px)_minmax(0,1fr)_28px] gap-1.5 items-start">
                <Input
                  className="font-mono text-[12px] h-8"
                  value={k}
                  onChange={(e) => {
                    const es = entries.slice();
                    es[i] = [e.target.value.replace(/\s+/g, "_"), d];
                    setEntries(es);
                  }}
                  aria-label="Option key"
                />
                <Textarea
                  rows={1}
                  className="text-[12.5px] min-h-8 py-1.5 [field-sizing:content] resize-none"
                  value={d}
                  onChange={(e) => {
                    const es = entries.slice();
                    es[i] = [k, e.target.value];
                    setEntries(es);
                  }}
                  aria-label={`Description for ${k}`}
                />
                {!readOnly && (
                  <button type="button" aria-label={`Remove ${k}`} onClick={() => setEntries(entries.filter((_, j) => j !== i))} className="h-8 grid place-items-center rounded text-muted hover:text-bad hover:bg-bad-soft cursor-pointer">
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            ))}
            {!readOnly && (
              <button type="button" onClick={() => setEntries([...entries, [`option_${entries.length + 1}`, ""]])} className="inline-flex items-center gap-1 text-[12.5px] text-accent font-medium mt-1 cursor-pointer">
                <Plus size={13} /> Add option
              </button>
            )}
          </div>
        )}
        {levels && (
          <ol className="space-y-1.5">
            {levels.map((l, i) => (
              <li key={i} className="grid grid-cols-[22px_1fr_auto] gap-1.5 items-center">
                <span className="text-[12px] text-muted tnum text-right">{i}</span>
                <Input
                  className="h-8 text-[12.5px]"
                  value={l}
                  onChange={(e) => {
                    const ls = levels.slice();
                    ls[i] = e.target.value;
                    set("criteria", ls);
                  }}
                />
                {!readOnly && (
                  <span className="flex">
                    <button type="button" aria-label="Move up" disabled={i === 0} onClick={() => { const ls = levels.slice(); [ls[i - 1], ls[i]] = [ls[i], ls[i - 1]]; set("criteria", ls); }} className="p-1.5 text-muted hover:text-ink disabled:opacity-30 cursor-pointer"><ArrowUp size={13} /></button>
                    <button type="button" aria-label="Move down" disabled={i === levels.length - 1} onClick={() => { const ls = levels.slice(); [ls[i + 1], ls[i]] = [ls[i], ls[i + 1]]; set("criteria", ls); }} className="p-1.5 text-muted hover:text-ink disabled:opacity-30 cursor-pointer"><ArrowDown size={13} /></button>
                    <button type="button" aria-label="Remove level" onClick={() => set("criteria", levels.filter((_, j) => j !== i))} className="p-1.5 text-muted hover:text-bad cursor-pointer"><Trash2 size={13} /></button>
                  </span>
                )}
              </li>
            ))}
            {!readOnly && (
              <button type="button" onClick={() => set("criteria", [...levels, ""])} className="inline-flex items-center gap-1 text-[12.5px] text-accent font-medium mt-1 cursor-pointer">
                <Plus size={13} /> Add level
              </button>
            )}
          </ol>
        )}
        {value.type === "noul" && (
          <div className="grid grid-cols-2 gap-2">
            {(["true", "false"] as const).map((k) => {
              const c = (value.criteria && !Array.isArray(value.criteria) ? value.criteria : {}) as Record<string, string>;
              return (
                <Field key={k} label={<span className="font-mono">{k}</span>}>
                  <Input value={c[k] ?? ""} onChange={(e) => set("criteria", { ...c, [k]: e.target.value })} />
                </Field>
              );
            })}
          </div>
        )}
      </div>

      <Field label="What the state contains at the call site">
        <Textarea rows={2} value={value.state_description} onChange={(e) => set("state_description", e.target.value)} />
      </Field>
      <Field label="Example state">
        <Textarea rows={4} className="font-mono text-[12px]" value={value.state_example} onChange={(e) => set("state_example", e.target.value)} />
      </Field>
    </fieldset>
  );
}
