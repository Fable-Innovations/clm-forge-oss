import type { Opportunity, TaskSchema } from "../api/types";

/** Label keys as the backend's TaskSchema.labels() returns them, with their descriptions. */
export function labelEntries(task: TaskSchema): [string, string][] {
  if (task.type === "choice") return Object.entries(task.criteria && !Array.isArray(task.criteria) ? task.criteria : {});
  if (task.type === "score") return (Array.isArray(task.criteria) ? task.criteria : []).map((c, i) => [String(i), c]);
  const c = task.criteria && !Array.isArray(task.criteria) ? task.criteria : {};
  return [
    ["true", c.true ?? "yes"],
    ["false", c.false ?? "no"],
  ];
}

/**
 * A prompt for a coding agent working in the target repo: find where this decision's inputs and
 * real outcomes live, export them read-only from the database, and save them under clm_data/ in the
 * shape the curator's importer reads (see README "Using your own data").
 */
export function dataExportPrompt(task: TaskSchema, opp: Pick<Opportunity, "files" | "target_examples"> | null, repo?: string | null): string {
  const labels = labelEntries(task);
  const keys = labels.map(([k]) => `\`${k}\``).join(", ");
  const target = Math.max(opp?.target_examples ?? 600, 600);
  const perLabel = Math.max(50, Math.ceil(target / Math.max(1, labels.length)));
  const files = (opp?.files ?? []).map((f) => `  - \`${f.path}\`${f.start_line != null ? ` (lines ${f.start_line}${f.end_line != null && f.end_line !== f.start_line ? `-${f.end_line}` : ""})` : ""}`);
  const base = task.name || "examples";

  return `You are preparing real training and test data for a small classifier that will replace an LLM decision in this codebase${repo ? ` (${repo})` : ""}. Find where this decision's inputs and real outcomes are stored, export historical examples from the database, and save them under \`clm_data/\` at the repository root. Every row must come from the database. Do not invent, paraphrase or synthesize rows, and do not modify application code.

## The decision

- Question: ${task.instructions}
- Answer type: ${task.type}
- Labels (use these exact keys):
${labels.map(([k, d]) => `  - \`${k}\`: ${d}`).join("\n")}
- What the classifier sees (the "state"): ${task.state_description}
${files.length ? `- Code that makes this decision today:\n${files.join("\n")}\n` : ""}
Example state, exactly as the classifier will see it:

\`\`\`
${task.state_example}
\`\`\`

## Steps

1. **Find the data.** Read the code above and follow it back to the database. Identify the tables and columns that make up the state, and where the real outcome of this decision is recorded. Prefer outcomes a person confirmed or that were settled later (a reviewer's decision, an override, the final adjudicated status) over the current LLM's own past predictions. If the LLM's predictions are the only label available, export them anyway as a column named \`llm_label\`, not \`label\`, and say so in the README.
2. **Connect read-only.** Use the connection settings the project already uses (environment variables, \`.env\`, a secrets manager, IaC outputs). Run only SELECT queries; never write, update or hold locks. If you cannot find a way to connect, stop and ask me. Do not guess credentials.
3. **Write the export query.** One row per decision, with these columns:
   - \`id\`: the source record's primary key.
   - One column per input field that makes up the state, with raw values (do not pre-format the state text; the importer renders it from the columns).
   - \`label\`: one of ${keys}, mapped from the real outcome.
   - \`raw_outcome\`: the outcome value as stored, so the mapping to \`label\` can be checked.
   - Optional: \`created_at\` and \`label_source\` (for example \`human\`, \`final_status\` or \`llm\`).
   Leave out every other field that is not known at the moment the decision is made: downstream statuses, reviewer notes, payments and anything computed afterwards. These leak the answer.
4. **Volume and balance.** Aim for at least ${target} rows and at most about 20,000, with at least ${perLabel} per label where they exist. Sample stratified by label if the natural distribution is skewed, and prefer recent records. Drop test or demo records, exact duplicates, and rows whose input fields are all empty. Rows whose outcome does not clearly map to a label should be left out, not guessed.
5. **Privacy.** Remove or mask personal data the decision does not need (names, emails, phone numbers, street addresses, account numbers). This text is sent to an LLM provider during curation, and test rows ship inside the trained package.
6. **Save.**
   - \`clm_data/${base}.csv\`: UTF-8, header row, standard CSV quoting. Multi-line text stays inside one quoted cell. Use \`clm_data/${base}.jsonl\` (one flat JSON object per line) instead if that is simpler.
   - \`clm_data/${base}.sql\`: the exact export query.
   - \`clm_data/README.md\`: source tables and joins, filters, date range, rows per label, where \`label\` comes from and how trustworthy it is, and what was masked.
   - Add \`clm_data/\` to \`.gitignore\`. Production data must not be committed. CLM Forge reads the folder even when it is gitignored.
7. **Verify.** Re-open the file and report: the total row count, the count per label, how many rows were dropped and why, and three rows rendered into the state format above so I can compare them with the example.

CLM Forge imports \`clm_data/\` on an experiment's first curation pass. Export before accepting a pending proposal; for an experiment that has already been curated, create a new experiment for this repository.`;
}
