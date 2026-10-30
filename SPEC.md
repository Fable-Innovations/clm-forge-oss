# CLM Forge — architecture contract

CLM Forge is a local app where a user links a repository (or describes a task in plain text) and an AI
agent finds where a **CLM** (Contrastive Language Model: frozen Qwen3-8B encoder + small trained
projection heads, https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) fits best, proposes it, and on
acceptance curates training data, trains a head locally, evaluates it against baselines, and ships a
package + an integration plan a coding agent can execute.

This file is the contract between the three parts (backend ML, backend agent/API, frontend). If you
change an interface here, say so in your final report.

Reference source for CLM (read-only, do not import from it): `vendor/clm-src/` — especially
`src/clm/schema.py`, `src/clm/heads.py`, `src/clm/engine.py`, `train/finetune.py` (`run_choice`),
`train/adapters.py`. Vendored copies we *do* import: `backend/forge/clm_core/{heads,schema}.py`.
Gotcha: `HeadPair._project` evaluates `self.state_head` before `ensure()` loads it — always call
`head_pair.ensure()` before `project_*`.

## Layout

```
clm-forge/
  SPEC.md  README.md
  backend/                 uv project, Python 3.12, package `forge`
    pyproject.toml
    encoders.json          the encoder registry (one table; see "Encoders")
    forge/
      clm_core/            vendored CLM head + schema (Apache-2.0)            [given]
      config.py            Settings (paths, provider defaults, thresholds)     [agent]
      encoders.py          encoder registry: keys, specs, HF-cache check, output roots, used-by      [agent]
      encoder.py           embedder for a spec (mlx / hf / llama.cpp) + disk cache; runs the model in a worker  [ML]
      encoder_backends.py  mlx / hf / llama.cpp backends (worker side, memory caps)                  [ML]
      embed_worker.py      embedding worker subprocess (lock, admission, bucketed batches)          [ML]
      memory.py            memory policy, measurement, buckets, batch sizing, admission, watchdog    [ML]
      model_lock.py        machine-wide one-model-at-a-time lock                                      [ML]
      trainer.py           head training with progress callbacks               [ML]
      evaluator.py         metrics, baselines, calibration, latency            [ML]
      inference.py         load a head + answer/rank (playground + package)    [ML]
      packager.py          build the deliverable package                        [ML]
      runtime_template/    source of the shipped pip package `clm_<task>`      [ML]
      rules.py             project state machine (the guardrail, no AI)        [agent]
      store.py             SQLite persistence                                  [agent]
      events.py            in-process pub/sub -> SSE                           [agent]
      llm/                 providers: claude (Agent SDK, subscription), codex (CLI, ChatGPT subscription), openai, ollama;
                           catalog (live model lists); repo_mcp (repo-confined tools over MCP for codex)  [agent]
      repo.py              clone/locate repo, scan, candidate-site detection   [agent]
      analyst.py           codebase analysis -> Proposal                       [agent]
      curator.py           training-data curation                              [agent]
      shaping.py           real-shaped generation: scrub, exemplars, leak / real-source guards  [agent]
      labelling.py         labelling queue for real rows (rules, undo, dataset upsert, labelling agent)  [agent]
      jobqueue.py          global model-job queue (FIFO, concurrency 1, preflight, persisted)  [agent]
      autopilot/           detection, balance, diagnosis, promotion, run loop, adapter client,
                           scheduler, journal, call groups, presets, validation, run descriptions
                           (see "Autopilot")                                   [agent]
      orchestrator.py      LangGraph loop over rules.py; runs stage jobs       [agent]
      api.py               FastAPI app, REST + SSE, serves frontend/dist        [agent]
    tests/
  frontend/                Vite + React 18 + TypeScript                        [UI]
  data/                    runtime (gitignored): forge.db, projects/<id>/..., cache/
  vendor/clm-src/          reference clone (gitignored)
```

Run: `cd backend && uv run forge-api` (port **8765**), `cd frontend && npm run dev` (port **5173**,
proxies `/api` to 8765 with `changeOrigin: false`, so the browser sees one origin). `uv run forge-api` also serves
`frontend/dist` if built. Neither the server nor the dev server (`server.cors: false`) sends a CORS grant: no page
of another origin may read Forge's responses; the UI is always same-origin.

## Environment (as developed and verified)

- Apple Silicon Mac (developed on 48 GB unified memory), **no CUDA**. Do **not** download
  full-precision Qwen3-8B by default.
- Reference encoder (the CLM reference head's own, `encoders.REFERENCE`; not a default): **MLX**, `mlx-community/Qwen3-8B-4bit` (already in HF cache). Embedding = final
  hidden state after the model's final norm (`model.model(ids)` in mlx_lm returns that), **last token**,
  float32, **L2-normalised**. Tokenise with `tokenizer.encode(text, add_special_tokens=False)`, keep the
  **tail** `max_len-1` tokens (max_len default 2048). Verified: ~1 s first call, 4096-d vector, and the
  reference head `CLM_v0.1-8B.pt` answers correctly zero-shot on these embeddings.
  Optional backend `transformers` (bf16, MPS), spec `hf:Qwen/Qwen3-8B` — not in the registry.
- Encoder registry: `backend/encoders.json` (see "Encoders"). Every encoder a project or target trains on is a
  registry key it picked itself; there is no app-wide or process-wide default (`FORGE_ENCODER` and the Settings
  encoder are gone).
- Reference head: `forge.clm_core.heads.download()` → `~/.cache/clm/CLM_v0.1-8B.pt` (75 MB,
  cfg width 1536, depth 3, gelu, layernorm, proj 512, logit_scale exp = 100).
- Claude: `claude-agent-sdk` (0.2.x) using the user's **Claude subscription login** (the local `claude`
  CLI's auth). Strip `ANTHROPIC_API_KEY` from the subprocess env when provider = `claude-subscription`.
  `ClaudeAgentOptions(output_format={"type":"json_schema","schema":...})` → `ResultMessage.structured_output`
  is verified working. Default model `claude-sonnet-5-5`; allow `claude-opus-5-5`.
- Ollama (optional) at `http://127.0.0.1:11434`.
- uv Pythons: 3.12 and 3.13 available. Use 3.12.

## Domain model (Pydantic, in `forge/models.py` — owned by [agent], imported by [ML])

```python
QuestionType = Literal["noul", "choice", "score"]

class TaskSchema(BaseModel):          # the typed question the CLM will answer
    name: str                          # slug, e.g. "ticket_router"; becomes package name clm_<name>
    question_id: str                   # key in the System One `questions` map, e.g. "department"
    type: QuestionType
    instructions: str                  # the question text (appended after the state)
    criteria: dict[str, str] | list[str] | None   # choice: {key: description}; score: ordered levels; noul: optional {"true":..,"false":..}
    state_description: str             # what the state text contains at the call site
    state_example: str

class FileRef(BaseModel):
    path: str; start_line: int | None = None; end_line: int | None = None; snippet: str | None = None

class Opportunity(BaseModel):
    id: str
    title: str
    files: list[FileRef]
    current_approach: str              # what the code does today (LLM call / regex / heuristic / none)
    task: TaskSchema
    fit_score: float                   # 0..1
    rationale: str
    expected_benefits: list[str]       # e.g. "p50 latency ~1.2s -> ~40ms", "no per-call API cost"
    risks: list[str]
    data_strategy: str                 # where examples come from
    target_examples: int               # how many to curate (default 600, cap 3000)
    success_metric: Literal["accuracy", "macro_f1"]
    success_threshold: float           # absolute floor on the test split
    integration_sketch: str

class Proposal(BaseModel):
    verdict: Literal["fit", "no_fit"]
    summary: str
    opportunities: list[Opportunity]   # ranked best first; empty when no_fit
    recommended_id: str | None
    no_fit_reasons: list[str]
    revision: int = 0

class Example(BaseModel):              # one labelled row, System One "typed decisions" format
    id: str
    state: str                         # state text WITHOUT the question (schema.state_text appends instructions)
    label: str                         # choice key | "true"/"false" | score level index as str
    probabilities: dict[str, float] | None = None   # optional soft target
    source: Literal["imported", "mined", "synthetic", "hard_negative", "user"]
    split: Literal["train", "val", "test"]
    rationale: str | None = None       # labeller's reason
    meta: dict = {}
    label_source: Literal["user", "teacher", "agent", "export"] | None = None
        # who labelled it; None on a synthetic/hard-negative row = the generator's label, confirmed by the teacher.
        # Rows written before this field existed are backfilled on read (no file is rewritten):
        # meta.label_origin teacher|export -> same; source user or meta.edited -> user;
        # mined + "label stated in source" -> export; any other real row -> teacher.
    provenance: str | None = None      # "real_shaped_synthetic": generated on scrubbed real exemplars; train-only;
                                       # meta.exemplar_hashes, meta.batch say which

REAL_SOURCES = ("imported", "mined", "user")   # an Example is real (not generated) when its source is one of these
REAL_SHAPED = "real_shaped_synthetic"

class RealRow(BaseModel):              # one real row in a labelling queue (forge/labelling.py)
    rid: str                           # = the Example id once labelled
    state: str
    question_meta: dict = {}           # {"question": {...}} when the row is asked its own question; provenance
    source: Literal["imported", "mined", "user", "export"]   # export = a row a head-set adapter ingested from a real-data source
    split: Literal["train", "val", "test"] = "train"
    label: str | None = None
    label_state: Literal["needs_label", "labelled", "excluded_by_agent"] = "needs_label"
    label_source: Literal["user", "teacher", "agent", "export"] | None = None
    suggested_label: str | None = None # the teacher's view when it may not label the row
    teacher_conf: float | None = None
    teacher_votes: list[dict] = []     # two-pass teacher: [{pass, label, conf}]
    head: str | None = None            # task name (project) or head id (head-set head dir)
    skipped: bool = False              # passed over for now; still pending
    excluded: bool = False             # the user went on without it; not pending until someone labels it
    adjudicated: bool = False          # the agent's label came from the adjudicator (the passes disagreed)
    exclusion_reason: str | None = None  # why the agent excluded it (label_state == excluded_by_agent)
    updated_at: str = ""
```

A real row's **final disposition** is `labelled`, `excluded_by_agent`, or set aside by the user (`excluded`);
`pending` (= undisposed) is `label_state == needs_label and not excluded`.

Dataset file: `data/projects/<pid>/dataset.jsonl` (one `Example` per line) + `task.json` (`TaskSchema`).
The question asked of every row is the project's `TaskSchema` — one question per project.

## Labelling real rows (`forge/labelling.py`)

**The standard: a real row without a label is never trained on until it has a label, and by default the
labelling agent gives it one (or excludes it with a reason); the user can opt into reviewing them and can
override any label.** Synthetic rows are labelled by their generator and checked by the teacher, as before;
this section is only about real rows (imported, mined, user, adapter export).

A label set is one directory with a queue:

- `real_rows.jsonl`: the current state of every real row (`RealRow`).
- `labelling_log.jsonl`: append-only actions `{seq, ts, rid, action (add|label|skip|teacher|agent|exclude|undo),
  actor (user|teacher|agent|system), prev, next, undo_of?, via?}`; `undo` reverts the latest user `label`/`skip` not yet
  undone. `via` (user actions made through the API): `{origin, user_agent}` of the browser session (`web_guard`).

Label set ids: `p:<pid>` = a project (`data/projects/<pid>/`); `h:<head_id>` = a head-set head directory
`<root>/<head_id>/` (`task.json` + `real_rows.jsonl`) under a root listed in `Settings.labelset_roots` or the
`FORGE_LABELSETS` environment variable (os.pathsep-separated).

Rules, enforced only in `LabelQueue`:

- A user label is final for the teacher and the agent: `apply_teacher` / `apply_agent` never touch a row whose
  `label_source` is `user`.
- An export label is ground truth: the teacher only fills `suggested_label` / `teacher_conf` on it; an agent
  answer that contradicts it is counted (`export_contradicted`, journalled by the autopilot) and never applied.
- A teacher result in `suggest` mode (the `mixed` policy) or a two-pass disagreement never labels a row.
- In a project, a row that becomes `labelled` is upserted into `dataset.jsonl` (same id, its `label_source`,
  its queue split); a row that goes back to `needs_label` (undo) is removed from it. `needs_label` rows are
  never in `dataset.jsonl`.
- `exclude` marks the pending rows `excluded` (set aside by the user): they stop blocking the review gate, stay
  in the queue and join the dataset when someone labels them. `excluded_by_agent` rows never join the dataset
  unless a user labels them.

### Labelling agent (`labelling.agent_label`, the `agent` policy)

1. The teacher answers every undisposed row (one vote per pass: a single pass for Forge projects; a head-set
   adapter's own labelling may ask two passes). A row with no vote at all is a teacher **failure**, not a disagreement: it
   is asked again, up to 2 more times.
2. Agreeing votes label the row: `label_source=agent`.
3. Disagreeing votes go to the **adjudicator** (a stronger teacher, configurable), which sees both passes'
   labels and rationales: a pass label -> labelled, `adjudicated=true`; an abstention or a third answer ->
   `excluded_by_agent` with the reason. A vote outside the row's options excludes the row likewise.
4. Rows the teacher still never answered stay `needs_label` and the job raises `AgentLabelError` after every
   other row was written: a failed teacher is never turned into a label or an exclusion. In a project that
   fails the `labelling` stage visibly; `retry` asks only for the rows still undisposed.

All writes go through `LabelQueue` (its lock); `labelling_log.jsonl` records them with actor `agent`.

**Locking.** `LabelQueue` and a head-set adapter (its ingest, labelling agent and teacher) may edit the same
`real_rows.jsonl` in a head directory. Every read-modify-write on either side holds an `fcntl.flock` on
`<dir>/.real_rows.lock` (in Forge, `forge.filelock.real_rows_lock(dir)`, which adds a process-local re-entrant
lock; an adapter written in Python can use it directly) and re-reads the file inside the lock, so a user label
made while an adapter commit runs waits and survives, and neither side overwrites the other's rows.
`RealRow` keeps keys it does not name (`extra="allow"`; `adjudication` = `{model, decision, reason}` is named),
so a queue rewrite never drops what the adapter wrote.

Label policy, per project (`rec.label_policy`; set at accept or with `PUT .../policy`, changeable later; a
project that never set one uses `agent`):

| policy | during curation | in the `labelling` stage | human gate |
|---|---|---|---|
| `agent` (default) | queued as `needs_label`; no teacher call | the labelling agent disposes every row | never |
| `user` | queued as `needs_label`; no teacher call | nothing | `label_review` while rows are undisposed |
| `teacher` | teacher-labelled (`label_source=teacher`) into the dataset, mirrored into the queue for review | the teacher labels what is still pending | `label_review` for rows it could not label |
| `mixed` | the teacher fills `suggested_label` + `teacher_conf`; the row stays `needs_label` | suggestions for new rows | `label_review` |

A user label recorded with `label_override` (any non-terminal state, `POST .../labels/override`) is applied
at the start of the next `labelling` stage, before the agent runs; while no stage locks the queue the user can
also label directly (`POST /labelsets/{lsid}/rows/{rid}`).

Head-set queues (`h:`) are filled and labelled by the head set's adapter (`ingest`, and the labelling agent
behind its `label` verb), which trains on their `labelled` rows as train-only data. The API never sends a head
queue's rows to an LLM. A head the adapter reports `retired` (`inspect`) is skipped by the autopilot (journalled
`skip`, outcome `retired`) and never fails a run.

## Rules layer (`forge/rules.py`, python-statemachine) — the guardrail

States: `created`(initial) → `analyzing` → `proposed` → `accepted` → `curating` → `labelling` →
`selecting` → `training` → `evaluating` → `reviewing` → `packaging` → `delivered`(final). Also
`label_review` (the opt-in human labelling gate), `no_fit`(final), `cancelled`(final), `errored`.
`labelling` is a system stage: its job applies recorded overrides, runs the label policy (the labelling agent
by default) and settles. `proposed` and `label_review` are human gates: a server restart leaves them as they
are; every other non-final state's job is queued again from its start (not failed to `errored`).

| event | from → to | actor | guard |
|---|---|---|---|
| start | created → analyzing | system | |
| propose | analyzing → proposed | **agent** | proposal.verdict == fit |
| declare_no_fit | analyzing, proposed → no_fit | **agent** | proposal.verdict == no_fit |
| revise | proposed → analyzing | **human** | feedback non-empty |
| accept | proposed → accepted | **human** only | opportunity chosen |
| begin_curation | accepted → curating | system | |
| curated | curating → labelling | system | |
| labels_settled | labelling → selecting | system | `undisposed_rows` == 0 (every real row labelled, excluded by the agent or set aside by the user), dataset ≥ min_examples (default 120) and every label seen in train |
| needs_review | labelling → label_review | system | `label_policy` != `agent` and `undisposed_rows` > 0 |
| review_done | label_review → labelling | **human** only | `mode=continue` with no undisposed rows, or `mode=exclude` (sets the rest aside; the reason records how many) |
| label_override | any non-terminal state → itself | **human** only | none; records `{rid, label}`, applied by the next `labelling` stage; starts no job |
| selected | selecting → training | system | a base model was chosen, and it is the experiment's own encoder (`base_model`) |
| trained | training → evaluating | system | checkpoint exists |
| evaluated | evaluating → reviewing | system | report exists |
| ship | reviewing → packaging | **agent** | `gate_passed` (deterministic, see below) |
| recurate | reviewing → curating | **agent** | attempts < max_attempts (default 3) |
| give_up | reviewing → no_fit | **agent** | attempts ≥ max_attempts or agent judges hopeless |
| packaged | packaging → delivered | system | manifest exists |
| fail | any non-final except label_review → errored | system | |
| retry | errored → (state it failed in) | **human** | |
| cancel | any non-final → cancelled | **human** | |

`Facts` for the labelling guards: `undisposed_rows`, `excluded_by_agent`, `label_policy`.
Labelling, relabelling and "send to teacher" inside `label_review` change the queue and the dataset, never the
state. Dataset and queue edits are refused (409) while `curating`, `labelling`, `selecting`, `training` or
`evaluating` (record a `label_override` instead). The teacher job is an override helper of the review gate:
`POST /labelsets/p:<pid>/teacher` is refused (409) in any other state, `review_done` is refused (409) while the
job runs, any transition out of `label_review` cancels it, and a job that finds the gate left drops its
result instead of writing the dataset.

`gate_passed` = test `success_metric` ≥ `success_threshold` **and** fine-tuned beats zero-shot
reference head **and** beats majority-class baseline. The agent can never `accept` (human-only events
are not offered to it) and can never ship a failed gate (guard rejects; logged as `REJECTED`).
Machine keeps `path` [(src, event, dst)] and `log`; every transition stores actor + reason.

## Event stream (SSE `GET /api/projects/{pid}/events`, `text/event-stream`, each `data:` is JSON)

On connect the server first replays the persisted history for that project (so a reload rebuilds the
UI), then streams live. Every event: `{"type", "ts" (ISO), "seq" (int, monotonic per project), ...}`.

| type | payload |
|---|---|
| `state` | `from, event, to, actor ("agent"\|"human"\|"system"), reason` |
| `rejected` | `event, actor, reason, error` |
| `log` | `stage, level ("info"\|"warn"\|"error"), message` |
| `chat` | `role ("user"\|"assistant"), content, id` ; `chat_delta` `{id, delta}` for streaming tokens (optional) |
| `tool` | `name, input (short str), output (short str)` — agent exploration steps (file reads, greps) |
| `proposal` | `proposal: Proposal` |
| `curation` | `phase ("importing"\|"mining"\|"generating"\|"reviewing"\|"labelling"\|"dedup"\|"splitting"\|"done"), done, total, counts: {by_label:{}, by_source:{}, by_split:{}, needs_label}, workers?: {label: {done, total, generated}}` — `needs_label` = real rows waiting in the queue |
| `labels` | `counts: {needs_label, user, teacher, agent, adjudicated, export, excluded, excluded_by_agent, suggested, total}` — the queue changed outside a request (labelling stage, teacher job, review) |
| `autopilot` | `action, reason, counts, head, round, target, run` — a journal entry of an autopilot target that drives this project |
| `embed` | `done, total` (encoding progress, any stage) |
| `base_model` | `candidates: [{spec, name, detail?, url?, status ("pending"\|"running"\|"done"\|"failed"), load_s?, embed_ms?, val_acc?, val_soft_ce?, init_val_acc?, best_epoch?, dim?, seconds?, error?, chosen?}], chosen: spec\|null` — base-model selection progress; `chosen` set on the final event |
| `train_epoch` | `epoch, epochs, train_loss, val_acc, val_soft_ce, lr, logit_scale` |
| `train_step` | `epoch, step, steps, loss` (throttled ≤ 5/s) |
| `train_snapshot` | `epoch, similarity: {row_labels: [str], col_labels: [str], values: [[float]]}, points: [{x, y, label, split, kind ("state"\|"option"), text}]` — sim matrix = mean projected cosine of val states (rows grouped by gold label) vs options; points = 2-D PCA of projected val states + options (≤ 400 points). Emitted at epoch 0 (init) and each epoch. |
| `eval` | `report: EvalReport` |
| `package` | `manifest: PackageManifest` |
| `error` | `stage, message, traceback (str)` |

## REST (`/api`)

Request guard (`forge/web_guard.py`, installed by `create_app`; the server binds 127.0.0.1 only): every request
must be addressed to Forge (`Host` 127.0.0.1 / localhost on the server port or the UI dev port); an `/api` request
the browser marks `Sec-Fetch-Site` other than `same-origin` / `none` is refused; POST / PUT / PATCH / DELETE need
`Origin` = the Forge UI origin and `X-Forge-Session` = the per-server token from `GET /api/session` (random per
start, never stored or logged; the page fetches it once and again after a 403 naming the session). Each refusal is
a 403 whose `detail` says in plain words what was missing. No other page may frame Forge (clickjacking: a framed
Forge page's own scripts hold the session): every response carries `X-Frame-Options: DENY` and
`Content-Security-Policy: frame-ancestors 'none'`, and a non-`/api` request the browser marks `Sec-Fetch-Dest`
`iframe` / `frame` / `embed` / `object` with `Sec-Fetch-Site` other than `same-origin` is refused. No
`Access-Control-Allow-*` header is ever sent (no CORS middleware). User label actions record `via: {origin,
user_agent}` in `labelling_log.jsonl`; undo skips an entry `redacted` by a deletion at the source.

```
GET    /api/session                     -> {token, header: "X-Forge-Session"}  (same Host / Sec-Fetch-Site rules; Cache-Control no-store)
GET    /api/health                      -> {ok, encoders: [{key, name, backend, cached, model, loaded, worker, error?, used_by}], unregistered_encoders: [spec],
                                           providers: {claude: bool, openai: bool, ollama: bool, ollama_models: [str]}}
                                           (encoders = every registry encoder an experiment or a target trains on; none is loaded to report it)
GET    /api/providers[?refresh=true]    -> [{id, label, available, detail, models: [{id, label, description?}]}]  (live per provider, cached 10 min)
GET    /api/settings / PUT              -> Settings (provider, model, ollama_url, openai_base_url, thresholds, max_attempts; no encoder field;
                                           openai_api_key is write-only: PUT a value to set, null to clear; GET returns openai_api_key_set;
                                           PUT naming an app-wide encoder field (encoder, default_encoder, base_models, ...): 422)
GET    /api/encoders                    -> {encoders: [{key, name, backend, repo, file, revision, hidden, quant,
                                           approx_gb, approx_gb_source, note, spec, url, cached, used_by: {projects: [{id, name}], targets: [{id, name}]}}]}
GET    /api/encoders/recommendation     -> {key, name, profile, reason, budget_gb, considered: [{key, name, approx_gb, fits, cached, why_not}]}
                                           the encoder Forge recommends for a new experiment (forge.encoder_choice; key null when none fits)
GET    /api/projects                    -> [ProjectSummary]
POST   /api/projects                    {name?, repo?: str (https git URL or absolute local path), task?: str, provider?, model?, encoder: registry key (required)}
                                           -> ProjectSummary (starts analysis); 422 without a registered encoder
GET    /api/projects/{pid}              -> ProjectDetail
PUT    /api/projects/{pid}/encoder      {key} -> ProjectDetail   sets base_model (encoder_source from user), emits encoder_changed {from_name, to_name},
                                           journals encoder_changed on the project's autopilot targets (409 while a stage job or the teacher runs, 422 unknown key)
DELETE /api/projects/{pid}
GET    /api/projects/{pid}/events       SSE
POST   /api/projects/{pid}/chat         {message} -> 202   (agent replies via events; in `proposed` it may revise the proposal)
POST   /api/projects/{pid}/accept       {opportunity_id, overrides?: {task?: TaskSchema, target_examples?, success_threshold?}} -> 202
POST   /api/projects/{pid}/revise       {feedback} -> 202
POST   /api/projects/{pid}/cancel | /retry -> 202
GET    /api/projects/{pid}/dataset?split=&label=&source=&q=&offset=&limit= -> {total, items: [Example], counts}
PATCH  /api/projects/{pid}/dataset/{eid}   {label?, split?, state?}   (not while curating/selecting/training/evaluating; sets source=user, label_source=user, and the queue row's user label)
DELETE /api/projects/{pid}/dataset/{eid}
GET    /api/projects/{pid}/training     -> {history: [train_epoch events], snapshots: [train_snapshot events]}
GET    /api/projects/{pid}/eval         -> EvalReport | 404
POST   /api/projects/{pid}/infer        {state, model?: "finetuned"|"reference"} -> {answer (System One Answer dict), latency_ms, embed_ms, head_ms}
POST   /api/projects/{pid}/rank         {state, candidates: [str], instructions?} -> [{rank, candidate, prob}]
GET    /api/projects/{pid}/package      -> PackageManifest | 404
GET    /api/projects/{pid}/package/download   -> zip
GET    /api/projects/{pid}/package/file?path= -> text (for INTEGRATION.md etc.)
POST   /api/playground/infer            {state, task: TaskSchema} -> same as /infer, reference head only (no project)
GET    /api/labelsets                   -> [{id, kind ("project"|"head"), name, counts, state?, policy?, dir?}]  projects with a queue + registered head dirs with real_rows.jsonl
GET    /api/labelsets/{lsid}/queue?head=&state=needs_label|teacher|user|export|labelled|all&offset=&limit=
                                        -> {id, name, kind, total, items: [RealRow + options: [{key, description}]], counts, task, policy, state, editable}
POST   /api/labelsets/{lsid}/rows/{rid} {label} | {skip: true} -> {row, counts}   (actor is always the user; 404 unknown row, 422 label not an option, 409 locked)
POST   /api/labelsets/{lsid}/undo       -> {row | null, counts}   reverts the latest user action
PUT    /api/labelsets/{lsid}/policy     {policy: agent|user|teacher|mixed}   (projects only)
POST   /api/labelsets/{lsid}/teacher    {mode?: label|suggest} -> 202 {rows}   background teacher job over the pending rows (projects in `label_review` only, else 409; head queues: 409, the adapter's `label` verb labels them)
POST   /api/projects/{pid}/labels/review {mode: continue|exclude} -> 202   fires review_done as human (409 while a teacher job runs)
POST   /api/projects/{pid}/labels/override {rid, label} -> 202 {pending: [{rid, label}]}   fires label_override as human (404 unknown row, 422 not an option)

GET    /api/autopilot/targets           -> [TargetView]   TargetView = Target + {encoder_name, output_roots: {encoder key: folder} (head sets), schedule, last_run: RunView,
                                           pending_actions: [{id, action, approved, reason, queued, last_error}]}
GET    /api/autopilot/targets/{id}      -> TargetView
POST   /api/autopilot/targets           Target (id optional: derived from name) -> 201 TargetView   (409 the id exists; journals target_created)
PUT    /api/autopilot/targets/{id}      Target (without id) -> TargetView   full replace (creates when missing); journals encoder_changed / target_edited
PATCH  /api/autopilot/targets/{id}/enabled {enabled} -> TargetView   journals target_enabled / target_disabled
DELETE /api/autopilot/targets/{id}      runs and journal stay; their views say target_removed
GET    /api/autopilot/headsets          -> [{key, name, description, dataset_root, output_base, protected_roots, sources: [{kind, label, enabled_by_default,
                                           settings: {name: {values, default, multiple}}}], fixed_heads, encoders, output_roots,
                                           recommendation: {key, name, reason}}]   (the encoder a new target on it starts on)
GET    /api/autopilot/headsets/{preset}/heads?encoder= -> [{head, group, retired}]   the preset adapter's read-only inspect; the output folder is not checked (inspect writes nothing). 404 preset, 422 unknown encoder (its `detail` is plain words, no field path), 502 adapter failed
PUT    /api/autopilot/targets/{id}/schedule {cadence: daily|weekly, weekday?: 0..6 (Monday=0), time_local: HH:MM, enabled} -> Schedule
POST   /api/autopilot/targets/{id}/run  -> 202 Run   (a target with a queued or running run gets that run; 409 disabled)
GET    /api/autopilot/runs?target=      -> [RunView]
GET    /api/autopilot/runs/{id}         -> RunView = Run + {progress, timeline, counts, results, encoder (name), target_name, target_removed}
GET    /api/autopilot/journal?target=&head=&action=&run=&after=&limit= -> [JournalEntry] newest first
GET    /api/autopilot/queue             -> {active: [ModelJob] in queue order, recent: [ModelJob], autopilot_current}
GET    /api/autopilot/groups            -> [{target, run, computed_at, report: GroupsReport | null}]
GET    /api/autopilot/heads/{head}/provenance?target= -> {target, path, provenance}   (404 none, 422 bad head id)
```

`lsid` = `p:<pid>` | `h:<head_id>`. `POST /accept` also takes `overrides.label_policy`.

Every autopilot target route answers an invalid body with **422 `{errors: [{field, message}], detail}`**:
`field` is dotted (`headset.output_root`, `headset.policy.min_count`, `headset.sources.0.settings.stage`,
`project.project_id`, `id`), `message` is plain words ("Output folder is inside a protected folder (…)", "must be
at least 1"; pydantic prefixes stripped, `forge.autopilot.validation`), `detail` joins them.

Run fields (written by `runner.py`): `first_started` (set once; `started` is when this execution began),
`result.plan = {heads, phases, started_at}` (written once, after the first `inspect`), `result.timings =
{step key: {s, e}}` (`e` null = running), per head `result.heads[h].started / ended`, `result.encoder` (registry
key). Derived (`describe.py`): `progress = {fraction, heads_done, heads_total, current: {label, head}, detail,
elapsed_s, eta_s, eta_note} | null` (null: the run predates plans), `timeline = [{key, label, head, phase, state
done|running|pending, started, ended}]`, `counts = {heads: {h: {generated, labelled, excluded}}, total}`,
`results = [{head, control_f1, val_f1, final_f1, reference_f1, incumbent_f1, gate_passed, outcome, decision, rounds}]`.

`ProjectSummary = {id, name, state, created_at, updated_at, repo, task, provider, model, verdict?, metric?}`
`ProjectDetail = ProjectSummary + {path: [[src,event,dst]], allowed_human_events: [str], proposal?, accepted_opportunity_id?, task_schema?, base_model (spec), encoder: {key, name, spec, source: {from: create|user|checkpoint|settings, detail?, at, migration?}} | null, encoder_recommendations: {opportunity id: {key, name, reason, profile}}, attempts, chat: [...], label_policy (agent when never set), label_overrides: [{rid, label}], labels: LabelCounts}`

## Encoders (`backend/encoders.json`, `forge/encoders.py`)

One table, read as a file by `forge.encoders` (and by any head-set adapter that wants the same pins: it reads the
file, it never imports `forge`). Entry: `key, name, backend (mlx|llamacpp), repo,
file (GGUF only), revision, hidden, quant, approx_gb (largest measured embedding-worker peak; null when never
measured, never guessed), approx_gb_source, note`. `spec` = `mlx:<repo>` | `llamacpp:<repo>:<file>`.
`cached(enc)` = the pinned snapshot (and GGUF file) is under `$HF_HOME/hub` (no load). `output_root(base, key)` =
`<base>-<key>` for every key: a preset's base is typically the folder of the heads the head set already ships, a
protected root, so a derived folder is never it. `used_by` = projects by `base_model`, targets by
`headset.encoder` / their project's `base_model`. Entries: `qwen3-4b`, `qwen3-0.6b`, `qwen3-0.6b-8bit`,
`qwen3-0.6b-bf16`, `qwen3-4b-gguf-q4km`, `qwen3-8b-mlx-4bit`, `winnow-12b-gguf-q8`.

**No app-wide encoder.** Each experiment owns its encoder (`base_model`, a spec of a registry
entry) and each head-set target its own (`headset.encoder`, a key); Settings hold none. A new experiment's encoder is
a required registry key on `POST /api/projects`; the create dialog preselects `GET /api/encoders/recommendation`, the
proposal recommends one per opportunity (`ProjectDetail.encoder_recommendations`), and `PUT
/api/projects/{id}/encoder` changes it until a stage job runs. A new head-set target starts on its preset's
recommendation (`Preset.recommendation()`), the user can pick another; the encoder is required.

**Recommendation** (`forge/encoder_choice.py`, `recommend(task?, backends?, preferred?)`): eligible = a measured peak
(`approx_gb`, never guessed) that fits this machine's model-job budget (`Settings.memory` resolved) on the allowed
backends; an unmeasured encoder is listed, never recommended. A preset's own encoder (`Preset.recommended_encoder`,
the encoder its heads are built for) wins when eligible. Else by task profile: `rich` (a choice among > 4 options, per-call
options, or a score) → the widest that fits, the reference head's encoder first among equals (warm start); `simple`
(yes/no, ≤ 4 fixed options) and `general` (no task yet) → the narrowest at least 2560-d, the GGUF build first (the
one packages ship on), then a downloaded one, then the smaller peak. The result carries the reason in words and
every encoder considered with why not.

**Which spec a caller embeds with** (`get_encoder(spec)` has no default): curation dedup, selection and training use
the experiment's `base_model` (`Orchestrator.encoder_spec`; `NoEncoder` when it has none); serving, inference and
System One use the checkpoint's `embed_model` (`encoders.head_spec`; a checkpoint without one raises
`CheckpointEncoderMissing`, never a default); the reference head pairs with `encoders.REFERENCE` (its 4096-d input);
an autopilot target trains on its own key. Health reports every encoder in use. Base-model selection's candidates
are the experiment's own encoder (one: the stage records it); the `base_models` setting is gone.

**Migration** (`forge/encoder_migration.py`, once per database, at server start, before experiments are restored):
an experiment without `base_model` gets the encoder its own `best_head.pt` was trained on (`embed_model`), else the
app-wide one the settings named at migration time (`default_encoder`, else the older `encoder` spec), recorded as
`encoder_source {from: checkpoint|settings, detail, at, migration}` and a `log` event (stage `migration`); with
neither, it keeps none and its stream says to pick one. Every CLM checkpoint under an experiment's folder that lacks
`embed_model` gets the experiment's written into its `cfg` once (a base-model selection probe gets the encoder its
`selection/<name>` folder names, or is listed as skipped). The stale `encoder_pinned` flag is dropped. The app-wide
fields (`encoder`, `default_encoder`, `unregistered_encoder`, `base_models`, `unregistered_base_models`) are
stripped from `settings.json` on load; the original is kept as `settings.json.bak-pre-experiment-encoders` (the
migration reads it there).

## ML interfaces ([ML] implements, [agent] calls)

```python
# encoder.py
class Encoder:                                   # one per spec via get_encoder(spec); the spec is always the caller's
    spec: str; backend: str ("mlx"|"hf"|"llamacpp"); model_name: str; dim: int | None  # width known after first vector
    def embed(self, texts: list[str], on_progress: Callable[[int,int],None] | None = None,
              use_cache: bool = True, on_status: Callable[[str,str],None] | None = None) -> np.ndarray  # [n,dim] float32 L2-normed; disk cache keyed by sha1(model|max_len|text) in data/cache/emb/
    def embed_to_file(self, texts, path, on_progress=None, on_status=None) -> np.memmap  # streamed per batch to .npy
    def load(self, on_status=None) -> None; def loaded(self) -> bool; def unload(self) -> None
    # the model runs in a `python -m forge.embed_worker` subprocess under Settings.memory (see README "Memory");
    # raises memory.MemoryAdmissionError / memory.MemoryBudgetExceeded / model_lock.ModelLockTimeout
def get_encoder(spec: str) -> Encoder            # resolve specs through forge.encoders (see "Encoders")

# base_models.py
def select_base_model(task, rows, out_dir, candidates: list[str], emit, should_stop) -> {"chosen": spec, "candidates": [...]}
    # candidates = the experiment's own encoder (required; one candidate is recorded as chosen); given several, a
    # full head-training run per candidate on train, scored on val; writes base_model.json

# trainer.py
@dataclass
class TrainConfig: epochs: int = 30; batch: int = 64; lr: float = 5e-4; weight_decay: float = 0.01;
                   patience: int = 6; targets: Literal["soft","hard"] = "soft"; init: Literal["reference","scratch"] = "reference";
                   seed: int = 1234; device: str = "mps" if available else "cpu"
def train(task: TaskSchema, examples: list[Example], out_dir: Path, cfg: TrainConfig,
          emit: Callable[[dict], None], should_stop: Callable[[], bool] = lambda: False) -> TrainResult
#   emits train_epoch / train_step / train_snapshot / embed / log dicts (without ts/seq; [agent] stamps them)
#   writes out_dir/best_head.pt (same checkpoint format as CLM: state_head, action_head, logit_scale, cfg, epoch, metrics), history.json
#   loss: bidirectional in-batch InfoNCE over distinct option texts (port of run_choice infonce_loss); warm start from reference head

# evaluator.py
def evaluate(task, examples, ckpt: Path, out_dir: Path, emit, teacher: Callable[[list[str]], list[str]] | None = None) -> EvalReport
#   scores the TEST split only; raises ValueError when there is no test split (no val fallback)
class EvalReport(BaseModel):
    metric: str; threshold: float; gate_passed: bool; gate_reasons: list[str]
    models: list[ModelScore]          # "finetuned", "reference" (zero-shot CLM), "majority", optional "teacher" (the LLM on a ≤50-example test sample)
    confusion: {labels: [str], matrix: [[int]]}             # finetuned, test split
    reliability: [{bin_lo, bin_hi, confidence, accuracy, count}]   # 10 bins, finetuned
    ece: float
    per_label: [{label, precision, recall, f1, support}]
    latency: {embed_ms_p50, embed_ms_p95, head_ms_p50, head_ms_p95, warm_cached_ms_p50, n, encoder_peak_bytes}
                                      # 5 untimed warm-up calls, then n=60 timed single-state calls
    errors: [{id, state, gold, pred, probabilities}]  # ≤ 25 worst test errors
    n_test: int
    val: {n, accuracy, macro_f1, confusion, errors} | None   # finetuned on val (None without val rows)
class ModelScore(BaseModel): name: str; accuracy: float; macro_f1: float; n: int; note: str | None = None

# inference.py
class Predictor:  # loads ckpt (or reference), caches option projections
    def __init__(self, ckpt: Path | None, task: TaskSchema)
    def answer(self, state: str) -> tuple[dict, dict]    # (System One answer dict via clm_core.schema, timings)
    def rank(self, state, candidates, instructions=None) -> list[dict]

# packager.py
def build_package(project_dir: Path, task, proposal_opportunity: Opportunity, report: EvalReport,
                  repo_summary: str, emit) -> PackageManifest
class PackageManifest(BaseModel): name: str; version: str; dir: str; zip: str; files: [{path, bytes, description}]; integration_md: str (relative path); created_at
```

### Package contents (`data/projects/<pid>/package/clm_<name>/`)

- `clm_<name>/` pip-installable (pyproject, hatchling), import `from clm_<name> import classify` (or
  `judge`/`score` by type) returning `{label, probabilities, confidence}`; ships `head.pt`, `task.json`;
  embeds via a pluggable encoder (`mlx` default on Apple Silicon, `openai`-compatible `/v1/embeddings`
  URL e.g. vLLM on a GPU box, documented); includes `serve` CLI (FastAPI, System One-compatible
  `POST /v1/systemone`) and a `fallback` hook (confidence < τ → caller's original path).
- built wheel in `dist/`
- `EVAL_REPORT.md` (+ `eval_report.json`)
- `INTEGRATION.md` — written for a coding agent: goal, exact files/lines to change (from the
  opportunity), install steps, code snippet at the call site, confidence-threshold fallback to the
  existing path, feature flag, tests to add (incl. golden examples from the test split), rollout and
  rollback, how to retrain. Generated from a template + LLM polish.
- `examples/golden.jsonl` (test split)

## Agent loop ([agent])

- LangGraph `decide → act → route` over `rules.py`; the LLM only proposes events among the machine's
  currently allowed **agent** events, with a required `reason`; illegal proposals logged as `rejected`.
- Stage work (curation, training, evaluation, packaging) are deterministic jobs started on entering a
  state; they emit events and fire the `system` completion event.
- Decision points for the LLM: in `analyzing` (propose / declare_no_fit — after running the analyst),
  in `reviewing` (ship / recurate / give_up — sees the EvalReport scores + VAL confusions and error
  examples, never test rows, because its guidance becomes training data; on recurate it also
  writes curation guidance, e.g. "more hard negatives between billing and refunds").
- Human gates: accept, revise, cancel, retry, review_done (opt-in policies only); label_override is recorded
  in any non-terminal state. Chat is free-form and available in every state; in
  `proposed` the assistant may emit a revised Proposal (revision+1) in response to chat.
- Providers: `claude-subscription` (Agent SDK; analysis runs as an agentic session whose only
  tools are forge's repo-confined `list_files`/`grep`/`read_file`, served in-process over MCP; Claude
  Code's built-ins are off (`tools=[]`, since its `Read` accepts absolute paths), a PreToolUse hook
  denies any other tool before it runs, and the cwd is an empty temp dir), `openai`
  (Chat Completions; analysis is a bounded tool loop over read-only `list_files`/`grep`/`read_file`
  executed in Python and confined by `repo.safe_path`; any OpenAI-compatible base URL), `codex-subscription`
  (`codex exec` on the ChatGPT login with shell and outward tools disabled, reading code only via the
  same confined tools served by `forge.llm.repo_mcp`) and `ollama`
  (no tool use: we build a repo digest — tree, README, candidate sites found by `repo.py` greps for
  LLM calls / classifiers / regex routers / if-elif ladders over strings — and ask for the structured
  Proposal). All return the same Pydantic models. Model ids come from each provider's live listing.
- Tracing: MLflow to `data/mlflow.db` (opt-in, `FORGE_MLFLOW=1`).

## Curation pipeline ([agent], `curator.py`)

0. **Import** (`importer.py`) real rows from exported tables: `clm_data/*.{csv,tsv,jsonl}` (read even
   when gitignored) plus tracked tables elsewhere. Per table the LLM sees a column profile + 3 sample
   rows and returns a `TableMapping` (relevant?, `{Column}` template rendering a row into the state,
   optional label column + raw-value → label map). Mappings are validated (known columns, label
   column not in the template) and applied in code, so text is never retyped. Export labels are kept
   (`label_source=export`); a row whose label cell is blank is real but unlabelled, and so is every row of a
   table without a label column: those follow the project's label policy (see "Labelling real rows"); a
   non-blank value that maps to no label is skipped. Source `imported`. Generation only fills the remaining gap.
1. **Mine** real states from the repo when the opportunity points at them: test fixtures, prompt
   few-shot examples, enum/label definitions, sample data files, docstrings. LLM extracts
   `(state, label?)` pairs. A stated label is kept (`label_source=export`); states without one follow the
   label policy like unlabelled imported rows.
2. **Generate** synthetic states per label (diverse personas, lengths, styles), in batches, with
   **hard negatives** (near-boundary cases between confusable labels). One worker agent per label,
   all running in parallel; their batches share a `gen_concurrency` semaphore (default 8).
2b. **Orchestrator QA** before labelling: an orchestrator agent reviews a random sample (24) of each
   worker's output and rejects off-label, unrealistic, label-giveaway or near-duplicate texts. Labels
   whose pass rate is below `qa_min_pass` (0.75) are regenerated once with its feedback and
   re-reviewed; after that the passing texts go on and the teacher relabel filters the rest.
2c. **Real-shaped generation** (`shaping.py`): when real labelled train rows exist (imported, mined, kept
   from a previous pass; never the frozen test rows), every generation prompt carries 4 of them as
   **exemplars** (stratified over labels and key templates), each **scrubbed** (names, emails, phone numbers,
   URLs, ids/uuids and numbers become `<NAME>`, `<EMAIL>`, `<PHONE>`, `<URL>`, `<ID>`, `<NUM>`) with its key
   template, length band and register. The instruction asks for the same structure and register with new
   people, topics and wording, labelled as the worker's label. Texts that still contain a placeholder are
   dropped. These rows carry `provenance=real_shaped_synthetic`, `meta.exemplar_hashes`, `meta.batch`, and are
   **train-only** (never split into val/test).
3. **Label** every synthetic state with the teacher LLM (independent from the generator
   prompt — generate-then-relabel; drop rows where the relabel disagrees with the intended label,
   count them as `disagreements`). Real rows without a label are not relabelled here: the label policy
   decides (only the `teacher` policy labels them in this step; under `agent` the `labelling` stage does).
4. **Dedup** (exact + embedding cosine > 0.95), **balance** (cap per label), **split** 70/15/15
   stratified; real rows (imported, then mined) preferred for test, and the log reports the
   real fraction of the test split.
   An exported table may carry a `split` column: rows marked `test` (or `holdout`) are a frozen
   test split. All of them are kept as test, never sampled, balanced or re-split; dedup removes
   train rows that duplicate them; train/val (85/15) come only from the other rows. The column
   is hidden from the table-mapping LLM. A table with a `split` column that exceeds the read
   sample is skipped rather than imported partially.
4b. **Leak check** after the split: every generated train row is dropped when its 4-gram Jaccard with any
   val/test row is ≥ 0.8 (`forbidden`); real-shaped rows are also dropped on any 8-gram shared with the real
   pool, a 4-gram Jaccard ≥ 0.5 with a real row, or a token from the real identifier set (names, emails,
   ids, long numbers, URLs). Drops per guard: `stats.shaped_drops`.
5. Write `dataset.jsonl` (never a `needs_label` row), hand real rows to the queue (pending ones, plus the
   teacher-labelled real rows that made it into the dataset, for review), emit `curation` progress throughout
   with `counts.needs_label`. `curated` always moves to `labelling`, which disposes the pending rows.

## Frontend ([UI])

Vite + React 18 + TS, React Router, TanStack Query, Recharts, Tailwind CSS v4. Dark/light themes.
Pages: Projects (list + "New project" dialog: repo URL/path, task text, provider/model, **encoder** (required;
preselected from `GET /api/encoders/recommendation`, its reason shown under it)) · **Autopilot**
(`/autopilot`: target cards (name, kind, encoder, enabled toggle, Edit, Delete with a "Runs and journal stay."
confirm, Run now, the last run's `RunProgress` linking to its page, pending one-time actions (e.g. "redraw val ... in the next
scheduled or manual run", with who approved it and the last error), the schedule editor), "New target", the
model-job queue, run chips linking to the run page, the filterable `ActivityJournal`, `GroupReadiness` and a
`ProvenancePanel`; `?run=<id>` redirects to the run page) · **Target dialog** (`autopilot/TargetDialog.tsx`, state in
`autopilot/targetForm.ts`): name (+ id on create, derived from it), kind (head set | project), what (project, or
head-set preset + heads chips from `inspect`), encoder (`EncoderSelect`, limited to the preset's encoders, required; a
new target starts on the preset's recommendation when its derived folder is outside the protected roots
(`savableEncoders`), else on none, and the hint says why) and output folder (derived, editable), data sources
(toggle + the settings the preset declares per kind, each a choice among its declared `values`; turning a source
off asks first, `disableSourceWarning`: the adapter removes the rows it supplied at the next run), schedule (`ScheduleEditor`, once the
target exists), policy (collapsed); each 422 field error shows under its field · **Run page**
(`/autopilot/runs/:runId`, lazy `pages/AutopilotRunPage.tsx`): header (target name, trigger, state, encoder,
`RunProgress`: "3 of 9 heads · Training ticket-priority (round 1) · 42 min elapsed · ~1 h 10 min left"),
`StepTimeline` (grouped by head, collapsible), `HeadResultTable` (control F1, final F1, reference / incumbent F1, gate,
Promoted / Kept, rounds), row counts, the run's journal, then `RunDetail` per head; components in
`src/autopilot/` · `EncoderSelect` (`components/`, options "name · backend · quant · ~GB · Downloaded / Not
downloaded · used by N") in the new-experiment dialog, the experiment header ("Encoder", tooltip: where it came
from), the proposal (`workspace/ProposalEncoder.tsx`: Forge's recommendation for the selected opportunity, "Use
recommended") and the target dialog · the header's health chip "Encoders N in use" (one line per encoder in its
tooltip) · Project
workspace: left = agent chat (always visible, streams, shows tool steps compactly); right = tabs:
**Pipeline** (live state diagram of rules.py with path taken + who/why on each edge, event timeline,
REJECTED events highlighted), **Proposal** (ranked opportunity cards, file refs with snippets, fit
score, benefits/risks, editable TaskSchema, Accept / Request changes; accept starts curation directly, the
labelling agent being the default policy), **Data** (counts by
label/source/split charts, filterable table with each row's label source, edit/delete rows), **Labels**
(route `experiments/:pid/label`: an automatic stage showing who labelled the real rows — agent, adjudicated,
you, teacher, export, excluded by the agent, set aside — the policy chip (default `agent`, "Change" opens the
policy dialog), and `LabelQueue` as optional review and override: while a stage locks the queue a picked
label is recorded as a `label_override`; in `label_review` the gate's Label / Send to teacher / Exclude /
Continue), **Training** (live loss + val-acc
curves, animated similarity heatmap and 2-D embedding scatter with an epoch scrubber), **Evaluation**
(model comparison bars vs threshold line, gate pass/fail, confusion matrix, reliability diagram, per-
label F1, latency, error browser), **Playground** (type a state → live probability bars, fine-tuned vs
reference side by side, rank mode), **Package** (file list, download, rendered INTEGRATION.md with
copy-to-clipboard "hand to coding agent"). **Labelling** page (`/labelling`, `/labelling/:lsid`): every
label set, including head-set head dirs. Both use `components/LabelQueue.tsx`: state text (monospace, left),
question + options with criteria on keys 1–9 (right), teacher suggestion + confidence chip, label-source chip,
`S`/`→` skip, `←` previous, `⌘Z`/`U` undo, progress meter; filters Needs label / Agent / Teacher / Excluded /
All; an excluded row shows the agent's reason, an adjudicated row a chip. Settings page: agent, connections, guards
(no encoder section: there is no app-wide encoder). All live state comes from the SSE stream (the Autopilot pages
poll).

## Hosting (`forge/instance_lock.py`, `forge/service.py`)

One server per data folder: `forge-api` holds `<data>/forge-server.lock` (`fcntl.flock`, `{pid, port}`) and
refuses to start beside another holder, naming it (`instance_lock.holder` reads it without taking it). The
autopilot schedule needs a running server; `forge service install|uninstall|status` (console script `forge`)
manages a per-user launchd agent (label `service.LABEL`) rendered from
`forge/launchd/forge-api.plist.tmpl`: `python -m forge.api` in the backend folder, `RunAtLoad`, `KeepAlive`,
`ThrottleInterval` 60, environment = `PATH`, `HOME`, `LANG`, the variables Forge reads by exact name
(`service.ENV_NAMES`) and the names the head-set presets list in `env_passthrough` (`service.passthrough`, read
from the presets file the agent will use), with their values taken from the installing shell (`FORGE_DATA`
resolved); nothing is copied by prefix, so a token or key exported there is never written into the plist unless a
preset names it (an adapter that needs a secret should pass through the command that fetches it, not the secret).
The plist is written 0600;
stdout and stderr go to `<data>/logs/forge-api.log`, created 0600 in a 0700 folder before launchd opens it.
`install --dry-run` prints the plist only.

## Model-job queue (`forge/jobqueue.py`)

Every job that loads a model or drives the teacher LLM runs inside `JobQueue.slot(kind, owner)`: project
stages `curating`, `selecting`, `training`, `evaluating`, the teacher part of `labelling`, the label-review
teacher job, and the autopilot's `label`, `generate`, `train` and `gate` steps.

- FIFO, concurrency 1 per server; every job is a `model_jobs` row (`queued → deferred? → running → done |
  failed | cancelled`). A restart marks left-over rows `interrupted`; their owners queue them again
  (`Orchestrator.restore_all` re-enters the interrupted stage, the autopilot resumes the run at its cursor).
- **One server per data directory**: `forge-api` holds an exclusive `flock` on `<data>/forge-server.lock`
  (`forge/instance_lock.py`, content `{pid, port}`) and exits with the holder's pid and port when it is taken.
  The restore above assumes no other server is executing the same rows.
- **Preflight** before each job: macOS memory pressure (`kern.memorystatus_vm_pressure_level`) must be at most
  `Settings.preflight_max_pressure_level` (1 = normal), and no foreign model or training process may run
  (command lines matching mlx / llama-server / embed_worker / ollama runner ...; this server's own children
  and `Settings.preflight_ignore` markers — e.g. another local app's resident encoders — do not count). Otherwise the
  job is `deferred` with the reason (journalled / logged once per reason) and re-checked every
  `Settings.job_defer_s` (60 s). Nothing is ever killed.
- The queue never holds the machine-wide flock: the child that loads a model still takes `ModelJobLock`.

## Autopilot (`forge/autopilot/`)

The autopilot keeps **targets** trained without a human: it detects what each one needs, acts, and journals
every decision with its reason and counts. A target is one of two kinds:

- `project`: a Forge project, inspected through the orchestrator (see "Project targets");
- `headset`: an external **head set** (several heads trained and shipped together outside Forge), driven
  through its **adapter**, a subprocess that speaks the JSON-lines contract below. A head-set target is always
  made from a **head-set preset** (below), which is where the adapter command and its folders come from.

| module | owns |
|---|---|
| `__init__.py` | `AutopilotService`: registry, run queue (one autopilot run at a time), restore, reports |
| `targets.py` | `Target` (`id`, `name`, `project` \| `headset`), `HeadsetTarget` (preset, adapter argv, cwd, env, dataset root, encoder (registry key), output root, heads, protected roots / globs / files, `sources`, `adapter_config`), `Source`, `Policy`; the one-off row migrations |
| `presets.py` | head-set presets loaded from the presets file (below): adapter argv, cwd, env, dataset root, protected roots / globs / files, declared sources, encoder backends, base output root; `heads()` via `inspect` |
| `validation.py` | `FieldError(field, msg)`, `plain_errors(ValidationError)` → `[{field, message}]` |
| `describe.py` | step labels, `progress`, `timeline`, `counts`, `results` of a run (the UI only renders them) |
| `detect.py` | `HeadStatus` (the adapter's `inspect` shape) → `[Need]` (pure) |
| `balance.py` | balance target maths (pure) |
| `diagnose.py` | failed round → remedies, from VAL evidence only (pure) |
| `promotion.py` | the promotion rule (pure) |
| `runner.py` | the bounded, resumable run loop; writer guard |
| `adapter.py` | subprocess client for the head-set adapter contract |
| `scheduler.py` | daily / weekly cadence |
| `journal.py` | `autopilot_journal` append / query; `autopilot` events into a project's stream |
| `groups.py` | call-group readiness |
| `actions.py` | one-time actions the user approved (`resplit_val`), `autopilot_actions` |

Tables (`store.py`): `autopilot_targets(id, kind, config_json, enabled)`, `autopilot_schedules(target_id,
cadence, weekday, time_local, next_run, last_run, enabled)`, `autopilot_runs(id, target_id, trigger
schedule|manual|data, state queued|running|done|done_with_failures|failed|interrupted, step_cursor, round,
result_json, started, ended, created, first_started)`, `migrations(name, applied)` (`Store.migrate_once`), `autopilot_journal(id, run_id, target_id, head, round, action, reason, counts_json, ts)`,
`autopilot_actions(id, target_id, action, approved, reason, state queued|done, queued, run_id, done, last_error, result_json)`,
`model_jobs(id, kind, owner, state, enqueued, started, ended, error)`.

**Detectors** (every `Need` carries a reason and counts, and is journalled):

| need | fires when | remedy |
|---|---|---|
| `unlabelled` | a real row has no final disposition | agent labelling (`label`) |
| `imbalance` | a label group is below its target share | shaped generation for it (remedy `none`, with the reason, when `inspect` reports the head or group cannot be generated for) |
| `small_data` | a label group has < `min_count` (150) train rows | shaped generation (same) |
| `shape_gap` | ≥ 20% of real key templates absent from synthetic, length KS D > 0.25, or a channel gap | shaped generation steered to the uncovered templates / channels |
| `label_noise` | teacher disagreement > 15% or excluded > 10% of real rows | adjudication re-run (`label`) |
| `confusable` | one off-diagonal pair holds > 25% of a label's **val** errors (≥ 5 errors) | more rows of the confused label (the adapter has no pair focus yet) |
| `real_hurts` | candidate val macro-F1 < control val macro-F1 − 0.02 | `real_cap` (real rows per label) at the majority label's target share |
| `no_control` | a control mix has no run for the current train fingerprint | control run |
| `stale` | new real rows since the last run, or the schedule is due | full run |
| `smoke_rows` | `inspect` reports rows or generation stage folders that a smoke or test run wrote into the real dataset (`cleanup.smoke`, `cleanup.smoke_stages`) | `clean`: rows and stage folders removed (through the writer guard), counted in the `clean` journal entry |
| `reshape` | `inspect` reports shaped rows accepted under an older real-source guard (`cleanup.stale_guard`) | `clean`: recheck, remove the failures (balance generation replaces them) |
| `retrain` | the promoted head learned from rows removed since (`inspect` `retrain`: deleted at the source, source disabled, smoke, reshape) | full run |
| `resplit_val` | a one-time action the user approved is queued (`autopilot_actions`; not a detector) | `resplit-val` in the next **scheduled** or **manual** run (below) |

**Balance target**: prior `p_l` = Laplace-smoothed real labelled prior (the synthetic train prior below 30 real
labelled rows); share `t_l = max(0.5/K, 0.5·(1/K) + 0.5·p_l)` over the K label groups `inspect` reports in
`counts` (a head with fixed options: its labels; a head whose options change per call: whatever groups its adapter
balances on); rows are only added: `N* = max_l(n_l/t_l)`,
deficit `d_l = max(ceil(t_l·N*) − n_l, min_count − n_l, 0)`; at most `gen_cap` (600) generated rows per head per
round, allocated in proportion to the deficits; the rest is carried (journalled `carry`).

**Run loop** (head sets), per head: `pull-sources` + `ingest` (once per run) → `inspect` → needs → `label`
(when unlabelled / noisy; `clean` first when smoke rows or rows of an older real-source guard are reported,
journalled `clean` with the counts) → rounds (≤ `max_rounds`, 3): inspect, generate, control runs (`train --mix
synthetic` / `synthetic+shaped --attempts N`: the candidate's seed set and val choice, reused by fingerprint and
seed count), `train --mix all --attempts N` (attempts chosen on
**val** by the adapter), the **val gate** (val macro-F1 ≥ the head's `gate_threshold`); a failed round is
diagnosed from val evidence → the best val-passing round → `gate` (reads **test once per run**; a gate call
interrupted mid-way is not repeated in that run) → promotion → `promote` or stay in staging → `groups`.
A head `inspect` reports `retired` is skipped (journalled `skip`, outcome `retired`).
Generation asks only for what `inspect` reports it can do (`HeadStatus.generate`): each ask goes out under the
generate label reported for its balance group; a group no label fills, or a head `generate` cannot serve (the
adapter says why, e.g. no real rows to take the shape from), is journalled `skip` with the
reason and counts, and is not carried. **The journal and the run API hold counts, ids and reasons only**: a failed verb's message is its `ERROR <Type>:
<message>` line (adapter messages name ids, counts and reasons); everything else it wrote to stderr, which may quote
real rows (a teacher CLI's reply, a library message), goes to `<FORGE_DATA>/logs/autopilot/<ts>-<run>-<verb>.stderr.log`
(mode 0600, local like the rows) and the error names that file; a non-JSON stdout line is reported by its size;
`Journal.add` cuts every string (reason 1000, count values 200 characters). **Labelling finishes or fails**: after `label` every real row of the head
must be `labelled` or `excluded_by_agent` with a reason; an adapter that could not get there (a row the teacher never
answered after its retries, an export it could not check, a dispute the adjudicator did not settle) exits non-zero
with per-head counts and why, so the head fails (`head_failed`) instead of training on a half-labelled set; the
runner treats a `label` output that still reports undisposed rows the same way. **Heads are isolated**: a verb that exits non-zero fails that head
(`outcome=failed`, `error`, a `head_failed` journal entry with the verb and the adapter's own `ERROR` line; not
retried in that run) and the run goes on with the next head, still builds `groups`, and ends
`done_with_failures` (`run_done_with_failures`). A broken output contract (`AdapterContractError`: malformed
lines, no or two results; `ContractError`: a missing `gate_threshold`, `generate` block, val scores) or a
writer-guard breach fails the whole run.
Each step's output is stored under its key in `result.steps` and `step_cursor` names the last one; a resumed
run replays stored outputs. After every adapter call the **writer guard** compares hashes of the preset's
`protected_globs` under the dataset root (default `*/rows.jsonl`) and its `protected_files`; any change fails the
run. A step may be allowed to **create** named files that did not exist (`resplit-val`: each head's
`val_redrawn.jsonl`, `val_split.json`, `_val_archive/val-<run>.jsonl`); they are protected from the next step on
when the preset's globs cover them (list them in `protected_globs` for an adapter that implements `resplit-val`).

**One-time actions** (`actions.py`): something a target must do once because the user decided so, not because a
detector found a need. An action is queued with `Actions.queue(target, action, approved, reason)` (journalled
`action_queued`; `approved` says who approved it and when, in words) and shows on the target card until a run
completes it. After the first `inspect` a head-set run takes the target's queued actions. `resplit_val` runs in a
`schedule` or `manual` run (a data run journals `action_waiting`): journalled as a
`need`, then `resplit-val --heads <every non-retired head>` in a model-job slot (generator and teacher calls),
one `resplit_val` journal entry per head (new val rows and labels, the archived old val with its sha256, the leak
check, the frozen test's sha256, the procedure; a head it does not apply to is a `skip` with why),
`action_done`, and the heads are inspected again, so this run trains, early-stops and selects on the new val. A
failed `resplit-val` fails the run (`action_failed`; never trained on the val it replaces) and the action stays
queued with `last_error` for the next scheduled or manual run. A head-set's `output_root` may not lie inside a
`protected_roots` entry (e.g. the heads the head set already ships).

**Promotion**: the test gate passed (the adapter's `gate` `passed`, with its reasons in `gate_check.reasons`; an
extra test the adapter runs and reports as `extra_test: {passed, macro_f1}` is added to the reasons) **and** test
macro-F1 ≥ incumbent test macro-F1 − `tol` (0.01 per head, `Policy.tol_per_head`) **and** no regression on the
adapter's rolling real check set, when it keeps one: the gate's `check.passed` false blocks (which stage's check set
decides, and when it has enough rows, is the adapter's call, reported as `check.deciding` and `check.reason`;
candidate macro-F1 ≥ incumbent's, no tolerance); `null` (no check set big enough yet) leaves the first two rules to
decide and the reason is journalled with every stage's scores. No incumbent → the gate alone. Target policy
`promote_over_failing_incumbent` (default off; the target dialog's Policy checkbox): a candidate that misses its
test gate is promoted when the incumbent misses its own (`gate.incumbent.gate_passed` false) and the candidate's
test macro-F1 is strictly higher, the check set not blocking; journalled `gate_policy`, `promote
--over-failing-incumbent` re-checks the rule on the adapter's side. Fair baselines: rounds report val deltas against
each control mix, like for like (same `attempts` seeds, best on val, both sides). A control whose training
fingerprint equals the candidate's is the candidate: the adapter copies it instead of retraining
(`same_as_controls`) and the round reports delta 0 for it (journalled "same training set as the … control").

### Head-set presets (`presets.py`)

Presets are read from a JSON file on the server: `$FORGE_HEADSET_PRESETS` when set, else
`<data dir>/headset_presets.json`. No file = no head-set presets (only project targets can be created); a
malformed file is an error naming the file (`PresetFileError`). The file is re-read when it changes, so editing it
needs no restart. `backend/headset_presets.example.json` is a starting point.

```json
{"presets": [{"key": "my-heads", "name": "My heads", "description": "...",
  "adapter": ["uv", "run", "python", "-m", "my_pkg.autopilot"],
  "cwd": "/abs/dir", "env": {}, "env_passthrough": ["MY_TOKEN"],
  "dataset_root": "/abs/dir", "output_base": "/abs/dir/my-heads",
  "protected_roots": [], "protected_globs": [], "protected_files": [],
  "sources": [{"kind": "event_log", "label": "Event log", "enabled_by_default": true,
               "settings": {"stage": {"values": ["dev", "prod"], "default": "dev", "multiple": false}}}],
  "fixed_heads": null, "encoder_backends": ["mlx"], "adapter_config": {},
  "adapter_config_keys": ["epochs"], "recommended_encoder": null}]}
```

| field | |
|---|---|
| `key`, `name`, `description` | id (a target names it as `headset.preset`), display name, one line for the dialog |
| `adapter`, `cwd`, `env` | the adapter's argv prefix, its working directory, and variables added to the server's environment for it |
| `env_passthrough` | names of shell variables the adapter reads; `forge service install` copies them, with their values, into the launchd agent's environment (see "Hosting"), so name a secret-fetching command rather than a secret |
| `dataset_root` | the head datasets (`--target-config` `data_root`) |
| `output_base` | base of the output folders: a target's `output_root` is `<output_base>-<name>`, derived per encoder as `<output_base>-<key>` (`encoders.output_root`) |
| `protected_roots`, `protected_globs`, `protected_files` | no output root inside a protected root; files the writer guard hashes after every step (globs under the dataset root, plus absolute files) |
| `sources` | the real-data source kinds this head set's adapter can pull (below) |
| `fixed_heads` | `null`: the user picks heads from `inspect`; a list: the target always trains exactly these |
| `encoder_backends` | registry backends the adapter can train on (the encoder picker is limited to them) |
| `adapter_config`, `adapter_config_keys` | default adapter-only keys merged into `--target-config`; the only keys a target's `adapter_config` may set |
| `recommended_encoder` | the encoder the heads are built for; the new-target form preselects it when it fits this machine |

**Commands come only from this file.** What a head set runs and reads (`targets.SERVER_FIELDS`: `adapter`, `cwd`,
`env`, `dataset_root`, `protected_roots` / `_globs` / `_files`) is never taken from an API request: a body naming
one is a 422 per field ("set by the head-set preset; a request cannot set it"), a body without a preset is refused,
a stored row's copy is replaced from the preset on load and dropped from the row (migration
`autopilot_targets_server_fields`), and the runner builds the adapter from the preset alone (a target without one,
or whose output folder the preset does not allow, fails its run with the reason).

**Sources** are declared per preset: `kind` (the id the adapter knows), `label` (shown in the dialog),
`enabled_by_default`, and `settings`: per setting name the allowed `values`, a `default`, and `multiple` (a list
of values instead of one). A target's `sources: [{kind, enabled, settings}]` may name only kinds its preset
declares, each at most once, with settings the preset declares and values among its `values` (else a 422 on
`headset.sources.<i>.settings.<name>`); no setting is ever a command, path or URL. They reach the adapter as
`target_config()["sources"]`.

**Targets: names, edits.** A target has a display `name` (its id is derived from the name once, at creation). A
head-set target names its `preset`. `adapter_config` takes only the preset's `adapter_config_keys`; `output_root`
must be a `<output_base>-<name>` folder beside the preset's base (`Preset.output_root_refusal`). The encoder is
validated against the registry and the preset's `encoder_backends`. A project target's encoder is its project's
`base_model` (edited through `PUT /api/projects/{id}/encoder`, one owner).
Edits: an encoder change re-derives an output root that was derived (`<base>-<old key>` → `<base>-<new key>`;
a hand-set root stays; `TargetView.output_roots` shows where each encoder would write) and is journalled
`encoder_changed {from, to, from_name, to_name}`; the adapter's control fingerprint must include the encoder, so
the next run trains fresh controls. Other changes are journalled `target_edited {fields}`; also `target_created`,
`target_enabled` / `target_disabled`. A run records `result.encoder`; runs on different encoders are not compared.

**Progress model.** Units = 3 prelude steps (pull, ingest, inspect) + one per planned head + 1 (call groups); the
head in progress adds the index of the furthest phase it reached / 8 (`needs, label, generate, controls, train,
gate, decision, promote`); a head is done when it has an `outcome`. ETA = mean duration of completed non-skipped
heads × heads remaining, null ("ETA after the first head") before one completes. `detail` = the adapter's live
progress message. A run without a plan shows "Progress not recorded for this run". The UI polls (no SSE): 3 s
while a run is queued or running, then stops.

**Project targets** are inspected through the orchestrator (dataset counts per label and source, undisposed
queue rows) and their needs journalled into the project's stream; the project's pipeline acts on them (its
labelling is automatic). The autopilot never fires a human event on a project, and a `delivered` project is
final (retraining it is a new project).

**Scheduler**: ticks every 60 s; a due schedule enqueues one run (trigger `schedule`) and moves `next_run` to
the next future slot, so runs missed while the server was down enqueue once. A target with a queued or
running run coalesces further triggers into it. On start, runs left `running`/`interrupted` are re-queued
and resume at their cursor.

### Autopilot adapter contract

This section is the contract a head-set adapter implements. Forge never imports the adapter: it runs it as a
subprocess, one verb per step, so the adapter can use its own interpreter and dependencies, every model job frees
its memory when its process exits, and Forge's tests can use a fake adapter that prints the same JSON.

**Invocation.** `<adapter argv> <verb> --target-config <json> [--run <id>] [--flag value]...`, run in the preset's
`cwd` with the server's environment plus the preset's `env`. Flags: `--name value`; a boolean flag is bare
(`--adjudicate`); a list is one comma-joined value (`--heads a,b`); a repeated flag is given once per value
(`--focus-template T1 --focus-template T2`).

- **stdout** is JSON lines only: `{"event": "progress", "verb", "message", ...}` while working (the runner shows
  `message` as the run's live detail), then exactly one `{"event": "result", "verb", "data": {...}}`. Library
  prints, training logs and child processes go to stderr.
- **exit code** `0`: the verb did its job (a failed gate is a result: `passed: false`). Non-zero: failure; stderr
  ends with `ERROR <Type>: <message>`, naming ids, counts and reasons, never row text (it is the only stderr line
  the journal keeps). Nothing is retried or replaced silently.
- **idempotent** per runner cursor: a staged candidate's gate is stored and returned again (`cached: true`), a
  control is reused while its fingerprint holds (`reused: true`), a generation batch returns its stored result
  (`cached: true`), ingest adds new ids only, labelling touches only undisposed rows.
- **one model job per process**: `train` and `gate` load an encoder (through Forge's embedding worker, or anything
  that honours the machine-wide model lock); `generate` and `label` call the teacher / generator; the rest load
  nothing. The runner runs `label`, `generate`, `train`, `gate` and `resplit-val` inside a model-job slot.

**`--target-config`** = `HeadsetTarget.target_config()`, a JSON object:

| key | |
|---|---|
| `encoder` | the target's encoder registry key (`backend/encoders.json`; the adapter may read that file for the pinned repo and revision) |
| `data_root` | the preset's `dataset_root` |
| `output_root` | the target's output folder; never inside a protected root |
| `heads` | the heads the user picked (`[]` = all the adapter has) |
| `attempts` | seeds per training run (`Policy.attempts`) |
| `tol` | promotion tolerance per head (`Policy.tol_per_head`; missing head = the adapter's default) |
| `sources` | `[{kind, enabled, settings}]` as validated against the preset; `pull-sources` and `ingest` read only the enabled ones, and the adapter rejects a kind it can't pull |
| ... | the preset's `adapter_config` merged with the target's (keys from `adapter_config_keys` only); per call the runner may override `real_cap` |

**Verbs** (what the autopilot sends and reads; an adapter may return more):

| verb | arguments | `data` |
|---|---|---|
| `inspect` | `--heads a,b` (default all) | `{heads: [HeadStatus]}`; read-only, loads no model (also used by the target dialog's heads picker) |
| `pull-sources` | `--out <dataset_root>/_raw/real` | per enabled source: `file`, `rows`, `new_rows`, `labelled`, and optionally `snapshot_restamped` (journalled when set) |
| `ingest` | | `{heads: {head: {added, ...}}, disabled_exports_removed?}`; per head optionally `source_disabled_removed`, `_shaped`, `_artifacts`, `shaped_removed`, `stage_folders_removed`, `artifacts_removed` (journalled) |
| `label` | `--head H`, `--adjudicate` \| `--no-adjudicate` | `{heads: {H: AgentCounts}}` |
| `clean` | `--head H` | `{head, smoke: [id], reshape: {guard: [id]}, smoke_stages: [folder], ...}` |
| `generate` | `--head --label --n --exemplars --batch <run>-<round>-<i> [--focus-template T]... [--focus-channel C]...` | `{accepted, drops: {guard: n}, ...}`; only for heads whose `kind` is in `Policy.generate_kinds` (empty = every kind); `--label` is always a label `inspect` reported under `generate.groups` |
| `train` | `--head --encoder --mix synthetic\|synthetic+shaped\|all --attempts N` | `{path, fingerprint, val: {macro_f1, per_label, confusion}, reused, same_as_controls?}`; the adapter chooses between attempts on **val** only |
| `gate` | `--head --staging PATH` | `{passed, test_macro_f1, gate_check: {passed, reasons}, extra_test?, check?, incumbent: {macro_f1, gate_passed}, reference?, cached}`; reads the frozen **test** split, once per candidate |
| `promote` | `--head --staging PATH [--over-failing-incumbent]` | `{path, replaced, provenance}`; refuses a candidate whose gate failed unless `--over-failing-incumbent` and the rule holds |
| `groups` | | `{groups: {name: [GroupHead]}, call_cooccurrence, cross_group_calls, shared_encoder: {setting}}` |
| `resplit-val` | `--heads a,b --run R` | `{heads: {H: {applicable, cached, procedure, val: {file, rows, labels, sha256}, archived: {file, rows, labels, sha256}, leak_check: {duplicate, near_test, near_train}, test_sha256}}}`; optional (only when a `resplit_val` action is queued) |

`check` (rolling real check sets, optional) = `{stages: {stage: {n, candidate: {macro_f1, ...}, incumbent}},
deciding, passed (true|false|null), reason}`. `reference` (optional) = the test numbers of a reference head the head
set already ships (`{macro_f1, ...}`), shown beside the incumbent. `staging` paths are the adapter's own: the
autopilot passes back the `path` a `train` returned.

**`HeadStatus`** (`inspect`), the fields the autopilot reads:

| field | |
|---|---|
| `head`, `group`, `retired`, `kind` | registry facts (`retired`: a reason, or null) |
| `gate_threshold` | the head's test gate threshold, also used by the val gate; required |
| `counts` | per label group: `synthetic` (train), `shaped`, `real_labelled`, ...; plus `(undisposed)` and `(excluded)` |
| `generate` | `{available, reason, groups: {group: {label, reason}}}`: what `generate` can do for this head now; required before generation is planned |
| `teacher_disagreement_rate`, `excluded_rate` | label-noise inputs |
| `shape` | `uncovered_template_share`, `top_uncovered`, `length_ks_d`, `channel_gaps` |
| `last_run` | newest staged candidate: `staging`, `val` (macro_f1, per_label, confusion), `attempts`, `gate` |
| `incumbent` | the promoted head's test `macro_f1` (null when none) |
| `fingerprints`, `controls` | fingerprint per mix; whether each control mix already has a run |
| `stale.new_real_rows` | real rows not seen by the last candidate |
| `cleanup` | `{smoke, stale_guard, smoke_stages}` (optional) |
| `retrain` | `{n, kinds, reasons}` when the promoted head learned from rows removed since, else null (optional) |

**`AgentCounts`** (`label`): `agent_labelled`, `adjudicated`, `excluded_by_agent`, `still_disputed`,
`teacher_failed`, `export_checked`, `export_contradictions: [{rid, export_label, teacher_label}]`, `kept_user`,
`undisposed`, `counts`.

**`GroupHead`** (`groups`): `head`, `retired`, `promoted`, `encoder`, `gate_passed`, `macro_f1` (test),
`reference_macro_f1`, `tol`, `provenance`.

**Rules an adapter keeps**, which the autopilot's guarantees rest on:

- **Test is read once per candidate.** `train` trains on train + val only and chooses between attempts on val;
  `gate` scores the chosen head on the frozen test split and stores the result; a second `gate` returns it.
  Controls are never gated.
- **Fingerprints.** A training fingerprint covers the encoder, the hyperparameters and the train row ids, so a
  control is reused only while its training set holds, and an encoder change trains fresh controls.
- **Data safety.** Frozen splits and the files the preset protects are never written (the writer guard fails the
  run if they change); real rows are `split=train` only; real-row read-modify-writes hold the head directory's
  `.real_rows.lock` (see "Labelling real rows"). The output root is never inside a protected root.
- **Real data stays local.** Real rows go only where the head set's owner allows (its teacher, and scrubbed as
  generation exemplars); the journal gets counts and ids, never row text.

### Real-data sources (head sets)

`pull-sources` reads every enabled source of the target's `sources` (kinds and settings as the preset declares
them) into `<dataset_root>/_raw/real/` through the writer guard; `ingest` adds the new real rows to each head's
queue (`real_rows.jsonl`), `split=train` only, so frozen val/test splits never contain them. How a source is read
(an export of a local store, a service, files) and how its rows are deduplicated is the adapter's. The `pull`
journal entry counts new rows per source; the `ingest` entry counts, per head, what the adapter reports (added,
deleted-at-source, same-content, format-excluded rows, check sets).

What the autopilot expects of an adapter's sources:

- **Deletion reaches training.** A row deleted at its source leaves on the next `ingest`: from the queue, the
  labelling-log content (entries stay, `redacted`), the rows generated with it as an exemplar, the embedding cache
  and the unpromoted heads trained on it; a promoted head that trained on it is reported under `inspect`
  `retrain`, which the autopilot answers with a full run. An ingest that cannot check a stored row against a
  fresh snapshot of its source fails rather than keeping it.
- **A disabled source** is removed the same way: stored rows whose source the target no longer pulls are removed
  everywhere and counted per head (`source_disabled_removed`, `_shaped`, `_artifacts`; journal
  `source_disabled_removed`), with their pulled exports (`disabled_exports_removed`). The target dialog asks
  before a source is turned off.
- **Rolling real check sets** (optional): an adapter may hold the newest real rows of each stage out of training
  and score candidate and incumbent on them at `gate` (`check`), which promotion may not regress on.

**Call groups**: a group is ready when every non-retired head is promoted on the candidate encoder, passed
its gate and is within `tol` of its `reference` score; when the adapter reports `shared_encoder`, the
report notes that every group shares that encoder setting, so an encoder switch is one shipping decision for all
groups.
