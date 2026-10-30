# CLM Forge

<a href="https://fableinnovations.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/fable-logo-light.svg">
    <img alt="Fable Innovations" src="docs/fable-logo-dark.svg" height="40">
  </picture>
</a>

Built and maintained by [Fable Innovations](https://fableinnovations.com).

Link a repository (or describe a task in plain text). An AI agent looks for the place where a
**CLM** fits best. A CLM is a Contrastive Language Model: a frozen encoder (picked per experiment; Forge
recommends one) plus small trained projection heads (the reference head
[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) is trained on Qwen3-8B). The agent
proposes the change. Once you accept it, Forge curates training data, trains a head locally,
evaluates it against baselines, and ships a pip package together with an `INTEGRATION.md` that a
coding agent can carry out.

A CLM suits fast, calibrated scoring over a small, enumerable set of options: routing, triage,
intent, yes/no verification, guardrails, relevance and quality scoring, reranking and best-of-N. It
does not suit generation, multi-step reasoning or open-ended output. When nothing in the repo fits,
the analyst returns `no_fit`.

`SPEC.md` is the binding contract between the parts.

## Architecture

```
frontend/ (Vite + React)  ──REST + SSE──▶  backend/forge/api.py (FastAPI, :8765)
                                              │
                                   orchestrator.py  (LangGraph decide → act → route)
                                     │        │
                     rules.py (state machine, │  stage jobs: curator.py · trainer.py · evaluator.py · packager.py
                     the guardrail, no AI) ◀──┘  agent steps: analyst.py, review decision, chat
                                              │
                         llm/ (claude: Agent SDK on your subscription · ollama)   store.py (SQLite) · events.py (pub/sub)

autopilot/ (detect → act → journal, scheduled)  ──subprocess JSON lines──▶  head-set adapter (yours, named by a preset)
     │            └──▶ orchestrator (Forge projects)
jobqueue.py: one model / teacher job at a time for project stages and autopilot steps alike
```

**The invariant: the model proposes and the rules decide.** Every state change goes through
`rules.Project.fire(event, actor, reason)`:

- Each event belongs to exactly one actor (`agent`, `human` or `system`).
- The agent is offered only the agent events that are legal in the current state.
- Each decision is structured output with a required `reason`.
- An illegal proposal is recorded as a `rejected` event and never coerced into a legal one.
- The agent can never `accept`, because that event is human-only.
- The agent can never `ship` a failed gate. The `gate_passed` guard recomputes the gate from the
  EvalReport numbers: the fine-tuned metric must reach the threshold and beat both the zero-shot
  reference head and the majority baseline.
- Decision loops are bounded twice: they stop when the state changes or no agent move is left, and
  `recursion_limit` is a backstop.

```mermaid
stateDiagram-v2
    [*] --> created
    created --> analyzing : start (system)
    analyzing --> proposed : propose (agent, verdict = fit)
    analyzing --> no_fit : declare_no_fit (agent, verdict = no_fit)
    proposed --> no_fit : declare_no_fit (agent)
    proposed --> analyzing : revise (human, feedback)
    proposed --> accepted : accept (human only)
    accepted --> curating : begin_curation (system)
    curating --> labelling : curated (system)
    labelling --> selecting : labels_settled (system, every real row disposed, ≥ min_examples, all labels in train)
    labelling --> label_review : needs_review (system, opt-in human policy, rows undisposed)
    label_review --> labelling : review_done (human only, all labelled or the rest set aside)
    selecting --> training : selected (system, base model chosen from the candidates)
    training --> evaluating : trained (system, checkpoint)
    evaluating --> reviewing : evaluated (system, report)
    reviewing --> packaging : ship (agent, gate_passed)
    reviewing --> curating : recurate (agent, attempts < max)
    reviewing --> no_fit : give_up (agent)
    packaging --> delivered : packaged (system, manifest)
    delivered --> [*]
    no_fit --> [*]
    cancelled --> [*]
    note right of errored
        fail (system): any non-final state → errored
        retry (human): errored → the state it failed in
        cancel (human): any non-final state → cancelled
        label_override (human): any non-terminal state → itself,
        applied by the next labelling stage
    end note
```

### Pipeline

1. **Analyse** (`analyst.py`). With Claude, an agentic session uses the read-only tools `Read`,
   `Grep` and `Glob` with `cwd` set to the repo. Writes, Bash and web access are disallowed, and every
   step is streamed as a `tool` event. With Ollama there are no tools. Instead, `repo.py` builds a
   digest: the file tree, the README, and candidate sites found by grepping for LLM SDK calls,
   classify/label/route/score prompts, regex routers, if/elif ladders over strings, and
   sentiment/intent/toxicity libraries. Both providers return the same `Proposal`. File references
   are then grounded: paths that don't exist are dropped and snippets are re-read from disk.
2. **Decide.** The agent proposes `propose` or `declare_no_fit`. You can chat with it to revise the
   proposal, or accept an opportunity (optionally overriding the TaskSchema, `target_examples` or the
   threshold).
3. **Curate** (`curator.py`):
   - Import real rows from exported tables (see [Using your own data](#using-your-own-data)).
   - Mine the repo for real states.
   - Generate synthetic and hard-negative states with one worker agent per label, all in parallel
     (up to `gen_concurrency`, default 8, batches at once), only enough to fill what the real rows
     don't cover.
   - Orchestrator QA: before labelling, an orchestrator agent reviews a sample of each worker's output
     (off-label, unrealistic, label giveaways, near-duplicates). Rejected texts are dropped; a label
     whose sample pass rate is under `qa_min_pass` (default 0.75) is regenerated once with the
     orchestrator's feedback and reviewed again.
   - When real labelled rows exist, generation is **shaped on them**: every prompt carries 4 scrubbed real
     exemplars (names, emails, numbers, ids... replaced by typed placeholders) with their structure, length
     and register, and asks for new people, topics and wording. Those rows are train-only.
   - Have the teacher relabel each generated state independently and drop rows where it disagrees.
   - Real rows without a label follow the experiment's label policy (see
     [Labelling real rows](#labelling-real-rows)): by default the `labelling` stage's agent labels them.
   - After the split, generated train rows that copy a val/test row (4-gram Jaccard ≥ 0.8), and shaped rows
     that copy the real pool (a shared 8-gram, Jaccard ≥ 0.5, or a real name/email/id), are dropped.
   - Deduplicate on exact text and on embedding cosine.
   - Balance the labels and split 70/15/15, stratified, with real (imported, then mined) rows
     preferred for test. The log reports how much of the test split is real.
4. **The encoder.** Each experiment trains on its own encoder, picked when it is created (the dialog
   preselects Forge's recommendation and says why: the task, the encoder's measured memory peak against
   this machine's model budget, `forge/encoder_choice.py`). The proposal recommends one for each
   opportunity's task ("Use recommended"), and the experiment header changes it until a stage runs.
   There is no app-wide encoder. The `selecting` stage records the experiment's encoder (a base-model
   comparison runs only when given several candidates: each gets a full head-training run on train,
   scored on **val**; test is untouched). Qwen3-8B warm-starts from the reference CLM head; other
   encoders train their head from scratch. GGUF encoders (e.g. [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B))
   need `llama-server` on PATH (`brew install llama.cpp`).
5. **Train / evaluate / package.** The ML side does this in worker threads (`trainer.py`,
   `evaluator.py`, `packager.py`). The evaluator also scores the teacher LLM on up to 50 test rows.
6. **Review.** The agent proposes `ship`, `recurate` (with curation guidance) or `give_up`.
   `recurate` keeps the existing dataset, freezes the test split and adds guided rows.

## Using your own data

Export real examples from your database into the target repo and Forge trains and tests on them
instead of (or alongside) generated text:

- Put `.csv`, `.tsv` or `.jsonl` files in `clm_data/` at the repo root. Files there are read even
  when gitignored, which is where production exports belong. Tracked tables elsewhere are also
  considered.
- One row per decision. Include the input columns the call site sees, and if you have it, the
  column holding the real answer (for example `repair_status`). Column names don't matter.
- For each table the agent sees only the column profile and three sample rows. It decides whether
  the table is relevant, writes a template that renders a row into the call-site state text, and
  maps the answer column's raw values onto the task's labels. Rows are then rendered in code, so
  the text is copied verbatim. Mappings that put the answer column into the input, or name columns
  that don't exist, are rejected.
- Rows with an answer keep that label (shown as "export"). Rows without one (no answer column, or a
  blank cell in it) are real but unlabelled: the label policy you chose at accept decides who labels
  them. A value that maps to none of the labels is skipped.
- Large tables are sampled (uniform reservoir of 20k rows read, then ~1.3x the per-label target).
- To supply your own frozen test set, add a `split` column and mark those rows `test`. Forge keeps
  every one of them as the test split from the first pass on (no sampling, balancing or re-splitting),
  drops train rows that duplicate them, and draws train/val only from the other rows. A table with a
  `split` column larger than the 20k-row read sample is skipped; split it into smaller files.
- Imported text is sent to the LLM provider only when the teacher labels or suggests labels for it (the
  `teacher` or `mixed` policy, or **Send to teacher**); under the `user` policy no LLM sees unlabelled
  rows. Test rows are shipped in the package as `examples/golden.jsonl`. Scrub anything that must not
  leave the machine before exporting.

Import runs on every curation pass. A recurate keeps the earlier rows and the frozen test split, and
adds rows from tables exported since then to train/val only. The copy-paste export prompt is on the
Proposal tab and the Data tab at every stage.

## Labelling real rows

Every real row (imported, mined, or ingested by a head-set adapter) records who labelled it: **you**, the
labelling **agent**, the **teacher** LLM, or the **export** itself. A real row without a label is never
trained on until it has one. Labelling is an automatic stage; nothing waits for you unless you ask it to:

- **The labelling agent** (policy `agent`, the default). After curation the experiment enters `labelling`:
  the teacher labels every real row; when two teacher passes disagree an adjudicator decides, and a row it
  cannot settle (or an answer outside the options) is **excluded by the agent** with a reason. Teacher
  failures are retried twice and then fail the stage visibly instead of becoming labels. Then the
  experiment moves on to base-model selection by itself.
- **Opting into review.** From the Labels tab's policy chip: `I'll label them` (`user`), `The teacher labels
  them` (`teacher`) or `Teacher suggests, I decide` (`mixed`). With those, rows still undisposed after the
  stage stop the experiment at the `label_review` gate: label them, **Send to teacher**, or **Exclude** them
  (set aside; they join the dataset once labelled), then **Continue to training**.
- **Overrides.** Your label always wins. While a stage reads the dataset the queue is locked; a label you
  pick then is recorded as an override (`label_override`) and applied at the next labelling stage.
- **The Labels tab** (`experiments/<id>/label`) and the **Labelling page** (`/labelling`) show one row
  at a time: the state on the left, the question and its options with their descriptions on the right.
  Keys: `1`–`9` label, `S` or `→` skip, `←` back, `⌘Z` or `U` undo. Filter by *Needs label*, *Agent*,
  *Teacher*, *Excluded* (with the agent's reason) or *All*. Your label always wins: neither the teacher nor
  the agent overwrites it.
- Labelled rows go straight into the experiment's `dataset.jsonl` with their label source; the Data
  tab shows it on every row, and editing a row's label there also makes it yours. Edits are refused
  while a stage is reading or writing the dataset (curating, labelling, selecting, training, evaluating).
- Files: `data/projects/<id>/real_rows.jsonl` (current state) and `labelling_log.jsonl` (append-only
  action log, used by undo). Old datasets are read as before; their rows get a label source on read.

**Head-set label sets.** A head directory of a head set (`<root>/<head_id>/task.json` + `real_rows.jsonl`) is a
label set `h:<head_id>` on the Labelling page once its root is listed in Settings `labelset_roots` or in
`FORGE_LABELSETS` (os.pathsep-separated). Its rows are real data the head set's adapter ingested: train only (never
val/test), and labelled by the adapter (the autopilot calls its `label` verb), so the API refuses **Send to
teacher** for them. You can still label and override them here; the adapter and Forge share the queue file under a
lock.

## Autopilot

The autopilot keeps models trained without you: it detects what each target needs, does it, and writes
down why. Page: **Autopilot** (`/autopilot`). Contract: `SPEC.md` "Autopilot".

- **Targets.** A *project* is a Forge experiment, inspected through the orchestrator. A *head set* is a group of
  heads you train and ship outside Forge, driven through its **adapter**: a subprocess of yours that runs one verb
  per step (`inspect`, `pull-sources`, `ingest`, `label`, `generate`, `train`, `gate`, `promote`, ...) and prints
  JSON lines, so every model it loads is released when the step ends. The adapter contract is in `SPEC.md`
  "Autopilot adapter contract"; a head set is registered through a [head-set preset](#head-set-presets).
- **What it detects**: unlabelled real rows, class imbalance and labels under 150 train rows, synthetic data
  that does not look like the real data (missing key templates, length distribution, channels), noisy
  labels, confusable label pairs (from val errors), real rows that make the model worse than the synthetic
  control, missing control runs, and new real rows / a due schedule.
- **What it does**, per head, at most 3 rounds: agent labelling with adjudication; real-shaped generation
  sized by the balance target (at most 600 rows a round, the rest carried); control runs on synthetic data
  only (and synthetic + shaped) for a fair comparison; training with several seeds ranked on val; a val gate;
  diagnosis of a failed round from val evidence only. The test split is read once per run, for the final
  candidate. It is promoted when its test gate passes and it is within 0.01 macro-F1 of the incumbent;
  otherwise it stays in staging. Every step is journalled with its reason and counts. A head the generator
  cannot serve yet (the adapter says why, e.g. no real rows to copy the shape of) trains on what it has, and the
  skipped generation is journalled with the reason. A head whose step fails is marked failed with the adapter's
  error; the other heads still run, and the run ends **done with failures**.
- **Safety.** One model or teacher job at a time on the machine (`jobqueue.py`): before each job memory
  pressure must be normal and no other model process may run (markers in `Settings.preflight_ignore`
  excepted); otherwise the job waits and says why. Nothing is killed. A writer guard fails the run if any step
  changes a file the preset protects (`protected_globs`, `protected_files`: frozen splits, assembled datasets);
  an output folder can never be inside a protected root.
- **Schedule**: daily or weekly at a local time; runs missed while Forge was down run once on start. A
  restart resumes an interrupted run at its last finished step.
- **Only the Forge page changes Forge.** Forge listens on 127.0.0.1 only, answers only requests addressed to
  it (127.0.0.1 / localhost, not a rebound domain), refuses `/api` requests the browser marks as coming from
  another site, and accepts a change (POST/PUT/PATCH/DELETE) only with the Forge origin and the running
  server's session token, which the page fetches from `GET /api/session` (`backend/forge/web_guard.py`). No other
  site may show the Forge page in a frame (`X-Frame-Options: DENY`, CSP `frame-ancestors 'none'`). A
  label made in the labelling page is logged with the browser session it came from (`via`: origin, user agent).
  What a head set runs (its adapter command, folders, environment) comes only from the server-side presets file;
  a request that names one is refused, and the output folder must be one of the preset's `<base>-<name>` folders.
- **One server per data directory.** `forge-api` holds `data/forge-server.lock` (its pid and port) and refuses
  to start while another server holds it, saying which (a second server would re-queue the first one's live runs
  and run them twice).
- **Running without a terminal (launchd).** The schedule only fires while a server runs. To keep one running at
  login and after a crash, install the per-user launchd agent from the checkout and shell whose settings it should
  use (`PATH`, `HOME`, `LANG`, the `FORGE_*` settings Forge reads, by exact name (`forge/service.py` `ENV_NAMES`),
  and the variables your head-set presets name in `env_passthrough`, values included; nothing else, so a token
  exported in that shell never lands in the plist unless a preset names it: pass through the command that fetches
  a secret, not the secret):

  ```bash
  cd backend
  uv run forge service install --dry-run   # print the plist (~/Library/LaunchAgents/<label>.plist)
  uv run forge service install             # write it and load it (launchctl bootstrap gui/$UID)
  uv run forge service status              # plist, launchd's view, the data folder's lock holder, /api/health
  uv run forge service uninstall           # unload and remove it
  ```

  Stop a terminal server on the same data folder first: the agent's server would refuse to start beside it
  (launchd retries every 60 s). Its output, including the access log, goes to `<data>/logs/forge-api.log`
  (owner-only, as is the plist). The template is `backend/forge/launchd/forge-api.plist.tmpl` (`forge/service.py`).

- **Managing targets in the UI.** "New target" opens the target dialog: name, kind (head set or project), what it
  trains (a project, or a head-set preset and the heads to keep trained; none picked = every head the adapter
  reports), the encoder (required; it starts on the preset's recommendation when it fits this machine, with the
  reason under it) and its output folder (derived from the encoder, always `<base>-<key>`, so never the preset's
  base folder; edit it to set your own), the real-data sources the preset declares (each with its settings), the
  schedule and the policy numbers. Turning a source off asks first: at the next run the adapter removes the stored
  real rows it supplied, everywhere, journalled and counted. Each card has Edit, an enabled toggle and Delete (runs
  and journal stay), and shows one-time actions waiting for a run. A mistake comes back next to its field ("Output
  folder is inside a protected folder …").
- **Encoders per target.** Each target trains on its own encoder from the registry (`backend/encoders.json`);
  there is no app-wide default. Changing a target's encoder moves a derived output folder to the new encoder's
  (`<base>-<key>`), is journalled (`encoder_changed`), and the next run trains fresh controls. A project target's
  encoder is its experiment's (also editable in the experiment header).
- **One-time actions.** An action the user approved for a target (today `resplit_val`: redraw every head's
  validation split the way its frozen test split was built, through the adapter's `resplit-val` verb) waits on the
  target card and runs in the next **scheduled** or **manual** run; a data-triggered run leaves it waiting. The old
  val is archived with its hash, and training and selection use the new val from that run on.
- **Run progress.** Each run has a page (`/autopilot/runs/<id>`, linked from the target card and the run list):
  "3 of 9 heads · Training ticket-priority (round 1) · 42 min elapsed · ~1 h 10 min left", the steps grouped by
  head, the per-head results (control, final, reference / incumbent F1, gate, promoted or kept), row counts and the
  run's journal. It refreshes every 3 s while the run is going. Runs from before progress was recorded say so.

The same through the API (`SPEC.md` "REST"): `POST /api/autopilot/targets`, then the schedule. Forge accepts a
change only from the Forge page: the request must say it comes from the Forge origin and carry the running
server's session (`GET /api/session`); anything else gets a 403 that says what was missing. A script doing this
on purpose sends both (here for the example preset below):

```bash
S=$(curl -s http://127.0.0.1:8765/api/session | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
H=(-H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8765' -H "X-Forge-Session: $S")
curl -X POST http://127.0.0.1:8765/api/autopilot/targets "${H[@]}" -d '{
  "name": "Support ticket heads", "kind": "headset",
  "headset": {"preset": "support-ticket-heads", "encoder": "qwen3-0.6b-bf16",
              "sources": [{"kind": "ticket_export", "settings": {"stage": "prod"}}]}}'
curl -X PUT http://127.0.0.1:8765/api/autopilot/targets/support-ticket-heads/schedule "${H[@]}" \
  -d '{"cadence": "weekly", "weekday": 6, "time_local": "03:00"}'
```

### Head-set presets

A head set is registered on the server, never through the API: Forge reads its presets from a JSON file,
`$FORGE_HEADSET_PRESETS` if set, else `<data dir>/headset_presets.json` (`data/headset_presets.json` by default).
Without the file there are no head-set presets and only project targets can be created. Start from
`backend/headset_presets.example.json` (a hypothetical support-ticket head set):

```bash
cp backend/headset_presets.example.json data/headset_presets.json   # then edit the paths and the adapter command
```

Each preset names the adapter command (`adapter`, run in `cwd` with `env` added), the folders it reads and writes
(`dataset_root`, `output_base`), what the writer guard protects (`protected_roots`, `protected_globs`,
`protected_files`), the real-data sources the adapter can pull (each with a `kind`, a `label`, whether it starts
enabled, and its settings with their allowed `values`), the encoder backends it trains on, optional fixed heads,
the adapter-only config keys a target may set (`adapter_config_keys`) and a recommended encoder.
`env_passthrough` lists the shell variables the adapter needs when Forge runs under launchd (their values are
written into the agent's plist, so name a command that fetches a secret rather than the secret). The full schema is
in `SPEC.md` "Head-set presets". Forge re-reads the file when it changes.

**Real-data sources.** At each run `pull-sources` reads the target's enabled sources into
`<dataset_root>/_raw/real/` and `ingest` adds the new real rows to each head's labelling queue, train-only, so
frozen val/test splits never hold them. A row deleted at its source, or supplied by a source you turned off, is
removed on the next ingest from the queue, the rows generated from it and the unpromoted heads trained on it; a
promoted head that learned from it is retrained. What a source is and how it is read is the adapter's; the
expectations are in `SPEC.md` "Real-data sources (head sets)".

## Run

```bash
./start.sh          # http://127.0.0.1:8765 — syncs deps, rebuilds the UI if stale, serves it from the backend
./start.sh --dev    # backend + Vite hot reload at http://127.0.0.1:5173
```

Ctrl+C stops everything the script started, including Claude Agent SDK sessions and Vite. Ollama is
not managed by the script. Manual equivalent:

```bash
cd backend && uv sync && uv run forge-api          # http://127.0.0.1:8765 (also serves frontend/dist if built)
cd frontend && npm install && npm run dev          # http://127.0.0.1:5173 (proxies /api)
```

Tests:

```bash
cd backend && uv run pytest -q --ignore=tests/test_ml_e2e.py --ignore=tests/test_encoder_worker.py   # no model loaded
cd backend && uv run pytest -q tests/test_agent_*.py tests/test_api_*.py tests/test_labelling.py \
  tests/test_autopilot_*.py tests/test_jobqueue.py tests/test_evaluator_no_test_fails.py tests/test_web_guard.py
cd frontend && npm run typecheck && npm test
```

These tests use a fake LLM. Real runs are described under "Verified". `test_ml_e2e.py` and
`test_encoder_worker.py` load real encoders (marked `model_loading`: when Forge's memory admission refuses the
model, e.g. another model is resident, they are skipped with the refusal as the reason, `tests/conftest.py`);
everything else runs on fakes. `npm test` runs the component
tests (vitest + jsdom). Their dev dependencies are not in `package.json` yet; until they are, install
them without saving: `npm i --no-save vitest@5 jsdom@30 @testing-library/react@16 @testing-library/dom@10`.

Runtime data lives in `data/`: `forge.db`, `settings.json`, `projects/<id>/` (dataset.jsonl,
task.json, best_head.pt, eval_report.json, package/) and `repos/` (shallow clones). Set `FORGE_DATA`
to relocate it and `FORGE_PORT` to change the port. A project that was mid-stage when the server
restarted has that stage queued again on startup (no retry needed). An experiment waiting at the
`proposed` or `label_review` gate stays there.

### API

The REST and SSE endpoints are listed in `SPEC.md`, plus a few additions:

- `GET /api/rules` returns the transition table and the event-to-actor mapping.
- `GET /api/projects/{id}/events?follow=false` replays the event history and closes the stream.
- `ProjectDetail` also carries `log`, `max_attempts`, `error` and `failed_from`.
- The labelling queue routes (`/api/labelsets/...`, `/api/projects/{id}/labels/review|override`) and the
  autopilot routes (`/api/autopilot/...`) are in `SPEC.md`.

## Verified

These runs used a small fixture repo: a support app whose LLM classifies tickets into billing,
technical or account.

- **Ollama `qwen3:8b`.** Analysis went from `analyzing` to `proposed` in about 25 s and picked the
  router as a fit.
- **Claude `claude-sonnet-5-5`.** The agentic analysis (Glob → Grep → Read ×3) reached `proposed` in
  about 25 s, with file references at `router.py:21-36`. A chat revision then set the threshold to
  0.80 and `target_examples` to 150.
- **Full Claude pipeline.** Accept → curate (150 rows; 12 mined) → train (8 s, MPS) → evaluate →
  ship → package took about 2 minutes. Test accuracy was 0.958 for the fine-tuned head, against 0.708
  for the zero-shot reference, 0.333 for the majority baseline and 1.0 for the teacher (n = 24). The
  gate passed and the result was a 70 MB wheel.

## Providers

| provider | how | notes |
|---|---|---|
| `claude-subscription` (default) | `claude-agent-sdk` driving your local `claude` CLI login | `ANTHROPIC_API_KEY` is blanked in the subprocess, so no API key is billed. Analysis reads code only through the repo-confined `list_files`/`grep`/`read_file` tools (in-process MCP); Claude Code's built-in Read/Grep/Glob are off because they accept absolute paths, and a PreToolUse hook denies any other tool. Structured output uses `output_format: json_schema`. |
| `codex-subscription` | your local `codex` CLI's ChatGPT login (`codex exec --json`), no API key | `OPENAI_API_KEY`/`CODEX_API_KEY` are removed from the subprocess env. Codex's read-only sandbox doesn't restrict reads, so its shell and every other outward-reaching built-in are switched off and it reads code only through `forge.llm.repo_mcp` (the same repo-confined `list_files`/`grep`/`read_file`), from an empty temp dir with `--ignore-user-config`/`--ignore-rules`/`--ephemeral`. Structured output via `--output-schema`. Models from `codex app-server`'s `model/list`. |
| `openai` | OpenAI Chat Completions with an API key (Settings, or `OPENAI_API_KEY`); optional base URL for any OpenAI-compatible server | Analysis is a real tool loop over read-only `list_files` / `grep` / `read_file`, run in Python and confined to the repo by `repo.safe_path`. Structured output uses strict `json_schema`, falling back to JSON mode. The key is write-only over the API and `data/settings.json` is chmod 600. |
| `ollama` | `http://127.0.0.1:11434`, `format=<json schema>`, thinking off | No tool use; it analyses the repo digest instead. `qwen3:8b` works but is weaker at large nested edits (for example, chat-driven proposal revisions). |

Model lists are never hard-coded: `GET /api/providers` asks each provider what it offers right now
(the `claude` CLI's own model list, Codex's `model/list`, OpenAI's `/v1/models` filtered to chat models, Ollama's
`/api/tags`), cached for 10 minutes; the pickers have a Refresh button. You can set the provider and
model per experiment, or set a default in Settings.

Tracing is opt-in: with `FORGE_MLFLOW=1` and `uv sync --extra trace`, provider calls are recorded as
MLflow spans in `data/mlflow.db`. View them with `uv run mlflow ui --port 5001` (port 5000 collides
with the macOS AirPlay Receiver).

## Encoder notes

- **Registry.** Every encoder lives in `backend/encoders.json` (key, repo, pinned revision, width, quant,
  measured peak memory). There is no app-wide encoder: each experiment and target picks its own (Forge
  recommends one, `GET /api/encoders/recommendation`), the header's health chip lists every encoder in use, and
  serving uses the checkpoint's own `embed_model`. On its first start this version gives every experiment created
  before it its encoder (from its checkpoint, else the old Settings value, recorded in its stream), backfills
  `embed_model` into checkpoints that lack it, and strips the encoder fields from `settings.json` (the original is
  kept as `settings.json.bak-pre-experiment-encoders`).
  The embedding is the last token's final hidden state, L2-normalised, with a 2048-token tail; embeddings are
  disk-cached per model.
- **Memory.** Base models never load into the server process. Each `Encoder` runs its model in an
  embedding worker (`python -m forge.embed_worker`, log in `data/cache/embed-worker.log`) and every
  layer of the budget lives in `forge/memory.py`:
  - *One model per machine.* The worker holds an `fcntl` lock on `~/.cache/clm-forge/model-job.lock`
    while its model is loaded, so a second Forge server or test run waits (log: "waiting for model
    lock: pid N is running …"). An idle worker hands the lock to a waiter and exits; any worker
    exits after `idle_unload_s` without work. In one process, loading one encoder stops another.
    A request sent just as a worker left is resent to a fresh worker. Cancelling a project kills
    its worker, even while it waits for the lock or memory, which frees the lock.
  - *Admission.* Before loading, weights (file sizes; GGUF header for llama.cpp KV) + one max-width
    row + runtime must fit the budget, or the job is refused (`MemoryAdmissionError`), and must fit
    the free RAM minus the reserve, or the job waits up to `admission_wait_s`, then is refused.
  - *Framework caps.* MLX: `mx.set_memory_limit(budget)`, `mx.set_cache_limit(min(2 GB, budget/10))`,
    `mx.clear_cache()` after every batch. torch MPS: `set_per_process_memory_fraction`, `empty_cache`.
    llama.cpp: `-c max_len*np -np 2 -b/-ub max_len -ctk/-ctv q8_0 -fa on --cache-ram 0`.
  - *Fixed widths and sized batches.* Rows are right-padded to a fixed bucket (`buckets`), so MLX
    reuses buffers instead of caching one per width. Rows per batch come from the model shape, the
    bucket and the headroom measured after a max-width warm-up; a batch that peaks above 85% of the
    budget, fails to allocate, or runs while free RAM is under the reserve halves that bucket's rows.
  - *Hard stop.* The server polls the worker tree's footprint (phys_footprint/RSS) and kills it past
    the budget; the job fails with `MemoryBudgetExceeded` and the server keeps running.
  - *Streaming.* Vectors are cached per batch; head training reads the table from a memory-mapped
    `embeddings.npy` (deleted afterwards).

  Settings (`data/settings.json` → `memory`; `GET /api/settings` shows the resolved numbers as
  `memory_resolved`): `budget_gb` (default 60% of RAM: 28.8 GB on 48 GB), `reserve_gb` (default
  max(4 GB, 15%): 7.2 GB), `buckets` (32…2048), `max_rows` 16, `admission_wait_s` 300,
  `lock_wait_s` 3600, `idle_unload_s` 600, `llama_parallel` 2, `llama_kv_type` `q8_0`. A llama.cpp
  KV type other than f16 is part of the embedding cache key, so vectors never mix precisions.
- **Dedup threshold.** Qwen3 last-token embeddings are anisotropic: unrelated short texts already
  sit at a cosine of about 0.93–0.96, and paraphrases at about 0.98–0.99. Near-duplicate removal
  therefore starts at a cosine of 0.975 (`Settings.dedup_cosine`) instead of the 0.95 in SPEC. If a
  threshold would drop more than 30% of the new rows, it is raised automatically.

## Licences
CLM Forge is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attributions.

- CLM (reference heads and the vendored `clm_core`): Apache-2.0
- Qwen3: Apache-2.0
- Winnow-12B (Gemma 4 12B fine-tune): Apache-2.0
- MLflow: Apache-2.0
- Also in use: LangGraph and LangChain (MIT), python-statemachine (MIT), FastAPI (MIT), sse-starlette
  (BSD-3-Clause), claude-agent-sdk (MIT; using Claude is subject to your Anthropic subscription terms).
