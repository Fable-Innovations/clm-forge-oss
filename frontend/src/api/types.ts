// Mirrors backend/forge/models.py and SPEC.md. Keep in sync with the contract.

export type QuestionType = "noul" | "choice" | "score";
export type Source = "imported" | "mined" | "synthetic" | "hard_negative" | "user";
export type Split = "train" | "val" | "test";
export type Actor = "agent" | "human" | "system";
/** Who labelled a row. Null on a synthetic row: the generator's intended label, confirmed by the teacher.
 * `agent` = the labelling agent (teacher passes + adjudication). */
export type LabelSource = "user" | "teacher" | "agent" | "export";
/** A real row's disposition. `excluded_by_agent` rows carry `exclusion_reason`. */
export type LabelState = "needs_label" | "labelled" | "excluded_by_agent";
/** `agent` (the default) labels automatically; the others opt into the human `label_review` gate. */
export type LabelPolicy = "agent" | "user" | "teacher" | "mixed";

export type ProjectState =
  | "created"
  | "analyzing"
  | "proposed"
  | "accepted"
  | "curating"
  | "labelling"
  | "label_review"
  | "selecting"
  | "training"
  | "evaluating"
  | "reviewing"
  | "packaging"
  | "delivered"
  | "no_fit"
  | "cancelled"
  | "errored";

export interface TaskSchema {
  name: string;
  question_id: string;
  type: QuestionType;
  instructions: string;
  criteria: Record<string, string> | string[] | null;
  state_description: string;
  state_example: string;
}

export interface FileRef {
  path: string;
  start_line?: number | null;
  end_line?: number | null;
  snippet?: string | null;
}

export interface Opportunity {
  id: string;
  title: string;
  files: FileRef[];
  current_approach: string;
  task: TaskSchema;
  fit_score: number;
  rationale: string;
  expected_benefits: string[];
  risks: string[];
  data_strategy: string;
  target_examples: number;
  success_metric: "accuracy" | "macro_f1";
  success_threshold: number;
  integration_sketch: string;
}

export interface Proposal {
  verdict: "fit" | "no_fit";
  summary: string;
  opportunities: Opportunity[];
  recommended_id: string | null;
  no_fit_reasons: string[];
  revision: number;
}

export interface Example {
  id: string;
  state: string;
  label: string;
  probabilities?: Record<string, number> | null;
  source: Source;
  split: Split;
  rationale?: string | null;
  meta?: Record<string, unknown>;
  label_source?: LabelSource | null;
  /** "real_shaped_synthetic" for generated rows modelled on scrubbed real exemplars (train-only). */
  provenance?: string | null;
}

/** One real row in a labelling queue (backend forge.models.RealRow) plus the options it is asked. */
export interface RealRow {
  rid: string;
  state: string;
  question_meta: { question?: { type?: QuestionType; instructions?: string; criteria?: unknown }; [k: string]: unknown };
  source: "imported" | "mined" | "user" | "export";
  split: Split;
  label: string | null;
  label_state: LabelState;
  label_source: LabelSource | null;
  suggested_label: string | null;
  teacher_conf: number | null;
  teacher_votes: { pass?: string; label?: string | null; conf?: number | null }[];
  head: string | null;
  skipped: boolean;
  excluded: boolean;
  adjudicated?: boolean;
  exclusion_reason?: string | null;
  updated_at: string;
}
export interface LabelOption {
  key: string;
  description: string;
}
export type QueueRow = RealRow & { options: LabelOption[] };

export interface LabelCounts {
  needs_label: number;
  user: number;
  teacher: number;
  agent: number;
  adjudicated: number;
  export: number;
  excluded: number;
  excluded_by_agent: number;
  suggested: number;
  total: number;
}
export type QueueFilter = "needs_label" | "teacher" | "user" | "agent" | "export" | "labelled" | "excluded_by_agent" | "all";

export interface LabelQueuePage {
  id: string;
  name: string;
  kind: "project" | "head";
  total: number;
  items: QueueRow[];
  counts: LabelCounts;
  task: TaskSchema | null;
  policy: LabelPolicy | null;
  state: ProjectState | null;
  editable: boolean;
}
export interface LabelSetSummary {
  id: string;
  kind: "project" | "head";
  name: string;
  counts: LabelCounts;
  state?: ProjectState;
  policy?: LabelPolicy | null;
  dir?: string;
}
export interface LabelActionResult {
  row: QueueRow | null;
  counts: LabelCounts;
}

export interface ModelScore {
  name: string;
  accuracy: number;
  macro_f1: number;
  n: number;
  note?: string | null;
}

export interface ReliabilityBin {
  bin_lo: number;
  bin_hi: number;
  confidence: number;
  accuracy: number;
  count: number;
}

export interface LabelScore {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Latency {
  embed_ms_p50: number;
  embed_ms_p95: number;
  head_ms_p50: number;
  head_ms_p95: number;
  warm_cached_ms_p50: number;
}

export interface ErrorCase {
  id: string;
  state: string;
  gold: string;
  pred: string;
  probabilities: Record<string, number>;
}

export interface EvalReport {
  metric: string;
  threshold: number;
  gate_passed: boolean;
  gate_reasons: string[];
  models: ModelScore[];
  confusion: { labels: string[]; matrix: number[][] };
  reliability: ReliabilityBin[];
  ece: number;
  per_label: LabelScore[];
  latency: Latency;
  errors: ErrorCase[];
  n_test: number;
}

export interface PackageFile {
  path: string;
  bytes: number;
  description: string;
}

export interface PackageManifest {
  name: string;
  version: string;
  dir: string;
  zip: string;
  files: PackageFile[];
  integration_md: string;
  created_at: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  state: ProjectState;
  created_at: string;
  updated_at: string;
  repo?: string | null;
  task?: string | null;
  provider?: string | null;
  model?: string | null;
  verdict?: string | null;
  metric?: ProjectMetric | null;
}

/** Finetuned model's test score on the gate metric, as reported in project summaries. */
export interface ProjectMetric {
  name: "accuracy" | "macro_f1";
  value: number;
  threshold: number;
  gate_passed: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts?: string;
}

export interface ProjectDetail extends ProjectSummary {
  path: [string, string, string][];
  allowed_human_events: string[];
  proposal?: Proposal | null;
  accepted_opportunity_id?: string | null;
  task_schema?: TaskSchema | null;
  base_model?: string | null;
  /** The experiment's own encoder (registry key, name, spec, and where it came from); null only for an experiment
   * the per-experiment encoder migration could not give one (it must be picked before it trains). */
  encoder?: ExperimentEncoder | null;
  /** Forge's encoder recommendation per opportunity of the proposal (task, encoder sizes, this machine's budget). */
  encoder_recommendations?: Record<string, EncoderRecommendation>;
  base_model_selection?: BaseModelSelection | null;
  attempts: number;
  label_policy?: LabelPolicy | null;
  labels?: LabelCounts;
  /** User labels recorded with label_override, applied by the next labelling stage. */
  label_overrides?: { rid: string; label: string }[];
  chat: ChatMessage[];
}

/** Where an experiment's encoder came from: picked at creation, changed by the user, or set by the one-time
 * migration from its checkpoint or the old app-wide setting. */
export interface EncoderSource {
  from: "create" | "user" | "checkpoint" | "settings";
  detail?: string;
  at?: string;
  migration?: string;
}
export interface ExperimentEncoder {
  key: string | null;
  name: string | null;
  spec: string;
  source?: EncoderSource | null;
}
/** forge.encoder_choice: the encoder Forge recommends, and why. key is null when nothing fits this machine. */
export interface EncoderRecommendation {
  key: string | null;
  name: string | null;
  reason: string;
  profile?: "rich" | "simple" | "general";
}
export interface EncoderRecommendationDetail extends EncoderRecommendation {
  budget_gb: number;
  considered: { key: string; name: string; approx_gb: number | null; fits: boolean; cached: boolean; why_not: string | null }[];
}
/** One encoder an experiment or autopilot target trains on (GET /api/health); nothing is loaded to report it. */
export interface HealthEncoder {
  key: string;
  name: string;
  backend: string;
  cached: boolean;
  model?: string | null;
  loaded: boolean;
  error?: string;
  used_by: { projects: EncoderUser[]; targets: EncoderUser[] };
}
export interface Health {
  ok: boolean;
  /** Every encoder in use (there is no app-wide encoder). */
  encoders: HealthEncoder[];
  /** Experiment encoder specs no registry entry has. */
  unregistered_encoders?: string[];
  providers: { claude: boolean; ollama: boolean; ollama_models: string[]; openai?: boolean };
}
export interface ProviderModel {
  id: string;
  label: string;
  description?: string | null;
}
/** One LLM provider as reported by GET /api/providers: whether it's usable and the models it lists. */
export interface ProviderInfo {
  id: string;
  label: string;
  available: boolean;
  detail: string;
  models: ProviderModel[];
}

export interface Settings {
  provider: string;
  model: string;
  ollama_url: string;
  openai_base_url?: string;
  /** Write-only: send a value to replace the stored key, null to clear it; never returned. */
  openai_api_key?: string | null;
  openai_api_key_set?: boolean;
  /** False when the saved key was entered for a different base URL; it is then never sent. */
  openai_api_key_applies?: boolean;
  openai_api_key_url?: string;
  openai_api_key_env?: boolean;
  thresholds?: Record<string, number>;
  max_attempts: number;
  [k: string]: unknown;
}

// ---- encoder registry (backend/encoders.json, GET /api/encoders) ------------------------------------------
export interface EncoderUser {
  id: string;
  name: string;
}
export interface Encoder {
  key: string;
  name: string;
  backend: "mlx" | "llamacpp";
  repo: string;
  file: string | null;
  revision: string;
  hidden: number;
  quant: string;
  /** Measured peak memory in GB, null when never measured on this machine. */
  approx_gb: number | null;
  approx_gb_source: string | null;
  note: string;
  spec: string;
  url: string;
  /** The pinned snapshot is in the Hugging Face cache. */
  cached: boolean;
  used_by: { projects: EncoderUser[]; targets: EncoderUser[] };
}
export interface EncodersView {
  encoders: Encoder[];
}

export interface DatasetPage {
  total: number;
  items: Example[];
  counts?: Counts;
}

export interface Counts {
  by_label: Record<string, number>;
  by_source: Record<string, number>;
  by_split: Record<string, number>;
  /** Real rows waiting in the labelling queue (curation events only). */
  needs_label?: number;
}

// System One answer dict (clm_core.schema.answer_from_probs)
export type Answer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | {
      type: "score";
      score: number;
      confidence: number;
      legend: Record<string, string>;
      probabilities: Record<string, number>;
    };

export interface InferResult {
  answer: Answer;
  latency_ms: number;
  embed_ms: number;
  head_ms: number;
}

export interface RankItem {
  rank: number;
  candidate: string;
  prob: number;
}

// ---- SSE events ----
interface Base {
  ts: string;
  seq: number;
}
export type CurationPhase = "importing" | "mining" | "generating" | "reviewing" | "labelling" | "dedup" | "splitting" | "done";

export interface SnapshotPoint {
  x: number;
  y: number;
  label: string;
  split: string;
  kind: "state" | "option";
  text: string;
}
export interface Similarity {
  row_labels: string[];
  col_labels: string[];
  values: number[][];
}

export type StateEv = Base & { type: "state"; from: string; event: string; to: ProjectState; actor: Actor; reason: string };
export type RejectedEv = Base & { type: "rejected"; event: string; actor: Actor; reason: string; error: string };
export type LogEv = Base & { type: "log"; stage: string; level: "info" | "warn" | "error"; message: string };
export type ChatEv = Base & { type: "chat"; role: "user" | "assistant"; content: string; id: string };
export type ChatDeltaEv = Base & { type: "chat_delta"; id: string; delta: string };
export type ToolEv = Base & { type: "tool"; name: string; input: string; output: string };
export type ProposalEv = Base & { type: "proposal"; proposal: Proposal };
export interface GenWorker {
  done: number;
  total: number;
  generated: number;
}
export type CurationEv = Base & { type: "curation"; phase: CurationPhase; done: number; total: number; counts: Counts; workers?: Record<string, GenWorker> };
export type EmbedEv = Base & { type: "embed"; done: number; total: number };
export interface BaseModelCandidate {
  spec: string;
  name: string;
  detail?: string;
  url?: string;
  status: "pending" | "running" | "done" | "failed";
  load_s?: number;
  embed_ms?: number;
  val_acc?: number;
  val_soft_ce?: number;
  init_val_acc?: number;
  best_epoch?: number;
  dim?: number;
  seconds?: number;
  error?: string;
  chosen?: boolean;
  note?: string;
}
export interface BaseModelSelection {
  chosen: string | null;
  candidates: BaseModelCandidate[];
}
export type BaseModelEv = Base & { type: "base_model" } & BaseModelSelection;
export type TrainEpochEv = Base & {
  type: "train_epoch";
  epoch: number;
  epochs: number;
  train_loss: number;
  val_acc: number;
  val_soft_ce: number;
  lr: number;
  logit_scale: number;
};
export type TrainStepEv = Base & { type: "train_step"; epoch: number; step: number; steps: number; loss: number };
export type TrainSnapshotEv = Base & { type: "train_snapshot"; epoch: number; similarity: Similarity; points: SnapshotPoint[] };
export type EvalEv = Base & { type: "eval"; report: EvalReport };
export type PackageEv = Base & { type: "package"; manifest: PackageManifest };
export type ErrorEv = Base & { type: "error"; stage: string; message: string; traceback: string };
export type LabelsEv = Base & { type: "labels"; counts: LabelCounts };
export type AutopilotEv = Base & {
  type: "autopilot";
  action: string;
  reason: string;
  counts: Record<string, unknown>;
  head: string | null;
  round: number | null;
  target: string;
  run: string | null;
};

export type ForgeEvent =
  | StateEv
  | RejectedEv
  | LogEv
  | ChatEv
  | ChatDeltaEv
  | ToolEv
  | ProposalEv
  | CurationEv
  | EmbedEv
  | BaseModelEv
  | TrainEpochEv
  | TrainStepEv
  | TrainSnapshotEv
  | EvalEv
  | PackageEv
  | ErrorEv
  | LabelsEv
  | AutopilotEv;

export const MAIN_STATES: ProjectState[] = [
  "created",
  "analyzing",
  "proposed",
  "accepted",
  "curating",
  "labelling",
  "selecting",
  "training",
  "evaluating",
  "reviewing",
  "packaging",
  "delivered",
];
export const FINAL_STATES: ProjectState[] = ["delivered", "no_fit", "cancelled"];

// ---- autopilot (backend forge/autopilot, SPEC "Autopilot") ---------------------------------------------
export interface AutopilotPolicy {
  min_count: number;
  gen_cap: number;
  max_rounds: number;
  attempts: number;
  exemplars: number;
  adjudicate: boolean;
  /** Promote a candidate that misses its test gate over an incumbent that misses its own, when it scores higher. */
  promote_over_failing_incumbent?: boolean;
  tol: number;
  tol_per_head: Record<string, number>;
  [k: string]: unknown;
}
/** One real-data source of a head-set target: a kind the target's preset offers, and its settings (each a value, or
 * a list of values for a multiple setting, from the preset's `sources[].settings`). */
export interface DataSource {
  kind: string;
  enabled: boolean;
  settings: Record<string, string | string[]>;
}
/** One setting of a preset source: the values it takes and its default (a list when `multiple`). */
export interface PresetSourceSetting {
  values: string[];
  default: string | string[];
  multiple: boolean;
}
/** A real-data source a head-set preset offers (GET /api/autopilot/headsets). */
export interface PresetSource {
  kind: string;
  label: string;
  enabled_by_default: boolean;
  settings: Record<string, PresetSourceSetting>;
}
export interface HeadsetTargetConfig {
  /** The head-set preset this target was made from (GET /api/autopilot/headsets); null when registered by hand. */
  preset?: string | null;
  adapter: string[];
  cwd: string;
  env: Record<string, string>;
  dataset_root: string;
  encoder: string;
  output_root: string;
  heads: string[];
  protected_roots: string[];
  protected_globs: string[];
  protected_files: string[];
  /** Real-data sources pull-sources and ingest read (only the enabled ones). */
  sources: DataSource[];
  /** Adapter-only --target-config keys (passed through to the preset's adapter). */
  adapter_config: Record<string, unknown>;
  policy: AutopilotPolicy;
}
/** A named head set a target can be made from (GET /api/autopilot/headsets). */
export interface HeadsetPreset {
  key: string;
  name: string;
  description: string;
  dataset_root: string;
  output_base: string;
  protected_roots: string[];
  /** The heads this preset always trains; null = pick from the adapter's inspect. */
  fixed_heads: string[] | null;
  /** Encoder registry keys the preset's adapter can train on. */
  encoders: string[];
  /** The output folder each of those encoders writes to (derived by the backend). */
  output_roots: Record<string, string>;
  /** The encoder a new target on this head set starts on (its own encoder when it fits this machine). */
  recommendation: EncoderRecommendation;
  /** The real-data sources a target on this head set can pull from. */
  sources: PresetSource[];
}
export interface PresetHead {
  head: string;
  group: string | null;
  retired: string | null;
}
/** The body POST / PUT /api/autopilot/targets take. */
export interface TargetBody {
  id?: string;
  name: string;
  kind: "project" | "headset";
  enabled: boolean;
  headset?: HeadsetTargetConfig | null;
  project?: { project_id: string; policy: Partial<AutopilotPolicy> } | null;
}
/** One field-level validation error from a 422 (field is dotted, e.g. headset.output_root). */
export interface FieldErrorItem {
  field: string;
  message: string;
}
export interface AutopilotSchedule {
  target_id: string;
  cadence: "daily" | "weekly";
  weekday: number | null;
  time_local: string;
  next_run: string | null;
  last_run: string | null;
  enabled: boolean;
}
export type RunState = "queued" | "running" | "done" | "done_with_failures" | "failed" | "interrupted";
export interface RoundRecord {
  round: number;
  staging: string | null;
  val_macro_f1: number;
  val_gate: boolean;
  val_reasons: string[];
  control_val_deltas: Record<string, number>;
  /** Real rows per label were capped at this many (a real_hurts remedy from the previous round). */
  real_cap?: number | null;
}
export interface PromotionDecision {
  promote: boolean;
  gate_passed: boolean;
  reasons: string[];
  test_macro_f1: number;
  incumbent_test_macro_f1: number | null;
  delta: number | null;
}
export interface HeadRunRecord {
  outcome?: string;
  started?: string;
  ended?: string;
  why?: string;
  /** outcome "failed": the adapter error (the verb and its stderr tail) */
  error?: string;
  path?: string;
  rounds?: RoundRecord[];
  chosen?: RoundRecord;
  carry?: Record<string, number>;
  needs?: string[];
  gate?: { test_macro_f1: number; reference?: { macro_f1: number } | null; decision: PromotionDecision };
}
export interface GroupHead {
  head: string;
  ready: boolean;
  promoted?: boolean;
  encoder?: string | null;
  gate_passed?: boolean;
  test_macro_f1?: number | null;
  baseline_4b_macro_f1?: number | null;
  within_tol?: boolean;
  why?: string;
}
export interface GroupsReport {
  encoder: string;
  groups: { name: string; heads: GroupHead[]; ready: boolean; reasons: string[] }[];
  co_occurrence: Record<string, unknown>;
  shared_encoder_setting: string | null;
  notice: string | null;
}
/** Derived by the backend (forge/autopilot/describe.py); the UI only renders it. */
export interface RunProgress {
  fraction: number;
  heads_done: number;
  heads_total: number;
  current: { label: string; head: string | null } | null;
  /** The adapter's live progress message. */
  detail: string | null;
  elapsed_s: number | null;
  eta_s: number | null;
  eta_note: string | null;
}
export interface TimelineEntry {
  key: string;
  label: string;
  head: string | null;
  phase: string | null;
  state: "done" | "running" | "pending";
  started: string | null;
  ended: string | null;
}
export interface RowCounts {
  generated: number;
  labelled: number;
  excluded: number;
}
export interface RunCounts {
  heads: Record<string, RowCounts>;
  total: RowCounts;
}
/** One head's numbers in a run (describe.results). */
export interface HeadResultRow {
  head: string;
  control_f1: number | null;
  val_f1: number | null;
  final_f1: number | null;
  reference_f1: number | null;
  incumbent_f1: number | null;
  gate_passed: boolean | null;
  outcome: string | null;
  /** promoted | kept | skipped | retired | failed | null (in progress) */
  decision: string | null;
  rounds: number;
}
export interface AutopilotRun {
  id: string;
  target_id: string;
  trigger: "schedule" | "manual" | "data";
  state: RunState;
  step_cursor: string | null;
  round: number;
  result: {
    steps?: Record<string, unknown>;
    heads?: Record<string, HeadRunRecord>;
    groups?: GroupsReport;
    error?: string;
    plan?: { heads: string[]; phases: string[]; started_at: string };
    timings?: Record<string, { s: string; e: string | null }>;
    encoder?: string;
    [k: string]: unknown;
  };
  /** When this execution began (a resumed run moves it on). */
  started: string | null;
  /** When the run first began; kept across resumes. */
  first_started?: string | null;
  ended: string | null;
  created: string;
  /** null for a run started before progress was recorded. */
  progress?: RunProgress | null;
  timeline?: TimelineEntry[];
  counts?: RunCounts;
  results?: HeadResultRow[];
  /** Name of the encoder the run trained on. */
  encoder?: string | null;
  target_name?: string | null;
  target_removed?: boolean;
}
export interface AutopilotTarget {
  id: string;
  name: string;
  kind: "project" | "headset";
  enabled: boolean;
  /** Display name of a head-set target's encoder. */
  encoder_name?: string | null;
  /** Where a head-set target would write on each encoder (a derived folder follows the encoder, a hand-set one
   * stays); computed by the backend, which applies the same rule on save. */
  output_roots?: Record<string, string> | null;
  headset?: HeadsetTargetConfig | null;
  project?: { project_id: string; policy: AutopilotPolicy } | null;
  schedule: AutopilotSchedule | null;
  last_run: AutopilotRun | null;
  /** One-time actions the user approved that wait for a run (e.g. resplit_val: the next scheduled or manual run). */
  pending_actions?: PendingAction[];
}
export interface PendingAction {
  id: number;
  action: string;
  approved: string;
  reason: string;
  queued: string;
  last_error: string | null;
}
export interface JournalEntry {
  id: number;
  run_id: string | null;
  target_id: string;
  head: string | null;
  round: number | null;
  action: string;
  reason: string;
  counts: Record<string, unknown>;
  ts: string;
}
export interface ModelJob {
  id: number;
  kind: string;
  owner: string;
  state: "queued" | "deferred" | "running" | "done" | "failed" | "cancelled" | "interrupted";
  enqueued: string;
  started: string | null;
  ended: string | null;
  error: string | null;
}
export interface QueueSnapshot {
  active: ModelJob[];
  recent: ModelJob[];
  autopilot_current: string | null;
}
export interface GroupsEntry {
  target: string;
  run: string | null;
  computed_at: string | null;
  report: GroupsReport | null;
}
export interface ProvenanceDoc {
  target: string;
  path: string;
  provenance: Record<string, unknown> & { class?: string };
}
