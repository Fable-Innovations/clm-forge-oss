/* Test-only fixtures: a small encoder registry, one example head-set preset ("support-ticket-heads") with a target
 * made from it, two autopilot runs, and a stubFetch route that serves them. Never imported by app code. */
import type {
  AutopilotPolicy,
  AutopilotRun,
  AutopilotTarget,
  Encoder,
  EncodersView,
  HeadResultRow,
  HeadsetPreset,
  HeadsetTargetConfig,
  JournalEntry,
  TargetBody,
  TimelineEntry,
} from "../api/types";
import type { Call, Route } from "./stubFetch";

const ago = (min: number) => new Date(Date.now() - min * 60_000).toISOString();

// ---- encoder registry ------------------------------------------------------------------------------------------
const reg = (key: string, name: string, backend: "mlx" | "llamacpp", repo: string, quant: string, hidden: number, gb: number | null, cached: boolean, file: string | null = null): Encoder => ({
  key,
  name,
  backend,
  repo,
  file,
  revision: "0".repeat(40),
  hidden,
  quant,
  approx_gb: gb,
  approx_gb_source: gb == null ? null : "measured peak",
  note: "",
  spec: backend === "mlx" ? `mlx:${repo}` : `llamacpp:${repo}:${file}`,
  url: `https://huggingface.co/${repo}`,
  cached,
  used_by: { projects: [], targets: [] },
});
export const ENCODERS: Encoder[] = [
  reg("qwen3-4b", "Qwen3-4B (MLX 4-bit)", "mlx", "mlx-community/Qwen3-4B-4bit", "4-bit", 2560, 4.21, true),
  reg("qwen3-0.6b", "Qwen3-0.6B (MLX 4-bit)", "mlx", "mlx-community/Qwen3-0.6B-4bit", "4-bit", 1024, 2.01, true),
  reg("qwen3-0.6b-8bit", "Qwen3-0.6B (MLX 8-bit)", "mlx", "mlx-community/Qwen3-0.6B-8bit", "8-bit", 1024, 2.34, true),
  reg("qwen3-0.6b-bf16", "Qwen3-0.6B (MLX bf16)", "mlx", "mlx-community/Qwen3-0.6B-bf16", "bf16", 1024, 2.97, true),
  reg("qwen3-4b-gguf-q4km", "Qwen3-4B (GGUF Q4_K_M)", "llamacpp", "Qwen/Qwen3-4B-GGUF", "Q4_K_M", 2560, 4.09, true, "Qwen3-4B-Q4_K_M.gguf"),
  reg("example-12b-gguf-q8", "Example-12B", "llamacpp", "example-org/Example-12B-GGUF", "Q8_0", 3840, null, false, "Example-12B-Q8_0.gguf"),
];
const encoderName = (key: string | null | undefined) => ENCODERS.find((e) => e.key === key)?.name ?? null;

/** GET /api/encoders, with the given users per encoder key. */
export function encodersView(usedBy: Record<string, Encoder["used_by"]> = {}): EncodersView {
  return { encoders: ENCODERS.map((e) => ({ ...e, used_by: usedBy[e.key] ?? { projects: [], targets: [] } })) };
}

/** GET /api/encoders/recommendation for a new experiment. */
export const EXPERIMENT_RECOMMENDATION = {
  key: "qwen3-4b-gguf-q4km",
  name: "Qwen3-4B (GGUF Q4_K_M)",
  profile: "general" as const,
  reason: "Qwen3-4B (GGUF Q4_K_M): 2560-d is wide enough for a new experiment before its task is known; its measured peak ~4.1 GB fits the 28.8 GB model budget",
  budget_gb: 28.8,
  considered: [],
};

// ---- the example head set -------------------------------------------------------------------------------------
export const HEADS_BASE = "/data/heads/support-ticket-heads";
export const HEADS = ["ticket-escalation", "ticket-language", "ticket-priority", "ticket-queue", "ticket-sentiment"];
const MLX = ENCODERS.filter((e) => e.backend === "mlx").map((e) => e.key);
const outputRoot = (key: string) => `${HEADS_BASE}-${key}`;

export const PRESET: HeadsetPreset = {
  key: "support-ticket-heads",
  name: "Support ticket heads",
  description: "Example head set: queue, priority, sentiment, language and escalation heads for support tickets.",
  dataset_root: "/data/support-ticket-data",
  output_base: HEADS_BASE,
  protected_roots: [HEADS_BASE],
  fixed_heads: null,
  encoders: MLX,
  output_roots: Object.fromEntries(MLX.map((k) => [k, outputRoot(k)])),
  recommendation: { key: "qwen3-4b", name: "Qwen3-4B (MLX 4-bit)", reason: "Qwen3-4B (MLX 4-bit): the encoder this head set is built for" },
  sources: [
    {
      kind: "event_log",
      label: "Event log",
      enabled_by_default: true,
      settings: { stage: { values: ["qa", "prod", "dev"], default: "qa", multiple: false } },
    },
    {
      kind: "shared_export",
      label: "Shared export",
      enabled_by_default: true,
      settings: { stages: { values: ["qa", "prod"], default: ["qa", "prod"], multiple: true } },
    },
  ],
};

const POLICY: AutopilotPolicy = { min_count: 150, gen_cap: 600, max_rounds: 3, attempts: 3, exemplars: 4, adjudicate: true, tol: 0.01, tol_per_head: {} };

const headset = (encoder: string): HeadsetTargetConfig => ({
  preset: PRESET.key,
  adapter: ["python", "-m", "example_adapter"],
  cwd: "/data",
  env: {},
  dataset_root: PRESET.dataset_root,
  encoder,
  output_root: outputRoot(encoder),
  heads: [],
  protected_roots: [HEADS_BASE],
  protected_globs: [],
  protected_files: [],
  sources: [
    { kind: "event_log", enabled: true, settings: { stage: "qa" } },
    { kind: "shared_export", enabled: true, settings: { stages: ["qa", "prod"] } },
  ],
  adapter_config: {},
  policy: POLICY,
});

/** A head-set target as GET /api/autopilot/targets/:id returns it. */
export function target(encoder = "qwen3-0.6b-bf16"): AutopilotTarget {
  return {
    id: "support-ticket-heads",
    name: "Support ticket heads",
    kind: "headset",
    enabled: true,
    encoder_name: encoderName(encoder),
    output_roots: Object.fromEntries(ENCODERS.map((e) => [e.key, outputRoot(e.key)])),
    headset: headset(encoder),
    project: null,
    schedule: { target_id: "support-ticket-heads", cadence: "daily", weekday: null, time_local: "03:00", next_run: ago(-600), last_run: null, enabled: true },
    last_run: null,
  };
}

/** What POST / PUT /api/autopilot/targets answer: the body as a stored target. */
function targetView(b: TargetBody, id: string): AutopilotTarget {
  const t = target(b.headset?.encoder);
  return { ...t, id, name: b.name, kind: b.kind, enabled: b.enabled, headset: b.headset ? { ...t.headset!, ...b.headset } : null, project: null, schedule: null };
}

// ---- runs ------------------------------------------------------------------------------------------------------
const step = (key: string, label: string, head: string | null, phase: string | null, s: number, e: number | null): TimelineEntry => ({
  key,
  label,
  head,
  phase,
  state: e == null ? "running" : "done",
  started: ago(s),
  ended: e == null ? null : ago(e),
});
const pending = (key: string, label: string, head: string | null, phase: string | null): TimelineEntry => ({ key, label, head, phase, state: "pending", started: null, ended: null });

/** A run 3 heads in: ticket-language generating rows, ticket-escalation still to come. */
export const RUNNING_RUN: AutopilotRun = (() => {
  const done: [string, string, number, number, number][] = [
    ["ticket-priority", "promoted", 0.86, 0.847, 0.794],
    ["ticket-queue", "promoted", 0.93, 0.955, 0.944],
    ["ticket-sentiment", "rejected", 0.69, 0.706, 0.787],
  ];
  const timeline: TimelineEntry[] = [step("pull", "Pulling real data", null, null, 175, 174), step("ingest", "Ingesting real rows", null, null, 174, 173)];
  let t = 172;
  for (const [h, o] of done) {
    timeline.push(step(`gen:${h}:1:0`, `Generating rows for ${h} (round 1)`, h, "generate", t, t - 20));
    timeline.push(step(`train:${h}:1`, `Training ${h} (round 1)`, h, "train", t - 20, t - 40));
    if (o === "promoted") timeline.push(step(`promote:${h}`, `Promoting ${h}`, h, "promote", t - 40, t - 40));
    t -= 41;
  }
  timeline.push(step("gen:ticket-language:1:0", "Generating rows for ticket-language (round 1)", "ticket-language", "generate", 2, null));
  timeline.push(pending("train:ticket-language", "Training ticket-language", "ticket-language", "train"));
  timeline.push(pending("train:ticket-escalation", "Training ticket-escalation", "ticket-escalation", "train"));
  timeline.push(pending("promote:ticket-escalation", "Promoting ticket-escalation", "ticket-escalation", "promote"));
  const results: HeadResultRow[] = done.map(([h, o, c, f, inc]) => ({
    head: h,
    control_f1: c,
    val_f1: f + 0.02,
    final_f1: f,
    reference_f1: null,
    incumbent_f1: inc,
    gate_passed: f >= 0.75,
    outcome: o,
    decision: o === "promoted" ? "promoted" : "kept",
    rounds: 1,
  }));
  return {
    id: "run-running",
    target_id: "support-ticket-heads",
    trigger: "manual",
    state: "running",
    step_cursor: "gen:ticket-language:1:0",
    round: 1,
    created: ago(175),
    started: ago(175),
    first_started: ago(175),
    ended: null,
    result: { plan: { heads: [...done.map((d) => d[0]), "ticket-language", "ticket-escalation"], phases: ["generate", "train", "promote"], started_at: ago(175) }, encoder: "qwen3-0.6b-bf16" },
    progress: {
      fraction: 0.6,
      heads_done: 3,
      heads_total: 5,
      current: { label: "Generating rows for ticket-language (round 1)", head: "ticket-language" },
      detail: "generating 92 rows for ticket-language",
      elapsed_s: 175 * 60,
      eta_s: 26 * 60,
      eta_note: null,
    },
    timeline,
    counts: { heads: {}, total: { generated: 4800, labelled: 312, excluded: 27 } },
    results,
    encoder: "Qwen3-0.6B (MLX bf16)",
    target_name: "Support ticket heads",
    target_removed: false,
  };
})();

/** A finished run from before progress was recorded. */
export const OLD_RUN: AutopilotRun = {
  id: "run-old",
  target_id: "support-ticket-heads",
  trigger: "schedule",
  state: "done",
  step_cursor: null,
  round: 1,
  created: ago(6000),
  started: ago(6000),
  ended: ago(5900),
  result: { heads: { "ticket-queue": { outcome: "promoted", rounds: [] } } },
  progress: null,
  timeline: [],
  counts: { heads: {}, total: { generated: 588, labelled: 40, excluded: 2 } },
  results: [{ head: "ticket-queue", control_f1: 0.84, val_f1: 0.9, final_f1: 0.9, reference_f1: null, incumbent_f1: 0.895, gate_passed: true, outcome: "promoted", decision: "promoted", rounds: 1 }],
  encoder: "Qwen3-0.6B (MLX bf16)",
  target_name: "Support ticket heads",
  target_removed: false,
};

const JOURNAL: JournalEntry[] = [
  { id: 2, run_id: RUNNING_RUN.id, target_id: "support-ticket-heads", head: "ticket-queue", round: 1, action: "promote", reason: "promoted ticket-queue", counts: {}, ts: ago(90) },
  { id: 1, run_id: RUNNING_RUN.id, target_id: "support-ticket-heads", head: null, round: null, action: "run_started", reason: "manual run", counts: {}, ts: ago(175) },
];

/** A stubFetch route over the fixtures above: presets, heads, targets (create / replace answer the body), runs, journal. */
export function autopilotRoute(presets: HeadsetPreset[] = [PRESET]): Route {
  return (c: Call) => {
    const path = c.url.pathname;
    if (path === "/api/encoders") return [200, encodersView()];
    if (path === "/api/projects" && c.method === "GET") return [200, []];
    if (path === "/api/autopilot/headsets") return [200, presets];
    const hm = path.match(/^\/api\/autopilot\/headsets\/([^/]+)\/heads$/);
    if (hm) return [200, HEADS.map((h) => ({ head: h, group: null, retired: null }))];
    if (path === "/api/autopilot/targets" && c.method === "POST") {
      const b = c.body as TargetBody;
      return [201, targetView(b, b.id ?? b.name)];
    }
    const tm = path.match(/^\/api\/autopilot\/targets\/([^/]+)$/);
    if (tm && c.method === "PUT") return [200, targetView(c.body as TargetBody, decodeURIComponent(tm[1]))];
    if (tm && c.method === "GET") return [200, target()];
    const rm = path.match(/^\/api\/autopilot\/runs\/(.+)$/);
    if (rm) {
      const r = [RUNNING_RUN, OLD_RUN].find((x) => x.id === decodeURIComponent(rm[1]));
      return r ? [200, r] : [404, { detail: "no run" }];
    }
    if (path === "/api/autopilot/journal") {
      const run = c.url.searchParams.get("run");
      return [200, JOURNAL.filter((e) => !run || e.run_id === run)];
    }
    return undefined;
  };
}
