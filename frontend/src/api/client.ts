import type {
  AutopilotRun,
  AutopilotSchedule,
  AutopilotTarget,
  GroupsEntry,
  JournalEntry,
  ProvenanceDoc,
  QueueSnapshot,
  DatasetPage,
  EncoderRecommendationDetail,
  EncodersView,
  HeadsetPreset,
  PresetHead,
  TargetBody,
  EvalReport,
  Example,
  Health,
  InferResult,
  LabelActionResult,
  LabelPolicy,
  LabelQueuePage,
  LabelSetSummary,
  QueueFilter,
  PackageManifest,
  ProjectDetail,
  ProjectSummary,
  RankItem,
  Settings,
  TaskSchema, ProviderInfo } from "./types";

export class ApiError extends Error {
  status: number;
  /** The parsed JSON error body (e.g. a target 422's `{errors: [{field, message}]}`), when there was one. */
  body: unknown;
  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

/* Forge accepts a change (POST/PUT/PATCH/DELETE) only from the Forge page with the running server's session
 * (backend forge/web_guard.py): the page fetches it once from the same-origin GET /api/session and sends it in
 * X-Forge-Session; after a server restart the old one is refused and the page fetches the new one, once. */
const SESSION_HEADER = "X-Forge-Session";
let session: Promise<string> | null = null;

async function sessionToken(refresh = false): Promise<string> {
  if (!session || refresh) {
    session = fetch("/api/session", { credentials: "same-origin" }).then(async (r) => {
      const j = (await r.json().catch(() => null)) as { token?: string; detail?: string } | null;
      if (!r.ok || !j?.token) throw new ApiError(r.status, j?.detail ?? `${r.status} could not open a Forge session`, j);
      return j.token;
    });
    session.catch(() => {
      session = null;
    });
  }
  return session;
}

/** Forget the cached session (tests). */
export function resetSession(): void {
  session = null;
}

async function send(method: string, url: string, body: unknown, refresh: boolean): Promise<Response> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET" && method !== "HEAD") headers[SESSION_HEADER] = await sessionToken(refresh);
  return fetch(url, {
    method,
    credentials: "same-origin",
    headers: Object.keys(headers).length ? headers : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  let res = await send(method, url, body, false);
  if (res.status === 403 && method !== "GET") {
    const d = (await res.clone().json().catch(() => null)) as { detail?: string } | null;
    if (d?.detail && /session/i.test(d.detail)) res = await send(method, url, body, true);
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    let j: unknown = null;
    try {
      j = await res.json();
      const d = (j as { detail?: unknown } | null)?.detail;
      if (typeof d === "string") msg = d;
      else if (d) msg = JSON.stringify(d);
    } catch {
      /* body not json */
    }
    throw new ApiError(res.status, msg, j);
  }
  const ct = res.headers.get("content-type") ?? "";
  if (res.status === 202 || res.status === 204) {
    return (ct.includes("json") ? await res.json().catch(() => undefined) : undefined) as T;
  }
  if (ct.includes("json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

export interface CreateProjectBody {
  name?: string;
  repo?: string;
  task?: string;
  provider?: string;
  model?: string;
  /** Registry key the experiment trains on (required: there is no app-wide default). */
  encoder: string;
}

export interface AcceptBody {
  opportunity_id: string;
  overrides?: { task?: TaskSchema; target_examples?: number; success_threshold?: number; label_policy?: LabelPolicy };
}

export interface DatasetQuery {
  split?: string;
  label?: string;
  source?: string;
  q?: string;
  offset?: number;
  limit?: number;
}

const enc = encodeURIComponent;

export interface QueueQuery {
  head?: string;
  state?: QueueFilter;
  offset?: number;
  limit?: number;
}

export const api = {
  health: () => req<Health>("GET", "/api/health"),
  providers: (refresh = false) => req<ProviderInfo[]>("GET", `/api/providers${refresh ? "?refresh=true" : ""}`),
  settings: () => req<Settings>("GET", "/api/settings"),
  saveSettings: (s: Settings) => req<Settings>("PUT", "/api/settings", s),
  encoders: () => req<EncodersView>("GET", "/api/encoders"),
  /** The encoder Forge recommends for a new experiment (before its task is known). */
  encoderRecommendation: () => req<EncoderRecommendationDetail>("GET", "/api/encoders/recommendation"),
  projects: () => req<ProjectSummary[]>("GET", "/api/projects"),
  createProject: (b: CreateProjectBody) => req<ProjectSummary>("POST", "/api/projects", b),
  project: (pid: string) => req<ProjectDetail>("GET", `/api/projects/${enc(pid)}`),
  deleteProject: (pid: string) => req<void>("DELETE", `/api/projects/${enc(pid)}`),
  /** Train the project on registry encoder `key` from its next selecting / training stage (409 while a job runs). */
  setProjectEncoder: (pid: string, key: string) => req<ProjectDetail>("PUT", `/api/projects/${enc(pid)}/encoder`, { key }),
  chat: (pid: string, message: string) => req<void>("POST", `/api/projects/${enc(pid)}/chat`, { message }),
  accept: (pid: string, b: AcceptBody) => req<void>("POST", `/api/projects/${enc(pid)}/accept`, b),
  revise: (pid: string, feedback: string) => req<void>("POST", `/api/projects/${enc(pid)}/revise`, { feedback }),
  cancel: (pid: string) => req<void>("POST", `/api/projects/${enc(pid)}/cancel`),
  retry: (pid: string) => req<void>("POST", `/api/projects/${enc(pid)}/retry`),
  dataset: (pid: string, q: DatasetQuery) => {
    const p = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => {
      if (v !== undefined && v !== "" && v !== null) p.set(k, String(v));
    });
    return req<DatasetPage>("GET", `/api/projects/${enc(pid)}/dataset?${p}`);
  },
  patchExample: (pid: string, eid: string, b: Partial<Pick<Example, "label" | "split" | "state">>) =>
    req<Example>("PATCH", `/api/projects/${enc(pid)}/dataset/${enc(eid)}`, b),
  deleteExample: (pid: string, eid: string) => req<void>("DELETE", `/api/projects/${enc(pid)}/dataset/${enc(eid)}`),
  eval: (pid: string) => req<EvalReport>("GET", `/api/projects/${enc(pid)}/eval`),
  infer: (pid: string, state: string, model: "finetuned" | "reference") =>
    req<InferResult>("POST", `/api/projects/${enc(pid)}/infer`, { state, model }),
  rank: (pid: string, state: string, candidates: string[], instructions?: string) =>
    req<RankItem[]>("POST", `/api/projects/${enc(pid)}/rank`, { state, candidates, instructions }),
  package: (pid: string) => req<PackageManifest>("GET", `/api/projects/${enc(pid)}/package`),
  packageFile: (pid: string, path: string) =>
    req<string>("GET", `/api/projects/${enc(pid)}/package/file?path=${enc(path)}`),
  packageDownloadUrl: (pid: string) => `/api/projects/${enc(pid)}/package/download`,
  playgroundInfer: (state: string, task: TaskSchema) =>
    req<InferResult>("POST", "/api/playground/infer", { state, task }),
  // labelling queues: lsid = p:<experiment> | h:<head>
  labelsets: () => req<LabelSetSummary[]>("GET", "/api/labelsets"),
  labelQueue: (lsid: string, q: QueueQuery = {}) => {
    const p = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => {
      if (v !== undefined && v !== "" && v !== null) p.set(k, String(v));
    });
    return req<LabelQueuePage>("GET", `/api/labelsets/${enc(lsid)}/queue?${p}`);
  },
  labelRow: (lsid: string, rid: string, b: { label: string } | { skip: true }) =>
    req<LabelActionResult>("POST", `/api/labelsets/${enc(lsid)}/rows/${enc(rid)}`, b),
  labelUndo: (lsid: string) => req<LabelActionResult>("POST", `/api/labelsets/${enc(lsid)}/undo`),
  setLabelPolicy: (lsid: string, policy: LabelPolicy) => req<{ policy: LabelPolicy }>("PUT", `/api/labelsets/${enc(lsid)}/policy`, { policy }),
  labelTeacher: (lsid: string, mode: "label" | "suggest" = "label") =>
    req<{ rows: number }>("POST", `/api/labelsets/${enc(lsid)}/teacher`, { mode }),
  /** Leave the opt-in label_review gate (review_done). */
  reviewLabels: (pid: string, mode: "continue" | "exclude") =>
    req<{ ok: boolean }>("POST", `/api/projects/${enc(pid)}/labels/review`, { mode }),
  /** Record a user label in any state; the next labelling stage applies it (label_override). */
  overrideLabel: (pid: string, rid: string, label: string) =>
    req<{ pending: { rid: string; label: string }[] }>("POST", `/api/projects/${enc(pid)}/labels/override`, { rid, label }),
  // autopilot
  autopilotTargets: () => req<AutopilotTarget[]>("GET", "/api/autopilot/targets"),
  autopilotTarget: (tid: string) => req<AutopilotTarget>("GET", `/api/autopilot/targets/${enc(tid)}`),
  createTarget: (b: TargetBody) => req<AutopilotTarget>("POST", "/api/autopilot/targets", b),
  putTarget: (tid: string, b: TargetBody) => req<AutopilotTarget>("PUT", `/api/autopilot/targets/${enc(tid)}`, b),
  setTargetEnabled: (tid: string, enabled: boolean) => req<AutopilotTarget>("PATCH", `/api/autopilot/targets/${enc(tid)}/enabled`, { enabled }),
  deleteTarget: (tid: string) => req<{ ok: boolean }>("DELETE", `/api/autopilot/targets/${enc(tid)}`),
  headsets: () => req<HeadsetPreset[]>("GET", "/api/autopilot/headsets"),
  headsetHeads: (preset: string, encoder: string) =>
    req<PresetHead[]>("GET", `/api/autopilot/headsets/${enc(preset)}/heads?encoder=${enc(encoder)}`),
  saveSchedule: (tid: string, b: { cadence: "daily" | "weekly"; weekday: number | null; time_local: string; enabled: boolean }) =>
    req<AutopilotSchedule>("PUT", `/api/autopilot/targets/${enc(tid)}/schedule`, b),
  runTarget: (tid: string) => req<AutopilotRun>("POST", `/api/autopilot/targets/${enc(tid)}/run`),
  autopilotRuns: (target?: string) => req<AutopilotRun[]>("GET", `/api/autopilot/runs${target ? `?target=${enc(target)}` : ""}`),
  autopilotRun: (rid: string) => req<AutopilotRun>("GET", `/api/autopilot/runs/${enc(rid)}`),
  journal: (q: { target?: string; head?: string; action?: string; run?: string; limit?: number } = {}) => {
    const p = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => {
      if (v !== undefined && v !== "" && v !== null) p.set(k, String(v));
    });
    return req<JournalEntry[]>("GET", `/api/autopilot/journal?${p}`);
  },
  modelQueue: () => req<QueueSnapshot>("GET", "/api/autopilot/queue"),
  groups: () => req<GroupsEntry[]>("GET", "/api/autopilot/groups"),
  provenance: (head: string, target?: string) =>
    req<ProvenanceDoc>("GET", `/api/autopilot/heads/${enc(head)}/provenance${target ? `?target=${enc(target)}` : ""}`),
};
