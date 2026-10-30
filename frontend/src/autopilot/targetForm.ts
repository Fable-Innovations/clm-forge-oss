/* The autopilot target form's state, pure: a target (or a preset) -> form, form -> request body, a 422 -> errors
 * per field. The dialog (TargetDialog.tsx) only renders this. */
import { ApiError } from "../api/client";
import type { AutopilotPolicy, AutopilotTarget, DataSource, FieldErrorItem, HeadsetPreset, HeadsetTargetConfig, PresetSource, TargetBody } from "../api/types";

/** The policy numbers the form edits; the rest of the policy round-trips untouched. */
export const POLICY_FIELDS: { key: keyof AutopilotPolicy & string; label: string; step?: number; hint: string }[] = [
  { key: "min_count", label: "Minimum rows per label", hint: "Fewer train rows than this in a label group is a need." },
  { key: "gen_cap", label: "Generated rows per round", hint: "The rest carries over to the next round or run." },
  { key: "max_rounds", label: "Rounds per run", hint: "1 to 10." },
  { key: "attempts", label: "Attempts per round", hint: "Seeds trained per round; the best on val is kept (1 to 10)." },
  { key: "exemplars", label: "Real exemplars per prompt", hint: "Scrubbed real rows shown to the generator (1 to 8)." },
  { key: "tol", label: "Promotion tolerance", step: 0.005, hint: "A candidate may trail the incumbent's test macro-F1 by this much." },
];

/** A source's label: the preset's, else its kind (a target registered by hand, or a kind its preset no longer offers). */
export const sourceLabel = (kind: string, p: HeadsetPreset | undefined) => p?.sources.find((s) => s.kind === kind)?.label ?? kind;

/** A setting key in words: `stage` -> "Stage", `min_age_days` -> "Min age days". */
export const settingLabel = (key: string) => {
  const w = key.replace(/[_-]+/g, " ").trim();
  return w.charAt(0).toUpperCase() + w.slice(1);
};

/** What the user confirms before turning a source (or one of its setting values) off: at the target's next run the
 * autopilot removes the stored real rows it supplied (backend: real_rows.ingest(enabled=...), removal kind
 * source_disabled). */
export function disableSourceWarning(label: string, value?: string): string {
  const what = value ? `${label} (${value})` : label;
  return (
    `Turn off ${what}? At this target's next run the autopilot removes every stored real row that came ` +
    `from it, from the shared dataset: their labels, the rows generated from them and the unpromoted heads trained on ` +
    `them. A promoted head that learned from them is retrained. Every removal is journalled and counted. Turning it ` +
    `back on pulls its rows again.`
  );
}

/** A preset source's settings before the user changes them (each setting's default). */
export const defaultSettings = (src: PresetSource): Record<string, string | string[]> =>
  Object.fromEntries(
    Object.entries(src.settings).map(([k, st]) => [
      k,
      st.multiple ? (Array.isArray(st.default) ? [...st.default] : st.default ? [st.default] : []) : Array.isArray(st.default) ? (st.default[0] ?? st.values[0] ?? "") : st.default,
    ]),
  );

export interface TargetForm {
  /** Set when editing; on create it is derived from the name until the user edits it. */
  id: string;
  idEdited: boolean;
  name: string;
  kind: "project" | "headset";
  enabled: boolean;
  projectId: string;
  preset: string;
  /** Empty = every head the adapter reports. */
  heads: string[];
  encoder: string;
  outputRoot: string;
  /** False while the output folder follows the encoder. */
  outputRootEdited: boolean;
  sources: DataSource[];
  policy: Partial<AutopilotPolicy>;
  /** The stored head-set config (edit) the body is built on, so fields the form doesn't show round-trip. */
  base: HeadsetTargetConfig | null;
  /** An edited target's output folder per encoder (from the backend). */
  outputRoots: Record<string, string> | null;
}

/** The id a new target gets from its name (the backend's targets.id_from_name). */
export const idFromName = (name: string) =>
  name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "")
    .slice(0, 64);

/** One row per source the preset offers: the target's own entry (settings it lacks filled from the defaults), else a
 * disabled one with the default settings. */
export function sourcesFor(offered: PresetSource[], have: DataSource[]): DataSource[] {
  return offered.map((src) => {
    const own = have.find((s) => s.kind === src.kind);
    return own ? { ...own, settings: { ...defaultSettings(src), ...own.settings } } : { kind: src.kind, enabled: false, settings: defaultSettings(src) };
  });
}

/** A new target's form. The encoder starts empty: a head-set preset fills in its recommendation (withPreset); there is
 * no app-wide default encoder. */
export function emptyForm(): TargetForm {
  return {
    id: "",
    idEdited: false,
    name: "",
    kind: "headset",
    enabled: true,
    projectId: "",
    preset: "",
    heads: [],
    encoder: "",
    outputRoot: "",
    outputRootEdited: false,
    sources: [],
    policy: {},
    base: null,
    outputRoots: null,
  };
}

export function toForm(t: AutopilotTarget): TargetForm {
  const h = t.headset ?? null;
  return {
    id: t.id,
    idEdited: true,
    name: t.name,
    kind: t.kind,
    enabled: t.enabled,
    projectId: t.project?.project_id ?? "",
    preset: h?.preset ?? "",
    heads: [...(h?.heads ?? [])],
    encoder: h?.encoder ?? "",
    outputRoot: h?.output_root ?? "",
    outputRootEdited: false,
    sources: (h?.sources ?? []).map((s) => ({ ...s, settings: { ...s.settings } })),
    policy: { ...(h?.policy ?? t.project?.policy ?? {}) },
    base: h,
    outputRoots: t.output_roots ?? null,
  };
}

/** `path` is `root` or lies under it (both absolute; trailing slashes ignored). */
const isInside = (path: string, root: string) => {
  const a = path.replace(/\/+$/, ""), r = root.replace(/\/+$/, "");
  return a === r || a.startsWith(`${r}/`);
};

/** The encoders a preset offers whose derived output folder it can save (outside every protected folder). */
export function savableEncoders(p: HeadsetPreset): string[] {
  return p.encoders.filter((k) => {
    const root = p.output_roots[k];
    return !!root && !p.protected_roots.some((r) => isInside(root, r));
  });
}

/** Apply a preset to a head-set form: its sources (each one it offers, on when the preset enables it by default), fixed
 * heads, and an encoder it can train on and save: the current one when the preset allows it, else the preset's
 * recommendation (forge.encoder_choice: the encoder the head set is built for, when it fits this machine), else none
 * (the user must pick one). */
export function withPreset(f: TargetForm, p: HeadsetPreset): TargetForm {
  const rec = p.recommendation?.key;
  const encoder = p.encoders.includes(f.encoder) ? f.encoder : rec && savableEncoders(p).includes(rec) ? rec : "";
  return {
    ...f,
    preset: p.key,
    heads: p.fixed_heads ? [...p.fixed_heads] : f.preset === p.key ? f.heads : [],
    encoder,
    sources: sourcesFor(
      p.sources,
      f.sources.length ? f.sources : p.sources.map((src) => ({ kind: src.kind, enabled: src.enabled_by_default, settings: defaultSettings(src) })),
    ),
  };
}

/** The output folder the form shows (and sends): the user's own; else, for a new target, the one its preset derives
 * for the encoder; else the stored target's folder for that encoder as the backend reports it (a derived folder
 * follows the encoder, a hand-set one stays). */
export function shownOutputRoot(f: TargetForm, p: HeadsetPreset | undefined): string {
  if (f.outputRootEdited) return f.outputRoot;
  if (!f.base) return (p && f.encoder && p.output_roots[f.encoder]) || f.outputRoot;
  return f.outputRoots?.[f.encoder] ?? f.base.output_root;
}

export function toBody(f: TargetForm, p: HeadsetPreset | undefined): TargetBody {
  const id = f.idEdited ? f.id.trim() : idFromName(f.name);
  if (f.kind === "project") {
    return { id, name: f.name.trim(), kind: "project", enabled: f.enabled, headset: null, project: { project_id: f.projectId, policy: f.policy } };
  }
  // only what the form edits: the server fills the rest (adapter command, dataset folder, protected roots) from the
  // preset and refuses a body that names any of those (commands never come from a request)
  const headset = {
    ...(f.base?.adapter_config ? { adapter_config: { ...f.base.adapter_config } } : {}),
    preset: f.preset || null,
    encoder: f.encoder,
    output_root: shownOutputRoot(f, p),
    heads: f.heads,
    sources: f.sources,
    policy: { ...(f.base?.policy ?? {}), ...f.policy },
  } as HeadsetTargetConfig;
  return { id, name: f.name.trim(), kind: "headset", enabled: f.enabled, headset, project: null };
}

/** A 422's `{errors: [{field, message}]}` -> {field: message}; anything else under "" (shown above the form). */
export function errorsByField(err: unknown): Record<string, string> {
  if (!err) return {};
  if (err instanceof ApiError && err.status === 422) {
    const items = (err.body as { errors?: FieldErrorItem[] } | null)?.errors;
    if (Array.isArray(items) && items.length) {
      const out: Record<string, string> = {};
      for (const e of items) out[e.field] = out[e.field] ? `${out[e.field]}; ${e.message}` : e.message;
      return out;
    }
  }
  return { "": err instanceof Error ? err.message : String(err) };
}
