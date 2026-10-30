import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check } from "lucide-react";
import { api } from "../api/client";
import { qk, useHeadsets, useProjects } from "../api/queries";
import type { AutopilotTarget, DataSource, PresetSource } from "../api/types";
import { EncoderSelect } from "../components/EncoderSelect";
import { Button, Dialog, Field, Input, Segmented, Select, Spinner } from "../components/ui";
import { cn } from "../lib/format";
import { ScheduleEditor } from "./ScheduleEditor";
import {
  disableSourceWarning,
  emptyForm,
  errorsByField,
  idFromName,
  POLICY_FIELDS,
  settingLabel,
  shownOutputRoot,
  sourceLabel,
  toBody,
  toForm,
  withPreset,
  type TargetForm,
} from "./targetForm";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2.5 pt-3 first:pt-0 border-t first:border-t-0 border-line-soft" aria-label={`${title} section`}>
      <h3 className="text-[11.5px] uppercase tracking-wide text-muted">{title}</h3>
      {children}
    </section>
  );
}

/** A form field with its server-side error (a 422's `errors[field]`) shown under it. */
function FormField({ label, error, hint, children, className }: { label: ReactNode; error?: string; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Field label={label} className={className} hint={error ? <span className="text-bad" role="alert">{error}</span> : hint}>
      {children}
    </Field>
  );
}

/**
 * Create or edit an autopilot target: kind, what it trains (a project, or a head-set preset and its heads), the
 * encoder and output folder, the real-data sources, the schedule and the policy. Field errors from the server
 * appear next to their field.
 */
export function TargetDialog({ open, onClose, target }: { open: boolean; onClose: () => void; target?: AutopilotTarget }) {
  const qc = useQueryClient();
  const presets = useHeadsets();
  const projects = useProjects();
  const [current, setCurrent] = useState<AutopilotTarget | undefined>(target);
  const [f, setF] = useState<TargetForm>(() => (target ? toForm(target) : emptyForm()));
  const [created, setCreated] = useState(false);
  useEffect(() => {
    if (!open) return;
    setCurrent(target);
    setCreated(false);
    setF(target ? toForm(target) : emptyForm());
    // a new dialog starts from the target (or the defaults) once; later refetches must not wipe the user's edits
  }, [open, target?.id]);
  const editing = !!current;
  const preset = presets.data?.find((p) => p.key === f.preset);
  // a new head-set form starts on the first preset
  useEffect(() => {
    if (open && !editing && f.kind === "headset" && !f.preset && presets.data?.length) setF((x) => withPreset(x, presets.data![0]));
  }, [open, editing, f.kind, f.preset, presets.data]);

  const heads = useQuery({
    queryKey: qk.headsetHeads(f.preset, f.encoder),
    queryFn: () => api.headsetHeads(f.preset, f.encoder),
    enabled: open && f.kind === "headset" && !!preset && !preset.fixed_heads && !!f.encoder,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const project = useQuery({
    queryKey: qk.project(f.projectId),
    queryFn: () => api.project(f.projectId),
    enabled: open && f.kind === "project" && !!f.projectId,
  });
  const [projectEncoder, setProjectEncoder] = useState<string | null>(null);
  useEffect(() => setProjectEncoder(project.data?.encoder?.key ?? null), [project.data?.encoder?.key]);

  const save = useMutation({
    mutationFn: async (form: TargetForm) => {
      if (form.kind === "project" && projectEncoder && projectEncoder !== (project.data?.encoder?.key ?? null)) {
        await api.setProjectEncoder(form.projectId, projectEncoder);
      }
      const body = toBody(form, preset);
      return editing ? api.putTarget(current!.id, body) : api.createTarget(body);
    },
    onSuccess: (t) => {
      qc.invalidateQueries({ queryKey: qk.autopilotTargets });
      qc.invalidateQueries({ queryKey: qk.encoders });
      qc.invalidateQueries({ queryKey: ["autopilot", "journal"] });
      if (editing) {
        onClose();
        return;
      }
      // created: stay open on the new target so its schedule can be set
      setCurrent(t);
      setF(toForm(t));
      setCreated(true);
    },
  });
  const errors = useMemo(() => errorsByField(save.error), [save.error]);
  const err = (field: string) => errors[field];
  const known = new Set(["id", "name", "kind", "headset", "project", "headset.preset", "headset.encoder", "headset.output_root", "headset.heads", "project.project_id"]);
  const otherErrors = Object.entries(errors).filter(
    ([k]) =>
      !known.has(k) && !k.startsWith("headset.sources") && !k.startsWith("sources") && !k.startsWith("headset.policy.") && !k.startsWith("project.policy."),
  );
  const set = (patch: Partial<TargetForm>) => setF((x) => ({ ...x, ...patch }));
  const setSource = (i: number, patch: Partial<DataSource>) => set({ sources: f.sources.map((s, j) => (j === i ? { ...s, ...patch } : s)) });
  const policyKey = f.kind === "headset" ? "headset.policy" : "project.policy";
  const shownId = f.idEdited ? f.id : idFromName(f.name);

  return (
    <Dialog open={open} onClose={onClose} title={editing ? `Edit ${current!.name}` : "New autopilot target"} width={680}>
      {/* not a <form>: the schedule section holds ScheduleEditor's own form */}
      <div className="space-y-4 max-h-[72vh] overflow-y-auto pr-1" role="group" aria-label="Target">
        {created && (
          <p className="text-[12.5px] text-good inline-flex items-center gap-1.5">
            <Check size={13} /> Created. Set its schedule below, or close.
          </p>
        )}
        {otherErrors.length > 0 && (
          <div role="alert" className="text-[12.5px] text-bad space-y-0.5">
            {otherErrors.map(([k, v]) => (
              <p key={k}>{k ? `${k}: ${v}` : v}</p>
            ))}
          </div>
        )}
        <div className="grid grid-cols-2 gap-3">
          <FormField label="Name" error={err("name")}>
            <Input aria-label="Name" value={f.name} onChange={(e) => set({ name: e.target.value })} placeholder="Support ticket heads (0.6B)" required />
          </FormField>
          {!editing && (
            <FormField label="Id" error={err("id")} hint="Derived from the name; used in links. Can't change later.">
              <Input aria-label="Id" className="font-mono text-[12.5px]" value={shownId} onChange={(e) => set({ id: e.target.value, idEdited: true })} />
            </FormField>
          )}
        </div>

        <Section title="Kind">
          {editing ? (
            <p className="text-[12.5px] text-ink-2">{f.kind === "headset" ? "Head set" : "Project"}</p>
          ) : (
            <Segmented
              value={f.kind}
              onChange={(k) => set({ kind: k })}
              options={[
                { value: "headset", label: "Head set" },
                { value: "project", label: "Project" },
              ]}
            />
          )}
        </Section>

        <Section title="What it trains">
          {f.kind === "project" ? (
            <FormField label="Project" error={err("project.project_id") ?? err("project")}>
              <Select aria-label="Project" value={f.projectId} onChange={(e) => set({ projectId: e.target.value })}>
                <option value="" disabled>
                  {projects.isLoading ? "Loading…" : "Pick a project"}
                </option>
                {(projects.data ?? []).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </Select>
            </FormField>
          ) : (
            <>
              {presets.data && !presets.data.length ? (
                <div className="text-[12.5px] text-muted space-y-1" role="status" aria-label="No head-set presets">
                  <p>No head-set presets are configured, so there is no head set to pick{editing && f.preset ? ` (this target names ${f.preset})` : ""}.</p>
                  <p>
                    Head-set presets are configured in <code className="font-mono">headset_presets.json</code> in the data folder, or via the{" "}
                    <code className="font-mono">FORGE_HEADSET_PRESETS</code> environment variable. Project targets work without them.
                  </p>
                  {(err("headset.preset") ?? err("headset")) && <p className="text-bad" role="alert">{err("headset.preset") ?? err("headset")}</p>}
                </div>
              ) : (
                <FormField label="Head set" error={err("headset.preset") ?? err("headset")} hint={preset?.description}>
                  <Select
                    aria-label="Head set"
                    value={f.preset}
                    onChange={(e) => {
                      const p = presets.data?.find((x) => x.key === e.target.value);
                      if (p) setF((x) => withPreset(x, p));
                    }}
                  >
                    {!f.preset && (
                      <option value="" disabled>
                        {presets.isLoading ? "Loading…" : "Registered by hand (no preset)"}
                      </option>
                    )}
                    {(presets.data ?? []).map((p) => (
                      <option key={p.key} value={p.key}>
                        {p.name}
                      </option>
                    ))}
                  </Select>
                </FormField>
              )}
              <FormField label="Heads" error={err("headset.heads")} hint={preset?.fixed_heads ? "This head set always trains these heads." : f.heads.length ? `${f.heads.length} picked` : "None picked: every head the adapter reports."}>
                <HeadPicker
                  options={preset?.fixed_heads ?? heads.data?.map((h) => h.head) ?? f.heads}
                  retired={Object.fromEntries((heads.data ?? []).filter((h) => h.retired).map((h) => [h.head, h.retired!]))}
                  value={f.heads}
                  fixed={!!preset?.fixed_heads}
                  loading={heads.isFetching}
                  error={heads.isError ? (heads.error as Error).message : null}
                  onChange={(hs) => set({ heads: hs })}
                />
              </FormField>
            </>
          )}
        </Section>

        <Section title="Encoder">
          {f.kind === "project" ? (
            <FormField label="Encoder" hint="The project's own encoder: it changes for the project everywhere, from its next training stage.">
              <EncoderSelect
                aria-label="Encoder"
                value={projectEncoder}
                placeholder={f.projectId ? "Pick the project's encoder" : "Pick a project first"}
                disabled={!f.projectId}
                onChange={setProjectEncoder}
              />
            </FormField>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              <FormField
                label="Encoder"
                error={err("headset.encoder")}
                hint={
                  !preset
                    ? "Required."
                    : preset.recommendation.key === f.encoder
                      ? `Recommended: ${preset.recommendation.reason}.`
                      : preset.recommendation.key
                        ? `Required. Forge recommends ${preset.recommendation.name} for ${preset.name}.`
                        : `Required. ${preset.recommendation.reason}.`
                }
              >
                <EncoderSelect
                  aria-label="Encoder"
                  value={f.encoder}
                  allowed={preset?.encoders}
                  disabledReason={preset ? `${preset.name} can't train on it` : undefined}
                  onChange={(k) => set({ encoder: k })}
                />
              </FormField>
              <FormField
                label="Output folder"
                error={err("headset.output_root")}
                hint={f.outputRootEdited ? "Set by hand: it stays when the encoder changes." : "Follows the encoder. Edit to set it by hand."}
              >
                <Input
                  aria-label="Output folder"
                  className="font-mono text-[12px]"
                  value={shownOutputRoot(f, preset)}
                  onChange={(e) => set({ outputRoot: e.target.value, outputRootEdited: true })}
                />
              </FormField>
            </div>
          )}
        </Section>

        {f.kind === "headset" && (
          <Section title="Data sources">
            {(err("headset.sources") ?? err("sources")) && <p className="text-[12.5px] text-bad" role="alert">{err("headset.sources") ?? err("sources")}</p>}
            {!f.sources.length && <p className="text-[12.5px] text-muted">{preset ? "This head set pulls no real data." : "No sources."}</p>}
            <ul className="space-y-2">
              {f.sources.map((s, i) => (
                <SourceRow
                  key={s.kind}
                  source={s}
                  def={preset?.sources.find((x) => x.kind === s.kind)}
                  stored={f.base?.sources.find((x) => x.kind === s.kind)}
                  label={sourceLabel(s.kind, preset)}
                  err={(path) => err(`headset.sources.${i}${path}`) ?? err(`sources.${i}${path}`)}
                  onChange={(patch) => setSource(i, patch)}
                />
              ))}
            </ul>
          </Section>
        )}

        <Section title="Schedule">
          {editing ? (
            <ScheduleEditor targetId={current!.id} schedule={current!.schedule} />
          ) : (
            <p className="text-[12.5px] text-muted">Set once the target exists (right after you create it).</p>
          )}
        </Section>

        <details className="pt-3 border-t border-line-soft" open={Object.keys(errors).some((k) => k.startsWith(policyKey))}>
          <summary className="text-[11.5px] uppercase tracking-wide text-muted cursor-pointer select-none">Policy</summary>
          <div className="grid grid-cols-2 gap-3 mt-2.5">
            {POLICY_FIELDS.map((p) => (
              <FormField key={p.key} label={p.label} hint={p.hint} error={err(`${policyKey}.${p.key}`)}>
                <Input
                  aria-label={p.label}
                  type="number"
                  step={p.step ?? 1}
                  value={String(f.policy[p.key] ?? "")}
                  onChange={(e) => set({ policy: { ...f.policy, [p.key]: e.target.value === "" ? undefined : Number(e.target.value) } })}
                />
              </FormField>
            ))}
          </div>
          <label className="flex items-start gap-2 text-[12.5px] mt-3 cursor-pointer">
            <input
              type="checkbox"
              className="mt-0.5"
              aria-label="Replace a failing head with a better failing one"
              checked={Boolean(f.policy.promote_over_failing_incumbent)}
              onChange={(e) => set({ policy: { ...f.policy, promote_over_failing_incumbent: e.target.checked } })}
            />
            <span>
              <span className="text-ink font-medium">Replace a failing head with a better failing one</span>
              <span className="block text-muted text-[11.5px]">
                When both the candidate and the promoted head miss the test gate, promote the candidate if it scores higher on test. Off by default; every use is journalled.
              </span>
            </span>
          </label>
        </details>

        <div className="flex items-center justify-end gap-2 pt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            {created ? "Close" : "Cancel"}
          </Button>
          <Button
            type="button"
            variant="primary"
            disabled={save.isPending || !f.name.trim() || (f.kind === "project" ? !f.projectId : !f.encoder || (!editing && !f.preset))}
            onClick={() => save.mutate(f)}
          >
            {save.isPending ? "Saving…" : editing ? "Save target" : "Create target"}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

/**
 * One real-data source: an on/off box with the preset's label, and per setting a select (one value) or a box per value
 * (several). Turning the stored target's source, or one value it reads, off asks first: the rows it supplied are removed
 * at the next run (a new target, or a source not on yet, has no rows to lose).
 * A source the preset doesn't describe (a target registered by hand) shows its settings as they are.
 */
function SourceRow({
  source: s,
  def,
  stored,
  label,
  err,
  onChange,
}: {
  source: DataSource;
  def: PresetSource | undefined;
  /** This source as the saved target has it (none on create). */
  stored: DataSource | undefined;
  label: string;
  /** The server's error at `headset.sources.<i><path>`. */
  err: (path: string) => string | undefined;
  onChange: (patch: Partial<DataSource>) => void;
}) {
  const e = err(".kind") ?? err("");
  const settingsErr = err(".settings");
  const setSetting = (key: string, v: string | string[]) => onChange({ settings: { ...s.settings, [key]: v } });
  const settings = def ? Object.entries(def.settings) : [];
  const unknown = Object.entries(s.settings).filter(([k]) => !def?.settings[k]);
  /** The saved target pulls this source (under `value` of setting `key`, when given): turning it off loses rows. */
  const pulls = (key?: string, value?: string) => {
    if (!stored?.enabled) return false;
    if (key === undefined) return true;
    const v = stored.settings[key];
    return Array.isArray(v) ? v.includes(value ?? "") : v === value;
  };
  return (
    <li className="grid grid-cols-[minmax(0,1fr)_minmax(140px,auto)] gap-3 items-start" aria-label={`${label} source`}>
      <label className="flex items-start gap-2 text-[12.5px] cursor-pointer">
        <input
          type="checkbox"
          className="mt-0.5"
          aria-label={label}
          checked={s.enabled}
          onChange={(ev) => {
            if (!ev.target.checked && pulls() && !confirm(disableSourceWarning(label))) return;
            onChange({ enabled: ev.target.checked });
          }}
        />
        <span>
          <span className="text-ink font-medium">{label}</span>
          {!def && <span className="block text-muted text-[11.5px]">Not described by a head-set preset: its settings are kept as they are.</span>}
          {e && <span className="block text-bad text-[11.5px]" role="alert">{e}</span>}
          {settingsErr && <span className="block text-bad text-[11.5px]" role="alert">{settingsErr}</span>}
        </span>
      </label>
      <div className="space-y-2">
        {settings.map(([key, st]) => {
          const name = settingLabel(key);
          const sErr = err(`.settings.${key}`);
          if (st.multiple) {
            const cur = Array.isArray(s.settings[key]) ? (s.settings[key] as string[]) : s.settings[key] ? [String(s.settings[key])] : [];
            return (
              <FormField key={key} label={name} error={sErr}>
                <div className="flex flex-wrap gap-x-3 gap-y-1 pt-1" role="group" aria-label={`${label} ${name}`}>
                  {st.values.map((v) => (
                    <label key={v} className="flex items-center gap-1 text-[12px]">
                      <input
                        type="checkbox"
                        aria-label={`${label} ${name} ${v}`}
                        checked={cur.includes(v)}
                        disabled={!s.enabled}
                        onChange={(ev) => {
                          if (!ev.target.checked && s.enabled && pulls(key, v) && !confirm(disableSourceWarning(label, v))) return;
                          setSetting(key, ev.target.checked ? st.values.filter((x) => x === v || cur.includes(x)) : cur.filter((x) => x !== v));
                        }}
                      />
                      {v}
                    </label>
                  ))}
                </div>
              </FormField>
            );
          }
          const value = String(Array.isArray(s.settings[key]) ? ((s.settings[key] as string[])[0] ?? "") : (s.settings[key] ?? ""));
          return (
            <FormField key={key} label={name} error={sErr}>
              <Select
                aria-label={`${label} ${name}`}
                value={value}
                disabled={!s.enabled}
                onChange={(ev) => {
                  // another value: the rows read under the current one are no longer pulled, so they are removed
                  if (s.enabled && pulls(key, value) && !confirm(disableSourceWarning(label, value))) return;
                  setSetting(key, ev.target.value);
                }}
              >
                {!st.values.includes(value) && <option value={value}>{value || "(none)"}</option>}
                {st.values.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </Select>
            </FormField>
          );
        })}
        {unknown.map(([key, v]) => (
          <FormField key={key} label={settingLabel(key)} error={err(`.settings.${key}`)}>
            <p className="font-mono text-[12px] text-ink-2 pt-1">{Array.isArray(v) ? v.join(", ") : v}</p>
          </FormField>
        ))}
      </div>
    </li>
  );
}

/** Chip multiselect of a head set's heads; nothing picked = every head. */
function HeadPicker({
  options,
  retired,
  value,
  fixed,
  loading,
  error,
  onChange,
}: {
  options: string[];
  retired: Record<string, string>;
  value: string[];
  fixed: boolean;
  loading: boolean;
  error: string | null;
  onChange: (v: string[]) => void;
}) {
  if (error) return <p className="text-[12.5px] text-bad">{error}</p>;
  if (loading && !options.length)
    return (
      <span className="inline-flex items-center gap-2 text-[12.5px] text-muted">
        <Spinner size={12} /> Asking the head set for its heads…
      </span>
    );
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Heads">
      {options.map((h) => {
        const on = value.includes(h);
        return (
          <button
            key={h}
            type="button"
            aria-pressed={on}
            disabled={fixed}
            title={retired[h] ? `retired: ${retired[h]}` : undefined}
            onClick={() => onChange(on ? value.filter((x) => x !== h) : [...value, h])}
            className={cn(
              "inline-flex items-center h-6 px-2 rounded-[6px] border text-[11.5px] font-mono cursor-pointer disabled:cursor-default",
              on ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-2 hover:text-ink",
              retired[h] && "opacity-60",
            )}
          >
            {h}
          </button>
        );
      })}
    </div>
  );
}
