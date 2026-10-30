import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import type { AutopilotTarget, HeadsetPreset } from "../api/types";
import { defaultSettings, disableSourceWarning, emptyForm, errorsByField, idFromName, savableEncoders, settingLabel, shownOutputRoot, sourceLabel, sourcesFor, toBody, toForm, withPreset } from "./targetForm";

const PRESET: HeadsetPreset = {
  key: "example-heads",
  name: "Example heads",
  description: "",
  dataset_root: "/data",
  output_base: "/heads",
  protected_roots: ["/heads"],
  fixed_heads: null,
  encoders: ["qwen3-0.6b", "qwen3-0.6b-bf16", "qwen3-4b"],
  output_roots: { "qwen3-0.6b": "/heads-qwen3-0.6b", "qwen3-0.6b-bf16": "/heads-qwen3-0.6b-bf16", "qwen3-4b": "/heads-qwen3-4b" },
  recommendation: { key: "qwen3-0.6b", name: "Qwen3-0.6B (MLX 4-bit)", reason: "Qwen3-0.6B (MLX 4-bit): the encoder this head set is built for" },
  sources: [
    { kind: "event_log", label: "Event log", enabled_by_default: true, settings: { stage: { values: ["qa", "prod", "dev"], default: "qa", multiple: false } } },
    { kind: "shared_export", label: "Shared export", enabled_by_default: false, settings: { stages: { values: ["qa", "prod"], default: ["qa", "prod"], multiple: true } } },
  ],
};
const TARGET: AutopilotTarget = {
  id: "example",
  name: "Example heads",
  kind: "headset",
  enabled: true,
  headset: {
    preset: "example-heads",
    adapter: ["uv", "run"],
    cwd: "/b",
    env: { A: "1" },
    dataset_root: "/data",
    encoder: "qwen3-0.6b-bf16",
    output_root: "/heads-qwen3-0.6b-bf16",
    heads: ["ticket-priority"],
    protected_roots: ["/heads"],
    protected_globs: ["*/rows.jsonl"],
    protected_files: ["/f.jsonl"],
    sources: [{ kind: "event_log", enabled: true, settings: { stage: "prod" } }],
    adapter_config: { python: "/py" },
    policy: { min_count: 150, gen_cap: 600, max_rounds: 3, attempts: 3, exemplars: 4, adjudicate: true, tol: 0.01, tol_per_head: { h: 0.02 } },
  },
  project: null,
  schedule: null,
  last_run: null,
  output_roots: { "qwen3-0.6b": "/heads-qwen3-0.6b", "qwen3-0.6b-bf16": "/heads-qwen3-0.6b-bf16", "qwen3-4b": "/heads-qwen3-4b" },
};

describe("targetForm", () => {
  it("round-trips what a target may set, never what its preset sets", () => {
    const body = toBody(toForm(TARGET), PRESET);
    const h = TARGET.headset!;
    expect(body).toEqual({ id: "example", name: "Example heads", kind: "headset", enabled: true, project: null, headset: {
      preset: h.preset, encoder: h.encoder, output_root: h.output_root, heads: h.heads, sources: h.sources, policy: h.policy,
      adapter_config: h.adapter_config } });
    // the command and the folders it reads are the preset's: a request never carries them
    for (const k of ["adapter", "cwd", "env", "dataset_root", "protected_roots", "protected_globs", "protected_files"])
      expect(body.headset).not.toHaveProperty(k);
  });

  it("an encoder change moves a derived output folder, a hand-set one stays", () => {
    const f = { ...toForm(TARGET), encoder: "qwen3-0.6b" };
    expect(shownOutputRoot(f, PRESET)).toBe("/heads-qwen3-0.6b");
    expect(toBody(f, PRESET).headset!.output_root).toBe("/heads-qwen3-0.6b");
    const hand = { ...f, outputRoot: "/elsewhere", outputRootEdited: true };
    expect(toBody(hand, PRESET).headset!.output_root).toBe("/elsewhere");
    // a stored folder that is not the preset's stays put, whatever the encoder
    const own = toForm({ ...TARGET, headset: { ...TARGET.headset!, output_root: "/mine" }, output_roots: { "qwen3-0.6b": "/mine", "qwen3-0.6b-bf16": "/mine" } });
    expect(shownOutputRoot(own, PRESET)).toBe("/mine");
    expect(shownOutputRoot({ ...own, encoder: "qwen3-0.6b" }, PRESET)).toBe("/mine");
  });

  it("a new head-set target names its preset and sends only what the form edits", () => {
    const f = withPreset({ ...emptyForm(), name: "Example heads (0.6B)" }, PRESET);
    expect(f.encoder).toBe("qwen3-0.6b");                       // the preset's recommendation
    // every source the preset offers, on when it is on by default, with each setting's default
    expect(f.sources).toEqual([
      { kind: "event_log", enabled: true, settings: { stage: "qa" } },
      { kind: "shared_export", enabled: false, settings: { stages: ["qa", "prod"] } },
    ]);
    const b = toBody({ ...f, policy: { gen_cap: 300 } }, PRESET);
    expect(b.id).toBe("example-heads-0.6b");
    expect(b.headset).toEqual({ preset: "example-heads", encoder: "qwen3-0.6b", output_root: "/heads-qwen3-0.6b", heads: [], sources: f.sources, policy: { gen_cap: 300 } });
  });

  it("a new target starts on the preset's recommendation, never on a protected folder or a global default", () => {
    expect(emptyForm().encoder).toBe("");                                    // no app-wide default encoder
    // a recommendation whose folder is protected is not preselected: the user must pick
    const p = { ...PRESET, encoders: ["qwen3-4b", "qwen3-0.6b"], output_roots: { "qwen3-4b": "/heads/4b", "qwen3-0.6b": "/heads-qwen3-0.6b" },
                recommendation: { key: "qwen3-4b", name: "Qwen3-4B", reason: "x" } };
    expect(savableEncoders(p)).toEqual(["qwen3-0.6b"]);
    expect(withPreset(emptyForm(), p).encoder).toBe("");
    // nothing fits this machine: no preselection
    expect(withPreset(emptyForm(), { ...PRESET, recommendation: { key: null, name: null, reason: "no registered encoder fits" } }).encoder).toBe("");
    // an encoder the user already picked, when the preset trains on it, is kept
    expect(withPreset({ ...emptyForm(), encoder: "qwen3-4b" }, PRESET).encoder).toBe("qwen3-4b");
    expect(shownOutputRoot(withPreset({ ...emptyForm(), encoder: "qwen3-4b" }, PRESET), PRESET)).toBe("/heads-qwen3-4b");
  });

  it("asks before turning a source, or one value it reads, off: its stored rows are removed", () => {
    expect(disableSourceWarning("Event log")).toMatch(/^Turn off Event log\? .*removes every stored real row/);
    expect(disableSourceWarning("Shared export", "qa")).toMatch(/^Turn off Shared export \(qa\)\?/);
  });

  it("fills a target's sources from the preset: its own entries kept, missing settings and sources defaulted", () => {
    expect(defaultSettings(PRESET.sources[1])).toEqual({ stages: ["qa", "prod"] });
    expect(sourcesFor(PRESET.sources, [{ kind: "shared_export", enabled: true, settings: {} }, { kind: "gone", enabled: true, settings: {} }])).toEqual([
      { kind: "event_log", enabled: false, settings: { stage: "qa" } },
      { kind: "shared_export", enabled: true, settings: { stages: ["qa", "prod"] } },
    ]);
    expect(sourceLabel("shared_export", PRESET)).toBe("Shared export");
    expect(sourceLabel("hand_made", undefined)).toBe("hand_made");
    expect(settingLabel("min_age_days")).toBe("Min age days");
  });

  it("a project target carries the project and its policy", () => {
    const b = toBody({ ...emptyForm(), kind: "project", name: "Tickets nightly", projectId: "p1", policy: { min_count: 20 } }, undefined);
    expect(b).toEqual({ id: "tickets-nightly", name: "Tickets nightly", kind: "project", enabled: true, headset: null, project: { project_id: "p1", policy: { min_count: 20 } } });
  });

  it("maps a 422 to errors per field, anything else to one message", () => {
    const e = new ApiError(422, "x", { errors: [{ field: "headset.output_root", message: "inside a protected folder" }, { field: "headset.policy.min_count", message: "must be at least 1" }] });
    expect(errorsByField(e)).toEqual({ "headset.output_root": "inside a protected folder", "headset.policy.min_count": "must be at least 1" });
    expect(errorsByField(new ApiError(409, "a target with id x already exists"))).toEqual({ "": "a target with id x already exists" });
    expect(errorsByField(null)).toEqual({});
  });

  it("derives ids like the backend", () => {
    expect(idFromName("  Support Ticket Heads (0.6B)! ")).toBe("support-ticket-heads-0.6b");
  });
});
