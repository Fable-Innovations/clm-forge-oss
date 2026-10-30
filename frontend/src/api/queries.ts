import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

export const qk = {
  health: ["health"] as const,
  providers: ["providers"] as const,
  settings: ["settings"] as const,
  projects: ["projects"] as const,
  project: (pid: string) => ["project", pid] as const,
  dataset: (pid: string) => ["dataset", pid] as const,
  eval: (pid: string) => ["eval", pid] as const,
  pkg: (pid: string) => ["package", pid] as const,
  pkgFile: (pid: string, path: string) => ["package-file", pid, path] as const,
  labelsets: ["labelsets"] as const,
  /** Every page and filter of one label set shares this prefix, so one invalidation refreshes them all. */
  labelQueue: (lsid: string) => ["label-queue", lsid] as const,
  encoders: ["encoders"] as const,
  encoderRecommendation: ["encoders", "recommendation"] as const,
  headsets: ["autopilot", "headsets"] as const,
  headsetHeads: (preset: string, encoder: string) => ["autopilot", "headset-heads", preset, encoder] as const,
  autopilotTargets: ["autopilot", "targets"] as const,
  autopilotRuns: (target: string) => ["autopilot", "runs", target] as const,
  autopilotRun: (rid: string) => ["autopilot", "run", rid] as const,
  journal: (q: Record<string, string | undefined>) => ["autopilot", "journal", q] as const,
  modelQueue: ["autopilot", "queue"] as const,
  groups: ["autopilot", "groups"] as const,
};

/** The label set id of an experiment's queue. */
export const projectLsid = (pid: string) => `p:${pid}`;

export function useLabelsets() {
  return useQuery({ queryKey: qk.labelsets, queryFn: api.labelsets });
}

export function useHealth() {
  return useQuery({ queryKey: qk.health, queryFn: api.health, refetchInterval: 15_000, retry: false });
}

export function useProviders() {
  return useQuery({ queryKey: qk.providers, queryFn: () => api.providers(), staleTime: 60_000 });
}

export function useProjects() {
  return useQuery({ queryKey: qk.projects, queryFn: api.projects, refetchInterval: 10_000 });
}

export function useProject(pid: string) {
  return useQuery({ queryKey: qk.project(pid), queryFn: () => api.project(pid) });
}

export function useSettings() {
  return useQuery({ queryKey: qk.settings, queryFn: api.settings, retry: false });
}

/** The encoder registry: names, sizes, whether each is downloaded, and what uses it. */
export function useEncoders() {
  return useQuery({ queryKey: qk.encoders, queryFn: api.encoders, staleTime: 30_000 });
}

/** Forge's encoder recommendation for a new experiment (forge.encoder_choice). */
export function useEncoderRecommendation(enabled = true) {
  return useQuery({ queryKey: qk.encoderRecommendation, queryFn: api.encoderRecommendation, staleTime: 60_000, enabled });
}

/** Named head sets a target can be made from. */
export function useHeadsets() {
  return useQuery({ queryKey: qk.headsets, queryFn: api.headsets, staleTime: 5 * 60_000 });
}

/** Poll every 3 s while a run is queued or running, then stop (SPEC: polling, no SSE). */
export const RUN_POLL_MS = 3_000;
export const isLive = (state: string | undefined | null) => state === "queued" || state === "running";
