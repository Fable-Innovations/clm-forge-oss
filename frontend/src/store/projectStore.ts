import type {
  BaseModelSelection,
  CurationEv,
  CurationPhase,
  EmbedEv,
  ErrorEv,
  EvalReport,
  ForgeEvent,
  PackageManifest,
  ProjectState,
  Proposal,
  RejectedEv,
  StateEv,
  ToolEv,
  TrainEpochEv,
  TrainSnapshotEv,
  TrainStepEv,
} from "../api/types";

export type ChatItem =
  | { kind: "msg"; id: string; role: "user" | "assistant"; content: string; streaming: boolean; seq: number; ts: string }
  | { kind: "tools"; key: string; steps: ToolEv[]; seq: number }
  | { kind: "state"; ev: StateEv; seq: number }
  | { kind: "rejected"; ev: RejectedEv; seq: number }
  | { kind: "error"; ev: ErrorEv; seq: number };

export interface TrainRun {
  index: number;
  startedSeq: number;
  epochs: TrainEpochEv[];
  snapshots: TrainSnapshotEv[];
  lastStep: TrainStepEv | null;
}

export interface ProjectStore {
  lastSeq: number;
  state: ProjectState | null;
  transitions: StateEv[];
  timeline: ForgeEvent[]; // state / rejected / log / error / proposal / eval / package
  chat: ChatItem[];
  proposal: Proposal | null;
  proposals: Proposal[];
  curation: (CurationEv & { phases: Partial<Record<CurationPhase, { done: number; total: number }>> }) | null;
  embed: EmbedEv | null;
  baseModel: BaseModelSelection | null;
  runs: TrainRun[];
  report: EvalReport | null;
  manifest: PackageManifest | null;
  rejected: RejectedEv[];
  errors: ErrorEv[];
}

export const emptyStore = (): ProjectStore => ({
  lastSeq: -1,
  state: null,
  transitions: [],
  timeline: [],
  chat: [],
  proposal: null,
  proposals: [],
  curation: null,
  embed: null,
  baseModel: null,
  runs: [],
  report: null,
  manifest: null,
  rejected: [],
  errors: [],
});

/** Apply a batch of events. Clones the top-level containers once per batch, then mutates the clones. */
export function applyEvents(prev: ProjectStore, events: ForgeEvent[]): ProjectStore {
  const fresh = events.filter((e) => typeof e.seq !== "number" || e.seq > prev.lastSeq);
  if (!fresh.length) return prev;
  const s: ProjectStore = {
    ...prev,
    transitions: prev.transitions.slice(),
    timeline: prev.timeline.slice(),
    chat: prev.chat.slice(),
    proposals: prev.proposals.slice(),
    runs: prev.runs.map((r) => ({ ...r, epochs: r.epochs.slice(), snapshots: r.snapshots.slice() })),
    rejected: prev.rejected.slice(),
    errors: prev.errors.slice(),
  };
  const currentRun = (): TrainRun => {
    if (!s.runs.length) s.runs.push({ index: 0, startedSeq: 0, epochs: [], snapshots: [], lastStep: null });
    return s.runs[s.runs.length - 1];
  };

  for (const e of fresh) {
    if (typeof e.seq === "number") s.lastSeq = Math.max(s.lastSeq, e.seq);
    switch (e.type) {
      case "state": {
        s.state = e.to;
        s.transitions.push(e);
        s.timeline.push(e);
        s.chat.push({ kind: "state", ev: e, seq: e.seq });
        if (e.to === "training" && e.event !== "retry") {
          const last = s.runs[s.runs.length - 1];
          if (!last || last.epochs.length || last.snapshots.length) {
            s.runs.push({ index: s.runs.length, startedSeq: e.seq, epochs: [], snapshots: [], lastStep: null });
          }
        }
        if (e.to === "curating") s.curation = null;
        if (e.to === "selecting" && e.event !== "retry") s.baseModel = null;
        break;
      }
      case "rejected":
        s.rejected.push(e);
        s.timeline.push(e);
        s.chat.push({ kind: "rejected", ev: e, seq: e.seq });
        break;
      case "error":
        s.errors.push(e);
        s.timeline.push(e);
        s.chat.push({ kind: "error", ev: e, seq: e.seq });
        break;
      case "log":
        s.timeline.push(e);
        break;
      case "chat": {
        const i = s.chat.findIndex((c) => c.kind === "msg" && c.id === e.id);
        const item: ChatItem = { kind: "msg", id: e.id, role: e.role, content: e.content, streaming: false, seq: e.seq, ts: e.ts };
        if (i >= 0) s.chat[i] = { ...item, seq: (s.chat[i] as { seq: number }).seq };
        else s.chat.push(item);
        break;
      }
      case "chat_delta": {
        const i = s.chat.findIndex((c) => c.kind === "msg" && c.id === e.id);
        if (i >= 0) {
          const m = s.chat[i];
          if (m.kind === "msg" && m.streaming) s.chat[i] = { ...m, content: m.content + e.delta };
        } else {
          s.chat.push({ kind: "msg", id: e.id, role: "assistant", content: e.delta, streaming: true, seq: e.seq, ts: e.ts });
        }
        break;
      }
      case "tool": {
        const last = s.chat[s.chat.length - 1];
        if (last && last.kind === "tools") s.chat[s.chat.length - 1] = { ...last, steps: [...last.steps, e] };
        else s.chat.push({ kind: "tools", key: `t${e.seq}`, steps: [e], seq: e.seq });
        break;
      }
      case "proposal":
        s.proposal = e.proposal;
        s.proposals.push(e.proposal);
        s.timeline.push(e);
        break;
      case "curation": {
        const phases = { ...(s.curation?.phases ?? {}) };
        phases[e.phase] = { done: e.done, total: e.total };
        s.curation = { ...e, phases };
        break;
      }
      case "embed":
        s.embed = e;
        break;
      case "base_model":
        s.baseModel = { chosen: e.chosen, candidates: e.candidates };
        break;
      case "train_epoch": {
        const r = currentRun();
        const j = r.epochs.findIndex((x) => x.epoch === e.epoch);
        if (j >= 0) r.epochs[j] = e;
        else r.epochs.push(e);
        break;
      }
      case "train_step": {
        const r = currentRun();
        r.lastStep = e;
        break;
      }
      case "train_snapshot": {
        const r = currentRun();
        const j = r.snapshots.findIndex((x) => x.epoch === e.epoch);
        if (j >= 0) r.snapshots[j] = e;
        else r.snapshots.push(e);
        break;
      }
      case "eval":
        s.report = e.report;
        s.timeline.push(e);
        break;
      case "package":
        s.manifest = e.manifest;
        s.timeline.push(e);
        break;
      default:
        break;
    }
  }
  return s;
}
