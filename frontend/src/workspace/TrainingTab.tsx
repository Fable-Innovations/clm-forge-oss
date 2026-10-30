import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, Radio, SkipBack } from "lucide-react";
import { Chip, Empty, InfoTip, Panel, Segmented } from "../components/ui";
import { Heatmap } from "../viz/Heatmap";
import { EmbeddingScatter } from "../viz/EmbeddingScatter";
import { MiniChart } from "../viz/MiniChart";
import { alignSnapshots, globalExtent, simMargin } from "../viz/align";
import { cn, int, num, pct } from "../lib/format";
import { BaseModelPanel } from "./BaseModelPanel";
import { useWs } from "./context";

function Stat({ label, value, sub, info, children }: { label: string; value: React.ReactNode; sub?: React.ReactNode; info?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="min-w-0 px-4 py-3 border-r border-line-soft last:border-r-0">
      <div className="text-[12px] text-muted flex items-center gap-1">
        {label}
        {info && <InfoTip>{info}</InfoTip>}
      </div>
      <div className="text-[20px] font-semibold tnum leading-tight mt-0.5">{value}</div>
      {sub && <div className="text-[11.5px] text-ink-2 mt-0.5 truncate">{sub}</div>}
      {children}
    </div>
  );
}

export function TrainingTab() {
  const { store, task, colorOf, labelText, labels, detail } = useWs();
  const runs = store.runs.filter((r) => r.epochs.length || r.snapshots.length);
  const [runIdx, setRunIdx] = useState<number | null>(null);
  const run = runs[runIdx ?? runs.length - 1];
  const live = store.state === "training" && run === runs[runs.length - 1];

  const aligned = useMemo(() => alignSnapshots(run?.snapshots ?? []), [run?.snapshots]);
  const extent = useMemo(() => globalExtent(aligned), [aligned]);
  const simDomain = useMemo<[number, number]>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of run?.snapshots ?? []) for (const row of s.similarity.values) for (const v of row) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return Number.isFinite(lo) ? [Math.floor(lo * 20) / 20, Math.ceil(hi * 20) / 20] : [0, 1];
  }, [run?.snapshots]);

  const [follow, setFollow] = useState(true);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [focus, setFocus] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (follow && aligned.length) setIdx(aligned.length - 1);
  }, [aligned.length, follow]);
  useEffect(() => {
    setFollow(true);
    setPlaying(false);
  }, [runIdx]);

  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => {
      setIdx((i) => {
        if (i >= aligned.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 620);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, aligned.length]);

  const snapEv = useMemo(() => (run ? run.snapshots.slice().sort((a, b) => a.epoch - b.epoch) : []), [run]);
  const curSnap = snapEv[idx];
  const curEpoch = aligned[idx]?.epoch ?? null;

  const threshold = useMemo(() => {
    const p = store.proposal ?? detail?.proposal;
    const o = p?.opportunities.find((x) => x.id === (detail?.accepted_opportunity_id ?? p.recommended_id));
    return o && o.success_metric === "accuracy" ? o.success_threshold : undefined;
  }, [store.proposal, detail]);

  const rows = useMemo(() => {
    const byE = new Map<number, Record<string, number> & { epoch: number }>();
    for (const e of run?.epochs ?? []) byE.set(e.epoch, { epoch: e.epoch, train_loss: e.train_loss, val_acc: e.val_acc, val_soft_ce: e.val_soft_ce, lr: e.lr, logit_scale: e.logit_scale });
    for (const s of run?.snapshots ?? []) {
      const r = byE.get(s.epoch) ?? ({ epoch: s.epoch } as Record<string, number> & { epoch: number });
      r.margin = simMargin(s.similarity.values, s.similarity.row_labels, s.similarity.col_labels);
      byE.set(s.epoch, r);
    }
    return [...byE.values()].sort((a, b) => a.epoch - b.epoch);
  }, [run]);

  const baseSel = store.baseModel ?? detail?.base_model_selection ?? null;
  const basePanel = baseSel && baseSel.candidates.length ? <BaseModelPanel sel={baseSel} active={store.state === "selecting"} /> : null;

  if (!run) {
    return (
      <div className={basePanel ? "p-4 space-y-4" : undefined}>
        {basePanel}
        <Empty title={store.state === "curating" ? "Curating data first" : store.state === "selecting" ? "Choosing the base model" : "No training yet"}>
          {store.state === "curating"
            ? "Curation runs first, then each base model is compared on the validation split, then the winner trains here. Loss curves, a similarity matrix and an embedding map will update live each epoch."
            : store.state === "selecting"
              ? "Training starts as soon as a base model is selected. Loss curves, a similarity matrix and an embedding map will update live each epoch."
              : "After you accept a proposal and curation finishes, a base model is selected and the head trains here. Every epoch updates the loss curves, the similarity matrix and the 2-D map of the projected space."}
        </Empty>
      </div>
    );
  }

  const last = run.epochs[run.epochs.length - 1];
  const epochs = last?.epochs ?? run.snapshots.length;
  const best = run.epochs.reduce<{ v: number; e: number } | null>((b, e) => (!b || e.val_acc > b.v ? { v: e.val_acc, e: e.epoch } : b), null);
  const step = run.lastStep;
  const stepFrac = live && step ? step.step / step.steps : 0;
  const scrub = (e: number) => {
    const i = aligned.findIndex((s) => s.epoch === e);
    if (i >= 0) {
      setFollow(false);
      setPlaying(false);
      setIdx(i);
    }
  };
  const xMax = Math.max(epochs, 1);
  const marginNow = curSnap ? simMargin(curSnap.similarity.values, curSnap.similarity.row_labels, curSnap.similarity.col_labels) : null;
  const marginInit = snapEv[0] ? simMargin(snapEv[0].similarity.values, snapEv[0].similarity.row_labels, snapEv[0].similarity.col_labels) : null;

  return (
    <div className="p-4 space-y-4">
      {basePanel}
      <div className="bg-panel border border-line rounded-[10px] grid grid-cols-2 md:grid-cols-5">
        <Stat label="Epoch" info="One epoch is one full pass over the training data. Training stops early when validation accuracy stops improving for several epochs in a row, so it may finish before the planned count. The bar shows progress through the current run." value={<>{last ? last.epoch : 0}<span className="text-muted text-[14px] font-normal"> / {epochs}</span></>}>
          <div className="h-[3px] mt-1.5 rounded-full bg-line overflow-hidden">
            <div className="h-full bg-accent transition-[width] duration-300" style={{ width: `${(((last?.epoch ?? 0) + stepFrac) / Math.max(1, epochs)) * 100}%` }} />
          </div>
        </Stat>
        <Stat label="Best val accuracy" info="The highest accuracy on the validation split (held-out examples used to pick the best epoch, not to train). The weights from that epoch are what gets saved and evaluated. The pass bar is applied later, to the separate test split." value={best ? pct(best.v) : "—"} sub={best ? `epoch ${best.e}${threshold ? ` · bar ${pct(threshold, 0)} on test` : ""}` : undefined} />
        <Stat label="Train loss" info="How wrong the model is on the training batches, measured with InfoNCE (a contrastive loss that pulls each text toward its correct label option and away from the others in the same batch). Lower is better, and it should trend down. Step-to-step noise is normal." value={last ? num(last.train_loss, 3) : "—"} sub={live && step ? `step ${step.step}/${step.steps}: ${num(step.loss, 3)}` : "InfoNCE, in-batch"} />
        <Stat label="Similarity margin" info="On validation examples, how much closer each text sits to its correct option than to the closest wrong option, averaged across labels. Positive means labels are separating, and bigger is better. “init” is the value before training started." value={marginNow != null ? (marginNow >= 0 ? "+" : "") + num(marginNow, 3) : "—"} sub={marginInit != null ? `own option − best other; init ${num(marginInit, 3)}` : undefined} />
        <Stat label="Logit scale" info="A learned “sharpness” setting that turns similarities into probabilities. Higher values make the model more decisive, and it is capped at 100. lr is the current learning rate: how big each weight update is." value={last ? num(last.logit_scale, 1) : "—"} sub={last ? `lr ${last.lr.toExponential(1)}` : undefined} />
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {runs.length > 1 && (
          <Segmented
            value={String(runIdx ?? runs.length - 1)}
            onChange={(v) => setRunIdx(Number(v) === runs.length - 1 ? null : Number(v))}
            options={runs.map((_, i) => ({ value: String(i), label: `Attempt ${i + 1}` }))}
          />
        )}
        {live && (
          <Chip tone="accent">
            <span className="w-1.5 h-1.5 rounded-full bg-accent pulse-dot" /> training live
          </Chip>
        )}
        {store.embed && store.embed.done < store.embed.total && store.state === "training" && <Chip>encoding {int(store.embed.done)}/{int(store.embed.total)} states</Chip>}
      </div>

      <div className="grid gap-4 grid-cols-1 xl:grid-cols-2">
        <Panel expandable title="Loss" info="Train loss (contrastive InfoNCE on training batches) and validation soft cross-entropy (how much probability the model gives the correct labels on held-out data). Both should fall. If train loss keeps dropping while validation rises, the model is memorising the training set (overfitting)." sub="Train InfoNCE vs validation soft cross-entropy. Click a point to scrub.">
          <MiniChart
            data={rows}
            xMax={xMax}
            series={[
              { key: "train_loss", name: "train loss", color: "var(--c1)" },
              { key: "val_soft_ce", name: "val soft-CE", color: "var(--c2)" },
            ]}
            yFmt={(v) => v.toFixed(2)}
            scrubEpoch={curEpoch}
            onScrub={scrub}
          />
          <div className="flex gap-4 text-[11.5px] text-ink-2 mt-1">
            <span className="inline-flex items-center gap-1.5"><span className="w-3 h-[2px] bg-[var(--c1)]" /> train loss</span>
            <span className="inline-flex items-center gap-1.5"><span className="w-3 h-[2px] bg-[var(--c2)]" /> val soft-CE</span>
          </div>
        </Panel>
        <Panel expandable title="Validation accuracy" info="Share of validation examples labelled correctly after each epoch. The best epoch here is the one that gets kept. The dashed line marks the pass bar, which the final model must reach on the separate test split." sub={threshold ? "Dashed line is the pass bar the test split must clear." : "Accuracy on the validation split each epoch."}>
          <MiniChart
            data={rows}
            xMax={xMax}
            series={[{ key: "val_acc", name: "val accuracy", color: "var(--accent)" }]}
            yDomain={[(dataMin: number) => Math.max(0, Math.floor(Math.min(dataMin, threshold ?? 1) * 10) / 10), 1] as unknown as [number, number]}
            yFmt={(v) => `${Math.round(v * 100)}%`}
            refY={threshold}
            refYLabel={threshold ? `pass bar ${pct(threshold, 0)}` : undefined}
            scrubEpoch={curEpoch}
            onScrub={scrub}
          />
        </Panel>
      </div>

      <Panel
        expandable title="How the projected space separates"
        info="The model turns every text and every label option into a point (an embedding) and predicts the option whose point is closest. This section replays how those points moved during training. Use play or the slider to step through epochs, starting from the untrained reference head (init). Good training makes each label's texts gather around their own option."
        sub="Each epoch the head re-projects validation states (dots) and option texts (diamonds). Training pulls each state toward its gold option; the matrix diagonal brightening is the same story in numbers."
        aside={
          <div className="flex items-center gap-1.5">
            {live && (
              <button
                onClick={() => setFollow(true)}
                className={cn("inline-flex items-center gap-1 h-7 px-2 rounded-[5px] text-[12px] font-medium border cursor-pointer", follow ? "border-accent/40 bg-accent-soft text-accent" : "border-line text-ink-2 hover:bg-sunken")}
                title="Follow the latest epoch"
              >
                <Radio size={12} /> Live
              </button>
            )}
          </div>
        }
      >
        <div className="flex items-center gap-3 mb-3">
          <button
            aria-label="Back to epoch 0"
            onClick={() => {
              setFollow(false);
              setPlaying(false);
              setIdx(0);
            }}
            className="p-1.5 rounded-[5px] text-ink-2 hover:bg-sunken cursor-pointer"
          >
            <SkipBack size={15} />
          </button>
          <button
            aria-label={playing ? "Pause" : "Play epochs"}
            onClick={() => {
              setFollow(false);
              if (!playing && idx >= aligned.length - 1) setIdx(0);
              setPlaying((p) => !p);
            }}
            className="w-8 h-8 grid place-items-center rounded-full bg-accent text-accent-ink cursor-pointer hover:brightness-110"
          >
            {playing ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
          </button>
          <div className="flex-1 min-w-0">
            <input
              type="range"
              min={0}
              max={Math.max(0, aligned.length - 1)}
              value={idx}
              onChange={(e) => {
                setFollow(false);
                setPlaying(false);
                setIdx(Number(e.target.value));
              }}
              className="w-full accent-[var(--accent)]"
              aria-label="Epoch"
            />
            <div className="flex justify-between text-[10.5px] text-muted tnum -mt-0.5 px-[2px]">
              <span>init</span>
              <span>epoch {aligned[aligned.length - 1]?.epoch ?? 0}</span>
            </div>
          </div>
          <div className="text-right w-[92px] shrink-0">
            <div className="text-[18px] font-semibold tnum leading-none">{curEpoch === 0 ? "init" : `epoch ${curEpoch ?? "—"}`}</div>
            <div className="text-[11px] text-muted tnum mt-1">{rows.find((r) => r.epoch === curEpoch)?.val_acc != null ? `val acc ${pct(rows.find((r) => r.epoch === curEpoch)!.val_acc)}` : curEpoch === 0 ? "reference head" : ""}</div>
          </div>
        </div>

        <div className="grid gap-5 grid-cols-1 lg:grid-cols-[minmax(300px,42%)_minmax(0,1fr)] items-start">
          <div className="max-w-[460px]">
            <h4 className="text-[12.5px] font-semibold mb-0.5 flex items-center gap-1">
              Similarity matrix
              <InfoTip>Rows group validation texts by their true label, and columns are the label options. Each cell is the average cosine similarity between them, where higher means closer. You want the diagonal (each label vs its own option) to be the brightest cell in its row.</InfoTip>
            </h4>
            <p className="text-[11.5px] text-muted mb-2">Mean cosine between projected validation states (rows, by gold label) and each option (columns).</p>
            {curSnap ? (
              <Heatmap
                values={curSnap.similarity.values}
                rowLabels={curSnap.similarity.row_labels}
                colLabels={curSnap.similarity.col_labels}
                domain={simDomain}
                colorOf={colorOf}
                labelText={labelText}
                rowTitle="gold label"
                colTitle="option"
                cellTitle={(r, c, v) => `${labelText(r)} states vs “${labelText(c)}” option: mean cos ${v.toFixed(3)}`}
              />
            ) : (
              <p className="text-[12px] text-muted">Waiting for the first snapshot…</p>
            )}
          </div>
          <div className="min-w-0">
            <div className="flex items-baseline justify-between gap-3 mb-2">
              <div>
                <h4 className="text-[12.5px] font-semibold mb-0.5 flex items-center gap-1">
                  Embedding map
                  <InfoTip>A 2-D picture of the model's high-dimensional space, flattened with PCA and aligned so epochs are comparable. Dots are validation texts coloured by true label, and diamonds are the label options. Well-separated clusters, each with its own diamond, mean the model can tell the labels apart. Click or hover a label to isolate it.</InfoTip>
                </h4>
                <p className="text-[11.5px] text-muted">2-D PCA, aligned across epochs. Dashed trails show where each option has moved. Click a point to isolate its label.</p>
              </div>
            </div>
            <EmbeddingScatter snaps={aligned} index={idx} extent={extent} colorOf={colorOf} labelText={labelText} focus={focus} onFocus={setFocus} />
            <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[12px]">
              {(labels.length ? labels : curSnap?.similarity.col_labels ?? []).map((l) => (
                <button
                  key={l}
                  onClick={() => setFocus((f) => (f === l ? null : l))}
                  onMouseEnter={() => setFocus(l)}
                  onMouseLeave={() => setFocus(null)}
                  className={cn("inline-flex items-center gap-1.5 px-1 rounded cursor-pointer", focus && focus !== l ? "opacity-40" : "text-ink-2")}
                >
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: colorOf(l) }} />
                  {labelText(l)}
                </button>
              ))}
              <span className="inline-flex items-center gap-1.5 text-muted ml-auto">
                <svg width="12" height="12" viewBox="0 0 12 12">
                  <path d="M6 1 L11 6 L6 11 L1 6 Z" fill="none" stroke="var(--ink)" strokeWidth="1.3" />
                </svg>
                option text
                <span className="w-2 h-2 rounded-full bg-muted ml-2" /> val state
              </span>
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 grid-cols-1 lg:grid-cols-3">
        <Panel defaultCollapsed title="Similarity margin" info="The similarity margin per epoch: how much closer texts sit to their correct option than to the best wrong one. Above the zero line means correct options are winning on average, and a rising curve means labels are separating." sub="Own option minus best other, averaged over labels.">
          <MiniChart data={rows} xMax={xMax} height={120} series={[{ key: "margin", name: "margin", color: "var(--c3)" }]} yFmt={(v) => v.toFixed(2)} refY={0} scrubEpoch={curEpoch} onScrub={scrub} />
        </Panel>
        <Panel defaultCollapsed title="Learning rate" info="How large each training update is. It ramps up over the first 10% of training (warm-up), then smoothly decays (cosine) so the model settles instead of bouncing around. This follows a fixed schedule, so its shape isn't a sign of good or bad training." sub="Warm-up then cosine decay.">
          <MiniChart data={rows} xMax={xMax} height={120} series={[{ key: "lr", name: "lr", color: "var(--c7)" }]} yFmt={(v) => v.toExponential(0)} scrubEpoch={curEpoch} onScrub={scrub} />
        </Panel>
        <Panel defaultCollapsed title="Logit scale" info="The learned temperature that turns similarities into probabilities. As it rises, predictions become more confident. Check the Calibration panel on the Evaluation tab to see whether that confidence is justified." sub="Learned temperature; higher means sharper probabilities.">
          <MiniChart data={rows} xMax={xMax} height={120} series={[{ key: "logit_scale", name: "logit scale", color: "var(--c4)" }]} yFmt={(v) => v.toFixed(0)} scrubEpoch={curEpoch} onScrub={scrub} />
        </Panel>
      </div>
      {task && <p className="text-[11.5px] text-muted">Question: “{task.instructions}”</p>}
    </div>
  );
}
