import type { SnapshotPoint, TrainSnapshotEv } from "../api/types";

export interface AlignedSnapshot {
  epoch: number;
  points: SnapshotPoint[]; // aligned coordinates
}

type P = [number, number];

function centroid(ps: P[]): P {
  const n = ps.length || 1;
  return [ps.reduce((a, p) => a + p[0], 0) / n, ps.reduce((a, p) => a + p[1], 0) / n];
}

/** Best orthogonal 2x2 (rotation or reflection) mapping centred `a` onto centred `b`. Returns a function. */
function procrustes(a: P[], b: P[]): (p: P) => P {
  const fit = (src: P[]) => {
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < src.length; i++) {
      sxx += src[i][0] * b[i][0] + src[i][1] * b[i][1];
      sxy += src[i][0] * b[i][1] - src[i][1] * b[i][0];
    }
    const th = Math.atan2(sxy, sxx);
    const c = Math.cos(th);
    const s = Math.sin(th);
    const rot = (p: P): P => [c * p[0] - s * p[1], s * p[0] + c * p[1]];
    const err = src.reduce((acc, p, i) => {
      const q = rot(p);
      return acc + (q[0] - b[i][0]) ** 2 + (q[1] - b[i][1]) ** 2;
    }, 0);
    return { rot, err };
  };
  const r1 = fit(a);
  const refl = a.map((p) => [-p[0], p[1]] as P);
  const r2 = fit(refl);
  if (r2.err < r1.err) return (p: P) => r2.rot([-p[0], p[1]]);
  return r1.rot;
}

/**
 * Per-epoch PCA has arbitrary sign/rotation. Align every snapshot to its predecessor with an orthogonal
 * Procrustes fit on the option points (same options every epoch), falling back to state points by index.
 */
export function alignSnapshots(snaps: TrainSnapshotEv[]): AlignedSnapshot[] {
  const sorted = snaps.slice().sort((x, y) => x.epoch - y.epoch);
  const out: AlignedSnapshot[] = [];
  let prev: SnapshotPoint[] | null = null;
  for (const s of sorted) {
    const pts = s.points;
    const c = centroid(pts.map((p) => [p.x, p.y] as P));
    let centred = pts.map((p) => ({ ...p, x: p.x - c[0], y: p.y - c[1] }));
    if (prev) {
      const keyOf = (p: SnapshotPoint, i: number) => (p.kind === "option" ? `o:${p.label}:${p.text}` : `s:${i}`);
      const prevMap = new Map<string, SnapshotPoint>();
      let si = 0;
      prev.forEach((p) => prevMap.set(keyOf(p, p.kind === "state" ? si++ : 0), p));
      const opts = centred.filter((p) => p.kind === "option");
      let pairsA: P[] = [];
      let pairsB: P[] = [];
      for (const p of opts) {
        const q = prevMap.get(keyOf(p, 0));
        if (q) {
          pairsA.push([p.x, p.y]);
          pairsB.push([q.x, q.y]);
        }
      }
      if (pairsA.length < 2) {
        pairsA = [];
        pairsB = [];
        let i = 0;
        for (const p of centred) {
          if (p.kind !== "state") continue;
          const q = prevMap.get(`s:${i++}`);
          if (q) {
            pairsA.push([p.x, p.y]);
            pairsB.push([q.x, q.y]);
          }
        }
      }
      if (pairsA.length >= 2) {
        const f = procrustes(pairsA, pairsB);
        centred = centred.map((p) => {
          const [x, y] = f([p.x, p.y]);
          return { ...p, x, y };
        });
      }
    }
    out.push({ epoch: s.epoch, points: centred });
    prev = centred;
  }
  return out;
}

/** Symmetric extent covering ~all points across all snapshots (stable axes so motion is real). */
export function globalExtent(snaps: AlignedSnapshot[]): number {
  const vals: number[] = [];
  for (const s of snaps) for (const p of s.points) vals.push(Math.abs(p.x), Math.abs(p.y));
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  return (vals[Math.floor(vals.length * 0.985)] || vals[vals.length - 1] || 1) * 1.08;
}

/** Diagonal margin: mean over rows of (sim[gold] − max other). Positive = rows prefer their own option. */
export function simMargin(values: number[][], rowLabels: string[], colLabels: string[]): number {
  let acc = 0;
  let n = 0;
  rowLabels.forEach((rl, i) => {
    const j = colLabels.indexOf(rl);
    if (j < 0) return;
    const others = values[i].filter((_, k) => k !== j);
    acc += values[i][j] - Math.max(...others);
    n++;
  });
  return n ? acc / n : 0;
}
