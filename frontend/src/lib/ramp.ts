import { cssVar } from "./colors";

function hexToRgb(h: string): [number, number, number] {
  const s = h.replace("#", "");
  const f = s.length === 3 ? s.split("").map((c) => c + c).join("") : s;
  const n = parseInt(f, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Continuous sequential ramp from the theme's --s0..--s7 steps. Returns (t in 0..1) => css rgb. */
export function makeRamp(): { color: (t: number) => string; dark: (t: number) => boolean } {
  const steps = Array.from({ length: 8 }, (_, i) => hexToRgb(cssVar(`--s${i}`)));
  const lum = (c: [number, number, number]) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
  const at = (t: number): [number, number, number] => {
    const x = Math.max(0, Math.min(1, t)) * (steps.length - 1);
    const i = Math.min(steps.length - 2, Math.floor(x));
    const f = x - i;
    const a = steps[i];
    const b = steps[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };
  return {
    color: (t) => {
      const c = at(t);
      return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
    },
    dark: (t) => lum(at(t)) < 0.45,
  };
}
