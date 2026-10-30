import { useEffect } from "react";

const SELECTOR = ".proximity, .proximity-rim";

/** One delegated pointer listener feeds the cursor position to whichever lit surface is under it,
 * as --mx/--my in element-local pixels. CSS draws the glow; nothing re-renders. */
export function useProximityLight() {
  useEffect(() => {
    if (window.matchMedia?.("(pointer: coarse)").matches) return;
    let frame = 0;
    let last: PointerEvent | null = null;
    const paint = () => {
      frame = 0;
      const e = last;
      if (!e) return;
      const root = document.documentElement.style;
      root.setProperty("--fx", `${e.clientX}px`);
      root.setProperty("--fy", `${e.clientY}px`);
      if (!(e.target instanceof Element)) return;
      let el: Element | null = e.target.closest(SELECTOR);
      // light every lit ancestor (a button inside a panel both respond)
      while (el) {
        const r = el.getBoundingClientRect();
        (el as HTMLElement).style.setProperty("--mx", `${e.clientX - r.left}px`);
        (el as HTMLElement).style.setProperty("--my", `${e.clientY - r.top}px`);
        el = el.parentElement?.closest(SELECTOR) ?? null;
      }
    };
    const onMove = (e: PointerEvent) => {
      last = e;
      if (!frame) frame = requestAnimationFrame(paint);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);
}
