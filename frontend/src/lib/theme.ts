import { useCallback, useEffect, useState } from "react";

export type ThemePref = "system" | "light" | "dark";

function systemDark() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
}

export function useTheme() {
  const [pref, setPref] = useState<ThemePref>(() => {
    try {
      const t = localStorage.getItem("forge-theme");
      if (t === "light" || t === "dark") return t;
    } catch {
      /* storage unavailable */
    }
    return "system";
  });
  const [sysDark, setSysDark] = useState(systemDark);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const h = () => setSysDark(mq.matches);
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }, []);
  useEffect(() => {
    const root = document.documentElement;
    if (pref === "system") delete root.dataset.theme;
    else root.dataset.theme = pref;
    try {
      if (pref === "system") localStorage.removeItem("forge-theme");
      else localStorage.setItem("forge-theme", pref);
    } catch {
      /* ignore */
    }
    window.dispatchEvent(new Event("forge-theme"));
  }, [pref]);
  const resolved: "light" | "dark" = pref === "system" ? (sysDark ? "dark" : "light") : pref;
  const toggle = useCallback(() => setPref(resolved === "dark" ? "light" : "dark"), [resolved]);
  return { pref, setPref, resolved, toggle };
}

/** Re-render when theme changes (for canvas charts reading CSS vars). */
export function useThemeVersion(): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    const bump = () => setV((x) => x + 1);
    window.addEventListener("forge-theme", bump);
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", bump);
    return () => {
      window.removeEventListener("forge-theme", bump);
      mq.removeEventListener("change", bump);
    };
  }, []);
  return v;
}
