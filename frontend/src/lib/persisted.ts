import { useCallback, useState } from "react";

/** useState that survives reloads. Storage can be unavailable (private mode), so every access is guarded. */
export function usePersisted<T>(key: string | null, initial: T): [T, (v: T | ((p: T) => T)) => void] {
  const [v, setV] = useState<T>(() => {
    if (!key) return initial;
    try {
      const raw = localStorage.getItem(`forge:${key}`);
      return raw == null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (next: T | ((p: T) => T)) =>
      setV((prev) => {
        const val = typeof next === "function" ? (next as (p: T) => T)(prev) : next;
        if (key) {
          try {
            localStorage.setItem(`forge:${key}`, JSON.stringify(val));
          } catch {
            /* ignore */
          }
        }
        return val;
      }),
    [key],
  );
  return [v, set];
}
