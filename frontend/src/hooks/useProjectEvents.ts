import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { ForgeEvent } from "../api/types";
import { applyEvents, emptyStore, type ProjectStore } from "../store/projectStore";
import { projectLsid, qk } from "../api/queries";

export type ConnStatus = "connecting" | "open" | "reconnecting";

/**
 * Subscribes to /api/projects/{pid}/events. The server replays history on connect, so a reload or
 * reconnect rebuilds the store; events are de-duplicated by seq. Events are batched per animation frame.
 */
export function useProjectEvents(pid: string): { store: ProjectStore; status: ConnStatus } {
  const [store, setStore] = useState<ProjectStore>(emptyStore);
  const [status, setStatus] = useState<ConnStatus>("connecting");
  const qc = useQueryClient();
  const buf = useRef<ForgeEvent[]>([]);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    setStore(emptyStore());
    setStatus("connecting");
    let es: EventSource | null = null;
    let closed = false;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const flush = () => {
      raf.current = null;
      const batch = buf.current;
      buf.current = [];
      if (!batch.length) return;
      setStore((prev) => applyEvents(prev, batch));
      const types = new Set(batch.map((b) => b.type));
      if (types.has("state")) {
        qc.invalidateQueries({ queryKey: qk.project(pid) });
        qc.invalidateQueries({ queryKey: qk.projects });
      }
      if (types.has("proposal")) qc.invalidateQueries({ queryKey: qk.project(pid) });
      if (types.has("eval")) qc.invalidateQueries({ queryKey: qk.eval(pid) });
      if (types.has("package")) qc.invalidateQueries({ queryKey: qk.pkg(pid) });
      if (types.has("labels") || types.has("state")) {
        qc.invalidateQueries({ queryKey: qk.project(pid) });
        qc.invalidateQueries({ queryKey: qk.labelQueue(projectLsid(pid)) });
      }
      if (batch.some((b) => b.type === "curation" && b.phase === "done") || types.has("state"))
        qc.invalidateQueries({ queryKey: qk.dataset(pid) });
    };

    const connect = () => {
      if (closed) return;
      es = new EventSource(`/api/projects/${encodeURIComponent(pid)}/events`);
      es.onopen = () => {
        attempt = 0;
        setStatus("open");
      };
      es.onmessage = (m) => {
        let ev: ForgeEvent;
        try {
          ev = JSON.parse(m.data) as ForgeEvent;
        } catch {
          return;
        }
        buf.current.push(ev);
        if (raf.current == null) raf.current = requestAnimationFrame(flush);
      };
      es.onerror = () => {
        es?.close();
        if (closed) return;
        setStatus("reconnecting");
        const delay = Math.min(15_000, 800 * 2 ** attempt++);
        timer = setTimeout(connect, delay);
      };
    };
    connect();
    return () => {
      closed = true;
      es?.close();
      if (timer) clearTimeout(timer);
      if (raf.current != null) cancelAnimationFrame(raf.current);
      raf.current = null;
      buf.current = [];
    };
  }, [pid, qc]);

  return { store, status };
}
