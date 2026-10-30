import { createContext, useContext } from "react";
import type { ProjectDetail, TaskSchema } from "../api/types";
import type { ProjectStore } from "../store/projectStore";

export interface WorkspaceCtx {
  pid: string;
  store: ProjectStore;
  detail: ProjectDetail | undefined;
  task: TaskSchema | null;
  labels: string[];
  colorOf: (label: string) => string;
  labelText: (label: string) => string;
  goTab: (tab: string) => void;
}

export const Ctx = createContext<WorkspaceCtx | null>(null);
export function useWs(): WorkspaceCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useWs outside workspace");
  return c;
}
