import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FolderGit2, MessageSquareText, Plus, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { qk, useProjects } from "../api/queries";
import { Button, Empty, Spinner } from "../components/ui";
import { StateBadge } from "../components/StateBadge";
import { PipelineTicks } from "../components/PipelineTicks";
import { NewProjectDialog } from "./NewProjectDialog";
import { providerLabel } from "../components/ProviderPicker";
import { timeAgo } from "../lib/format";

export function ProjectsPage() {
  const { data, isLoading, isError, error } = useProjects();
  const [open, setOpen] = useState(false);
  const [params, setParams] = useSearchParams();
  useEffect(() => {
    if (params.get("new") === "1") {
      setOpen(true);
      setParams({}, { replace: true });
    }
  }, [params, setParams]);
  const qc = useQueryClient();
  const del = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  });

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[1120px] mx-auto px-6 py-8">
        <div className="flex items-end justify-between gap-4 mb-6">
          <div>
            <h1 className="font-display text-[30px] font-normal tracking-[-0.03em] leading-none">Experiments</h1>
            <p className="text-[13.5px] text-ink-2 mt-3 max-w-[64ch] leading-relaxed">
              Each experiment finds one decision in a codebase that a frozen encoder (you pick it per experiment; Forge recommends one) plus a small trained head can answer in milliseconds, then curates, trains, evaluates and packages it.
            </p>
          </div>
          <Button variant="primary" icon={<Plus size={15} />} onClick={() => setOpen(true)}>
            New experiment
          </Button>
        </div>

        {isLoading && (
          <div className="flex items-center gap-2 text-ink-2 py-10 justify-center">
            <Spinner /> Loading experiments…
          </div>
        )}
        {isError && (
          <Empty title="Can't reach the backend">
            {(error as Error).message}. Start it with <code className="font-mono text-[12px]">cd backend && uv run forge-api</code>.
          </Empty>
        )}
        {data && data.length === 0 && (
          <div className="glass border border-dashed border-line rounded-[14px]">
            <Empty
              icon={<FolderGit2 size={28} />}
              title="No experiments yet"
              action={
                <Button variant="primary" icon={<Plus size={15} />} onClick={() => setOpen(true)}>
                  New experiment
                </Button>
              }
            >
              Link a repository or describe a task. The agent proposes where a CLM fits before anything is trained.
            </Empty>
          </div>
        )}
        {data && data.length > 0 && (
          <div className="glass border border-line rounded-[14px] overflow-hidden">
            <div className="grid grid-cols-[minmax(0,2fr)_140px_110px_86px_minmax(0,1.4fr)_84px_32px] gap-4 px-4 h-9 items-center text-[12px] text-muted border-b border-line">
              <span>Experiment</span>
              <span>State</span>
              <span>Progress</span>
              <span className="text-right">Test metric</span>
              <span>Model</span>
              <span className="text-right">Updated</span>
              <span />
            </div>
            {data.map((p) => (
              <div
                key={p.id}
                className="proximity group relative grid grid-cols-[minmax(0,2fr)_140px_110px_86px_minmax(0,1.4fr)_84px_32px] gap-4 px-4 py-3.5 items-center border-b border-line-soft last:border-b-0 transition-colors hover:bg-accent-soft/40"
              >
                <Link to={`/experiments/${p.id}/pipeline`} className="min-w-0 after:absolute after:inset-0">
                  <div className="font-medium text-[14px] truncate">{p.name}</div>
                  <div className="text-[12px] text-muted truncate flex items-center gap-1.5 mt-0.5">
                    {p.repo ? <FolderGit2 size={12} className="shrink-0" /> : <MessageSquareText size={12} className="shrink-0" />}
                    <span className={p.repo ? "font-mono text-[11.5px] truncate" : "truncate"}>{p.repo ?? p.task}</span>
                  </div>
                </Link>
                <div>
                  <StateBadge state={p.state} />
                </div>
                <PipelineTicks state={p.state} lastMain={p.state === "no_fit" ? "proposed" : p.state === "errored" ? "curating" : undefined} />
                <div className="text-right tnum text-[13px]">{p.metric != null ? p.metric.value.toFixed(3) : <span className="text-muted">—</span>}</div>
                <div className="text-[12px] text-ink-2 truncate">
                  {p.model}
                  <span className="text-muted"> on {providerLabel(p.provider)}</span>
                </div>
                <div className="text-right text-[12px] text-muted tnum">{timeAgo(p.updated_at)}</div>
                <button
                  aria-label={`Delete ${p.name}`}
                  title="Delete experiment"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm(`Delete experiment “${p.name}” and its data?`)) del.mutate(p.id);
                  }}
                  className="relative z-10 p-1.5 rounded-[5px] text-muted opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-bad hover:bg-bad-soft cursor-pointer"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      <NewProjectDialog open={open} onClose={() => setOpen(false)} />
    </div>
  );
}
