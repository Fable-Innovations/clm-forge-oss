import { useCallback, useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { FolderGit2, MessageSquareText } from "lucide-react";
import { api } from "../api/client";
import { qk, useEncoderRecommendation, useProviders, useSettings } from "../api/queries";
import { EncoderSelect } from "../components/EncoderSelect";
import { ProviderPicker } from "../components/ProviderPicker";
import { Button, Dialog, Field, Input, Segmented, Textarea } from "../components/ui";

function repoError(v: string): string | null {
  if (!v.trim()) return null;
  const s = v.trim();
  if (/^https?:\/\/\S+$/.test(s) || /^git@\S+:\S+$/.test(s)) return null;
  if (s.startsWith("/") || s.startsWith("~/")) return null;
  return "Use an https git URL or an absolute local path (starting with /).";
}

export function NewProjectDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [mode, setMode] = useState<"repo" | "task">("repo");
  const [repo, setRepo] = useState("");
  const [task, setTask] = useState("");
  const [name, setName] = useState("");
  const { data: settings } = useSettings();
  const { data: providers } = useProviders();
  const [provider, setProvider] = useState("claude-subscription");
  const [model, setModel] = useState("");
  // the encoder this experiment trains on: Forge's recommendation until the user picks another (no app-wide default)
  const rec = useEncoderRecommendation(open);
  const [encoder, setEncoder] = useState<string | null>(null);
  const shownEncoder = encoder ?? rec.data?.key ?? null;
  const nav = useNavigate();
  const qc = useQueryClient();

  useEffect(() => {
    if (open && settings) {
      setProvider(settings.provider);
      setModel(settings.model);
    }
  }, [open, settings]);

  const pick = useCallback((p: string, m: string) => {
    setProvider(p);
    setModel(m);
  }, []);

  const create = useMutation({
    mutationFn: () =>
      api.createProject({
        name: name.trim() || undefined,
        repo: mode === "repo" ? repo.trim() || undefined : undefined,
        task: task.trim() || undefined,
        provider,
        model,
        encoder: shownEncoder!,
      }),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: qk.projects });
      onClose();
      setRepo("");
      setTask("");
      setName("");
      setEncoder(null);
      nav(`/experiments/${p.id}/pipeline`);
    },
  });

  const rErr = mode === "repo" ? repoError(repo) : null;
  const canSubmit =
    (mode === "repo" ? repo.trim() && !rErr : task.trim().length > 8) && !create.isPending && (!!model || provider.endsWith("-subscription")) && !!shownEncoder;
  const recommended = rec.data?.key && shownEncoder === rec.data.key;
  const pinfo = providers?.find((p) => p.id === provider);
  const providerDown = pinfo && !pinfo.available;

  return (
    <Dialog open={open} onClose={onClose} title="New experiment" width={580}>
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) create.mutate();
        }}
      >
        <p className="text-[13px] text-ink-2 -mt-1">
          The agent reads the code (read-only), finds decisions a small trained head could make, and proposes one. Nothing is trained until you accept.
        </p>
        <Segmented
          size="md"
          value={mode}
          onChange={setMode}
          options={[
            { value: "repo", label: <span className="inline-flex items-center gap-1.5"><FolderGit2 size={13} /> Link a repository</span> },
            { value: "task", label: <span className="inline-flex items-center gap-1.5"><MessageSquareText size={13} /> Describe a task</span> },
          ]}
        />
        {mode === "repo" ? (
          <>
            <Field label="Repository" hint={rErr ?? "https git URL (cloned shallow) or an absolute path on this machine."}>
              <Input
                autoFocus
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                placeholder="https://github.com/acme/helpdesk.git  or  /path/to/helpdesk"
                className="font-mono text-[12.5px]"
                aria-invalid={!!rErr}
              />
            </Field>
            <Field label="Focus (optional)" hint="Point the agent at a part of the codebase or a decision you have in mind.">
              <Textarea rows={2} value={task} onChange={(e) => setTask(e.target.value)} placeholder="e.g. the ticket router in app/routing is slow and costs money" />
            </Field>
          </>
        ) : (
          <Field label="Task" hint="Describe the decision: what text comes in, which answers are possible.">
            <Textarea
              autoFocus
              rows={4}
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="Classify inbound support tickets into billing, refunds, technical, account or shipping."
            />
          </Field>
        )}
        <Field label="Name">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="from repo" />
        </Field>
        <ProviderPicker provider={provider} model={model} onChange={pick} />
        <Field
          label="Encoder"
          hint={
            rec.isError
              ? `Couldn't get Forge's recommendation: ${(rec.error as Error).message}. Pick one.`
              : rec.data && !rec.data.key
                ? `${rec.data.reason}. Pick one by hand.`
                : recommended
                  ? `Recommended: ${rec.data!.reason}. The proposal recommends one for the task it finds; you can change it until training starts.`
                  : rec.data?.key
                    ? `Forge recommends ${rec.data.name}. This experiment keeps the encoder you pick; change it in its header until training starts.`
                    : "The frozen base model this experiment's head trains on."
          }
        >
          <EncoderSelect aria-label="Encoder" value={shownEncoder} onChange={setEncoder} placeholder={rec.isLoading ? "Finding the recommended encoder…" : "Pick an encoder"} />
        </Field>
        {providerDown && (
          <p className="text-[12.5px] text-warn bg-warn-soft border border-warn/30 rounded-[6px] px-3 py-2">
            {pinfo.detail} The run will fail at analysis until this is fixed.
          </p>
        )}
        {create.isError && <p className="text-[12.5px] text-bad">Couldn't create the experiment: {(create.error as Error).message}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" disabled={!canSubmit}>
            {create.isPending ? "Starting…" : "Start analysis"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
