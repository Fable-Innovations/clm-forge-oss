import { useCallback, useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, KeyRound } from "lucide-react";
import { api } from "../api/client";
import { qk, useSettings } from "../api/queries";
import { ProviderPicker } from "../components/ProviderPicker";
import type { Settings } from "../api/types";
import { Button, Field, Input, Panel, Spinner } from "../components/ui";

const THRESHOLD_HINTS: Record<string, string> = {
  min_examples: "Curation must produce at least this many rows before training (guard on the curated event).",
  default_success_threshold: "Default pass bar proposed for new opportunities.",
  fallback_confidence: "Confidence floor written into INTEGRATION.md for the fallback path.",
};

export function SettingsPage() {
  const { data, isLoading, isError, error } = useSettings();
  const [s, setS] = useState<Settings | null>(null);
  const qc = useQueryClient();
  useEffect(() => {
    if (data) setS(structuredClone(data));
  }, [data]);
  const save = useMutation({
    mutationFn: (v: Settings) => api.saveSettings(v),
    onSuccess: (v) => {
      qc.setQueryData(qk.settings, v);
      setKey("");
      // a new key or URL changes what the providers list
      qc.invalidateQueries({ queryKey: qk.providers });
      qc.invalidateQueries({ queryKey: qk.health });
    },
  });
  const [key, setKey] = useState("");
  const pick = useCallback((provider: string, model: string) => setS((cur) => (cur ? { ...cur, provider, model } : cur)), []);
  const dirty = (s && data && JSON.stringify(s) !== JSON.stringify(data)) || !!key.trim();

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[760px] mx-auto px-6 py-8">
        <h1 className="font-display text-[30px] font-normal tracking-[-0.03em] leading-none">Settings</h1>
        <p className="text-[13.5px] text-ink-2 mt-1 mb-6">
          Defaults for new experiments. Running experiments keep the provider and model they started with. There is no app-wide encoder: each experiment and
          autopilot target picks its own when it is created.
        </p>
        {isLoading && (
          <div className="flex items-center gap-2 text-ink-2">
            <Spinner /> Loading…
          </div>
        )}
        {isError && <p className="text-bad text-[13px]">Couldn't load settings: {(error as Error).message}</p>}
        {s && (
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              save.mutate({ ...s, openai_api_key: key.trim() || "" });
            }}
          >
            <Panel title="Agent" sub="Which LLM analyses code, generates and labels data, and makes review decisions.">
              <ProviderPicker provider={s.provider} model={s.model} onChange={pick} />
            </Panel>
            <Panel title="Connections" sub="Where the providers are reached. Model lists refresh after you save.">
              <div className="grid grid-cols-2 gap-3">
                <Field
                  label="OpenAI API key"
                  hint={
                    data?.openai_api_key_set && !data.openai_api_key_applies
                      ? `The saved key was entered for ${data.openai_api_key_url || "api.openai.com"} and is not sent to this URL. Enter a key for this server.`
                      : data?.openai_api_key_set
                      ? "A key is saved and only sent to the URL it was entered for. Type a new one to replace it."
                      : data?.openai_api_key_env
                        ? "Using OPENAI_API_KEY from the server's environment."
                        : "Stored in output/settings.json (readable only by you) and never sent back to the browser."
                  }
                >
                  <div className="flex gap-2">
                    <Input
                      type="password"
                      autoComplete="off"
                      className="font-mono text-[12.5px]"
                      value={key}
                      onChange={(e) => setKey(e.target.value)}
                      placeholder={data?.openai_api_key_set ? "•••••••• saved" : "sk-…"}
                    />
                    {data?.openai_api_key_set && (
                      <Button type="button" variant="danger" size="sm" className="h-8.5" icon={<KeyRound size={13} />} onClick={() => save.mutate({ ...s, openai_api_key: null })}>
                        Remove
                      </Button>
                    )}
                  </div>
                </Field>
                <Field label="OpenAI base URL" hint="Leave empty for api.openai.com, or point at any OpenAI-compatible server.">
                  <Input className="font-mono text-[12.5px]" value={s.openai_base_url ?? ""} placeholder="https://api.openai.com/v1" onChange={(e) => setS({ ...s, openai_base_url: e.target.value })} />
                </Field>
                <Field label="Ollama URL" className="col-span-2">
                  <Input className="font-mono text-[12.5px]" value={s.ollama_url} onChange={(e) => setS({ ...s, ollama_url: e.target.value })} />
                </Field>
              </div>
            </Panel>
            <Panel title="Guards and thresholds" sub="Enforced by the rules layer, not by the agent.">
              <div className="grid grid-cols-2 gap-3">
                <Field label="Max curate → train → evaluate attempts" hint="After this many failed gates the agent can only give up.">
                  <Input type="number" min={1} max={10} value={s.max_attempts} onChange={(e) => setS({ ...s, max_attempts: Number(e.target.value) })} />
                </Field>
                {Object.entries(s.thresholds ?? {}).map(([k, v]) => (
                  <Field key={k} label={k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " ")} hint={THRESHOLD_HINTS[k]}>
                    <Input type="number" step="any" value={v} onChange={(e) => setS({ ...s, thresholds: { ...(s.thresholds ?? {}), [k]: Number(e.target.value) } })} />
                  </Field>
                ))}
              </div>
            </Panel>
            <div className="flex items-center justify-end gap-3">
              {save.isError && <span className="text-[12.5px] text-bad">{(save.error as Error).message}</span>}
              {save.isSuccess && !dirty && (
                <span className="text-[12.5px] text-good inline-flex items-center gap-1">
                  <Check size={13} /> Saved
                </span>
              )}
              <Button type="button" variant="ghost" disabled={!dirty} onClick={() => data && setS(structuredClone(data))}>
                Discard
              </Button>
              <Button type="submit" variant="primary" disabled={!dirty || save.isPending}>
                {save.isPending ? "Saving…" : "Save settings"}
              </Button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
