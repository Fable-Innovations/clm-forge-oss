import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "../api/client";
import { qk, useProviders } from "../api/queries";
import type { ProviderInfo } from "../api/types";
import { Field, Select, Spinner } from "./ui";
import { cn } from "../lib/format";

export function providerLabel(id: string | null | undefined): string {
  return id === "ollama" ? "Ollama" : id === "openai" ? "OpenAI" : id?.startsWith("claude") ? "Claude" : id?.startsWith("codex") ? "Codex" : (id ?? "");
}

/** Provider and model selects. Models are whatever the provider lists right now (no hard-coded names);
 * the refresh button re-asks the providers, e.g. after pulling an Ollama model or adding an API key. */
export function ProviderPicker({
  provider,
  model,
  onChange,
  className,
}: {
  provider: string;
  model: string;
  onChange: (provider: string, model: string) => void;
  className?: string;
}) {
  const { data, isLoading, isFetching, isError, error } = useProviders();
  const qc = useQueryClient();
  const info: ProviderInfo | undefined = data?.find((p) => p.id === provider);
  const models = info?.models ?? [];
  const known = models.some((m) => m.id === model);

  // keep the chosen model valid for the chosen provider
  useEffect(() => {
    if (info && models.length && !known && !model) onChange(provider, models[0].id);
  }, [info, models, known, model, provider, onChange]);

  const refresh = async () => {
    const fresh = await api.providers(true);
    qc.setQueryData(qk.providers, fresh);
    qc.invalidateQueries({ queryKey: qk.health });
  };
  const current = models.find((m) => m.id === model);

  return (
    <div className={cn("grid grid-cols-2 gap-3", className)}>
      <Field label="Provider" hint={isError ? `Couldn't ask providers: ${(error as Error).message}` : info?.detail}>
        <Select
          value={provider}
          onChange={(e) => {
            const p = data?.find((x) => x.id === e.target.value);
            onChange(e.target.value, p?.models[0]?.id ?? "");
          }}
        >
          {(data ?? [{ id: provider, label: providerLabel(provider), available: true, detail: "", models: [] }]).map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
              {p.available ? "" : " (not available)"}
            </option>
          ))}
        </Select>
      </Field>
      <Field
        label={
          <span className="flex items-center justify-between">
            Model
            <button
              type="button"
              onClick={refresh}
              disabled={isFetching}
              title="Ask the providers for their current models"
              className="inline-flex items-center gap-1 text-[11.5px] font-normal text-muted hover:text-ink cursor-pointer disabled:opacity-50"
            >
              {isFetching ? <Spinner size={11} /> : <RefreshCw size={11} />} Refresh
            </button>
          </span>
        }
        hint={current?.description ?? (model && !known && models.length ? "Not in this provider's current list." : undefined)}
      >
        <Select value={model} onChange={(e) => onChange(provider, e.target.value)} disabled={isLoading || (!models.length && !model)}>
          {isLoading && <option value={model}>Loading models…</option>}
          {!isLoading && !models.length && !model && <option value="">No models available</option>}
          {model && !known && !isLoading && <option value={model}>{model}</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id} title={m.id}>
              {m.label}
            </option>
          ))}
        </Select>
      </Field>
    </div>
  );
}
