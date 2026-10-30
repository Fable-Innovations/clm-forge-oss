import type { SelectHTMLAttributes } from "react";
import { useEncoders } from "../api/queries";
import type { Encoder, EncoderSource } from "../api/types";
import { Select } from "./ui";

/** "~3.0 GB", or "size not measured" when no peak was ever measured on this machine. */
export const encoderSize = (e: Encoder) => (e.approx_gb == null ? "size not measured" : `~${e.approx_gb.toFixed(1)} GB`);

export const encoderUsers = (e: Encoder) => e.used_by.projects.length + e.used_by.targets.length;

/** One line per encoder: name · backend · quant · ~GB · Downloaded/Not downloaded · used by N. */
export function encoderLabel(e: Encoder): string {
  const n = encoderUsers(e);
  return [e.name, e.backend, e.quant, encoderSize(e), e.cached ? "Downloaded" : "Not downloaded", n ? `used by ${n}` : null]
    .filter(Boolean)
    .join(" · ");
}

/** Where an experiment's encoder came from, in words (for its header's tooltip). */
export function encoderSourceText(src: EncoderSource | null | undefined): string {
  if (!src) return "";
  switch (src.from) {
    case "create":
      return "Picked when the experiment was created.";
    case "user":
      return "Changed by you.";
    case "checkpoint":
      return `Set by the per-experiment encoder migration from its trained checkpoint (${src.detail ?? "embed_model"}).`;
    case "settings":
      return `Set by the per-experiment encoder migration from the old app-wide setting (${src.detail ?? "settings"}).`;
  }
}

/**
 * Pick an encoder from the registry (GET /api/encoders). Used by the new-experiment dialog, an experiment's header
 * and proposal, and the autopilot target dialog (there is no app-wide default). `allowed` limits the choice (e.g. the encoders a head set's adapter can train on);
 * the rest stay listed but disabled, with `disabledReason`.
 */
export function EncoderSelect({
  value,
  onChange,
  allowed,
  disabledReason = "not supported here",
  placeholder = "Pick an encoder",
  className,
  ...rest
}: {
  value: string | null | undefined;
  onChange: (key: string) => void;
  allowed?: string[];
  disabledReason?: string;
  placeholder?: string;
  className?: string;
} & Omit<SelectHTMLAttributes<HTMLSelectElement>, "value" | "onChange">) {
  const { data, isLoading, isError, error } = useEncoders();
  if (isError) return <p className="text-[12.5px] text-bad">Couldn't load the encoders: {(error as Error).message}</p>;
  const list = data?.encoders ?? [];
  const known = !value || list.some((e) => e.key === value);
  return (
    <Select
      {...rest}
      className={className}
      value={value ?? ""}
      disabled={isLoading || rest.disabled}
      onChange={(e) => e.target.value && onChange(e.target.value)}
    >
      {(!value || !known) && (
        <option value="" disabled>
          {isLoading ? "Loading encoders…" : value && !known ? `${value} (not registered)` : placeholder}
        </option>
      )}
      {list.map((e) => {
        const off = !!allowed && !allowed.includes(e.key);
        return (
          <option key={e.key} value={e.key} disabled={off}>
            {encoderLabel(e)}
            {off ? ` (${disabledReason})` : ""}
          </option>
        );
      })}
    </Select>
  );
}
