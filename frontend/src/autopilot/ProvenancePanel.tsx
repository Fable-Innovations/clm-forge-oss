import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { Button, Chip, Input } from "../components/ui";

const CLASS_TONE: Record<string, "good" | "accent" | "warn"> = { synthetic_only: "good", real_shaped_synthetic: "accent", includes_user_data: "warn" };

/** What a promoted head was trained on (its provenance.json): synthetic only, shaped on real rows, or user data. */
export function ProvenancePanel({ heads }: { heads: string[] }) {
  const [head, setHead] = useState(heads[0] ?? "");
  const [asked, setAsked] = useState<string | null>(heads[0] ?? null);
  const q = useQuery({ queryKey: ["autopilot", "provenance", asked], queryFn: () => api.provenance(asked!), enabled: !!asked, retry: false });
  const p = q.data?.provenance;
  return (
    <div className="space-y-2 text-[12.5px]">
      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setAsked(head.trim() || null);
        }}
      >
        <Input aria-label="Head" list="provenance-heads" value={head} onChange={(e) => setHead(e.target.value)} placeholder="head id" className="w-[260px]" />
        <datalist id="provenance-heads">
          {heads.map((h) => (
            <option key={h} value={h} />
          ))}
        </datalist>
        <Button size="sm" type="submit">
          Show
        </Button>
      </form>
      {q.isError && <p className="text-muted">{(q.error as Error).message}</p>}
      {p && (
        <div className="space-y-1">
          <p className="flex items-center gap-2">
            <Chip tone={CLASS_TONE[p.class ?? ""] ?? "accent"}>{p.class ?? "unclassified"}</Chip>
            <span className="font-mono text-[11.5px] text-muted truncate">{q.data!.path}</span>
          </p>
          <pre className="font-mono text-[11.5px] bg-sunken/60 border border-line rounded-[10px] p-2 overflow-x-auto max-h-[260px]">{JSON.stringify(p, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
