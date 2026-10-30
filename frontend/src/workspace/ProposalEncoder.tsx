import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { EncoderRecommendation, ProjectDetail } from "../api/types";
import { EncoderSelect } from "../components/EncoderSelect";
import { Button, Field } from "../components/ui";

/**
 * The proposal's encoder pick: the experiment's own encoder, Forge's recommendation for the selected opportunity's
 * task (ProjectDetail.encoder_recommendations, forge.encoder_choice) and a one-click switch to it. Editable while the
 * proposal awaits a decision (PUT /api/projects/{id}/encoder); read-only after.
 */
export function ProposalEncoder({ pid, detail, opportunityId, editable }: { pid: string; detail: ProjectDetail | undefined; opportunityId: string; editable: boolean }) {
  const qc = useQueryClient();
  const rec: EncoderRecommendation | undefined = detail?.encoder_recommendations?.[opportunityId];
  const current = detail?.encoder?.key ?? null;
  const set = useMutation({
    mutationFn: (key: string) => api.setProjectEncoder(pid, key),
    onSuccess: (d) => qc.setQueryData(qk.project(pid), d),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.encoders }),
  });
  const differs = !!rec?.key && rec.key !== current;
  return (
    <div className="mt-3 pt-3 border-t border-line-soft" role="group" aria-label="Encoder for this proposal">
      <Field
        label="Encoder"
        hint={
          set.isError ? (
            <span className="text-bad">{(set.error as Error).message}</span>
          ) : rec ? (
            rec.key ? (
              <>Forge recommends {differs ? <b>{rec.name}</b> : "this one"} for this task: {rec.reason}.</>
            ) : (
              <>{rec.reason}. Pick one by hand.</>
            )
          ) : (
            "The frozen base model this experiment's head trains on."
          )
        }
      >
        <div className="flex gap-2 items-center">
          <EncoderSelect aria-label="Proposal encoder" className="flex-1" value={current} disabled={!editable || set.isPending} onChange={(k) => set.mutate(k)} placeholder="Pick this experiment's encoder" />
          {editable && differs && (
            <Button size="sm" icon={<Sparkles size={13} />} disabled={set.isPending} onClick={() => set.mutate(rec!.key!)}>
              Use recommended
            </Button>
          )}
        </div>
      </Field>
    </div>
  );
}
