import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CalendarClock } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { AutopilotSchedule } from "../api/types";
import { Button, Input, Select } from "../components/ui";
import { clock } from "../lib/format";

export const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

/** Daily or weekly retraining cadence for one autopilot target (machine local time). */
export function ScheduleEditor({ targetId, schedule }: { targetId: string; schedule: AutopilotSchedule | null }) {
  const qc = useQueryClient();
  const [cadence, setCadence] = useState<"daily" | "weekly">(schedule?.cadence ?? "weekly");
  const [weekday, setWeekday] = useState<number>(schedule?.weekday ?? 6);
  const [time, setTime] = useState(schedule?.time_local ?? "03:00");
  const [enabled, setEnabled] = useState(schedule?.enabled ?? true);
  useEffect(() => {
    if (!schedule) return;
    setCadence(schedule.cadence);
    setWeekday(schedule.weekday ?? 6);
    setTime(schedule.time_local);
    setEnabled(schedule.enabled);
  }, [schedule]);
  const save = useMutation({
    mutationFn: () => api.saveSchedule(targetId, { cadence, weekday: cadence === "weekly" ? weekday : null, time_local: time, enabled }),
    onSettled: () => qc.invalidateQueries({ queryKey: qk.autopilotTargets }),
  });
  const dirty =
    !schedule || schedule.cadence !== cadence || (cadence === "weekly" && schedule.weekday !== weekday) || schedule.time_local !== time || schedule.enabled !== enabled;
  return (
    <form
      className="flex flex-wrap items-center gap-2 text-[12.5px]"
      aria-label={`Schedule for ${targetId}`}
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate();
      }}
    >
      <CalendarClock size={13} className="text-muted" />
      <Select aria-label="Cadence" value={cadence} onChange={(e) => setCadence(e.target.value as "daily" | "weekly")} className="w-[100px]">
        <option value="daily">Daily</option>
        <option value="weekly">Weekly</option>
      </Select>
      {cadence === "weekly" && (
        <Select aria-label="Weekday" value={weekday} onChange={(e) => setWeekday(Number(e.target.value))} className="w-[120px]">
          {WEEKDAYS.map((d, i) => (
            <option key={d} value={i}>
              {d}
            </option>
          ))}
        </Select>
      )}
      <span className="text-muted">at</span>
      <Input aria-label="Time" type="time" value={time} onChange={(e) => setTime(e.target.value)} className="w-[110px]" />
      <label className="inline-flex items-center gap-1.5 text-ink-2 cursor-pointer">
        <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> enabled
      </label>
      <Button size="sm" variant={dirty ? "primary" : "ghost"} type="submit" disabled={!dirty || save.isPending}>
        {save.isPending ? "Saving…" : schedule ? "Save schedule" : "Set schedule"}
      </Button>
      {schedule?.next_run && !save.isError && <span className="text-muted">next run {clock(schedule.next_run)}</span>}
      {save.isError && <span className="text-bad">{(save.error as Error).message}</span>}
    </form>
  );
}
