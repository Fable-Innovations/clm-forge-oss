import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { Bot, Command, FlaskConical, FolderGit2, Moon, Settings2, Sun, Tags } from "lucide-react";
import { useHealth } from "../api/queries";
import type { Health, HealthEncoder } from "../api/types";
import { useTheme } from "../lib/theme";
import { useProximityLight } from "../lib/proximity";
import { cn } from "../lib/format";
import { CommandPalette } from "./CommandPalette";

/** Two bodies in orbit: a state (ion) and an option (ember), joined by the filament. */
function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
      <defs>
        <linearGradient id="forge-film" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#5ce1e6" />
          <stop offset="0.35" stopColor="var(--state)" />
          <stop offset="0.7" stopColor="#c792ff" />
          <stop offset="1" stopColor="var(--option)" />
        </linearGradient>
        <radialGradient id="forge-core-s"><stop offset="0" stopColor="#fff" stopOpacity=".9" /><stop offset=".45" stopColor="var(--state)" /><stop offset="1" stopColor="var(--state)" stopOpacity="0" /></radialGradient>
        <radialGradient id="forge-core-o"><stop offset="0" stopColor="#fff" stopOpacity=".9" /><stop offset=".45" stopColor="var(--option)" /><stop offset="1" stopColor="var(--option)" stopOpacity="0" /></radialGradient>
      </defs>
      <ellipse cx="16" cy="16" rx="13" ry="7" fill="none" stroke="url(#forge-film)" strokeWidth="1.2" transform="rotate(-28 16 16)" />
      <circle cx="9.5" cy="11" r="5" fill="url(#forge-core-s)" />
      <circle cx="22.5" cy="21" r="5" fill="url(#forge-core-o)" />
    </svg>
  );
}

/** Every encoder an experiment or autopilot target trains on (there is no app-wide encoder). */
function encodersItem(data: Health | undefined, isLoading: boolean): { label: string; ok: boolean | null; title: string; extra: string } {
  if (!data) return { label: "Encoders", ok: isLoading ? null : false, title: "checking…", extra: "" };
  const list = data.encoders ?? [];
  const unregistered = data.unregistered_encoders ?? [];
  const bad = list.filter((e) => e.error);
  const users = (e: HealthEncoder) => {
    const p = e.used_by.projects.length, t = e.used_by.targets.length;
    return [p ? `${p} experiment${p > 1 ? "s" : ""}` : "", t ? `${t} target${t > 1 ? "s" : ""}` : ""].filter(Boolean).join(", ");
  };
  const lines = [
    ...list.map((e) => `${e.name}: ${e.error ?? (e.loaded ? "loaded" : e.cached ? "not loaded (loads on first use)" : "not downloaded yet")} · used by ${users(e)}`),
    ...unregistered.map((u) => `${u}: not a registered encoder`),
  ];
  const loaded = list.filter((e) => e.loaded).length;
  return {
    label: "Encoders",
    ok: bad.length || unregistered.length ? false : list.length ? true : null,
    title: lines.length ? lines.join("\n") : "No experiment or autopilot target uses an encoder yet",
    extra: list.length ? `${list.length} in use${loaded ? `, ${loaded} loaded` : ""}` : "none in use",
  };
}

function HealthIndicator() {
  const { data, isError, isLoading } = useHealth();
  const enc = encodersItem(data, isLoading);
  const items: { label: string; ok: boolean | null; title: string; extra?: string }[] = isError
    ? [{ label: "Backend offline", ok: false, title: "GET /api/health failed — is `uv run forge-api` running on :8765?" }]
    : [
        enc,
        { label: "Claude", ok: isLoading ? null : !!data?.providers.claude, title: data?.providers.claude ? "Claude subscription login found" : "Claude CLI login not found" },
        { label: "OpenAI", ok: isLoading ? null : !!data?.providers.openai, title: data?.providers.openai ? "OpenAI API key or compatible server configured" : "No OpenAI API key (add one in Settings)" },
        {
          label: "Ollama",
          ok: isLoading ? null : !!data?.providers.ollama,
          title: data?.providers.ollama ? `models: ${data.providers.ollama_models.join(", ")}` : "Ollama not reachable",
        },
      ];
  return (
    <div className="flex items-center gap-1 text-[12px] text-ink-2">
      {items.map((it) => {
        const c = it.ok == null ? "var(--axis)" : it.ok ? "var(--good-mark)" : "var(--bad-mark)";
        return (
          <span key={it.label} title={it.title} aria-label={`${it.label}: ${it.title}`} className="inline-flex items-center gap-1.5 h-7 px-2 rounded-full cursor-default hover:bg-sunken/60">
            <span className="w-[7px] h-[7px] rounded-full" style={{ background: c, boxShadow: it.ok ? `0 0 8px ${c}` : undefined }} />
            {it.label}
            {it.extra && <span className="text-muted hidden xl:inline">{it.extra}</span>}
          </span>
        );
      })}
    </div>
  );
}

const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

export function Shell() {
  const { resolved, toggle } = useTheme();
  const [palette, setPalette] = useState(false);
  useProximityLight();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const link = ({ isActive }: { isActive: boolean }) =>
    cn(
      "proximity inline-flex items-center gap-1.5 h-8 px-3 rounded-full text-[13px] font-medium transition-[color,background-color,box-shadow]",
      isActive ? "bg-accent-soft text-ink shadow-[inset_0_0_0_1px_rgba(var(--glow),0.3)]" : "text-ink-2 hover:text-ink",
    );
  return (
    <div className="h-full flex flex-col">
      <div aria-hidden className="field-sense" />
      <header className="glass relative h-13 shrink-0 flex items-center gap-4 px-4 border-b border-line-soft">
        <NavLink to="/" className="flex items-center gap-2.5 mr-2 rounded-full">
          <Logo />
          <span className="font-display text-[15px] font-medium tracking-[-0.02em]">CLM Forge</span>
        </NavLink>
        <nav className="flex items-center gap-1">
          <NavLink to="/" end className={link}>
            <FolderGit2 size={14} /> Experiments
          </NavLink>
          <NavLink to="/playground" className={link}>
            <FlaskConical size={14} /> Playground
          </NavLink>
          <NavLink to="/labelling" className={link}>
            <Tags size={14} /> Labelling
          </NavLink>
          <NavLink to="/autopilot" className={link}>
            <Bot size={14} /> Autopilot
          </NavLink>
          <NavLink to="/settings" className={link}>
            <Settings2 size={14} /> Settings
          </NavLink>
        </nav>
        <div className="flex-1" />
        <button
          onClick={() => setPalette(true)}
          className="proximity proximity-rim hidden md:inline-flex items-center gap-2 h-8 pl-3 pr-1.5 rounded-full border border-line bg-sunken/50 text-[12.5px] text-muted hover:text-ink-2 cursor-pointer min-w-[220px]"
        >
          Jump to…
          <span className="flex-1" />
          <kbd className="inline-flex items-center gap-0.5 h-5 px-1.5 rounded-full border border-line text-[11px] text-ink-2">
            {isMac ? <Command size={10} /> : "Ctrl "}K
          </kbd>
        </button>
        <HealthIndicator />
        <button
          onClick={toggle}
          aria-label={`Switch to ${resolved === "dark" ? "light" : "dark"} theme`}
          title={`Switch to ${resolved === "dark" ? "light" : "dark"} theme`}
          className="proximity p-2 rounded-full text-ink-2 hover:text-ink cursor-pointer"
        >
          {resolved === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        {/* the filament: traces once on load, then rests as the bridge's lit edge */}
        <span aria-hidden className="boot-trace absolute left-0 right-0 -bottom-px h-px" style={{ background: "var(--film)", boxShadow: "0 0 10px rgba(var(--glow),.8)" }} />
      </header>
      <main className="boot-resolve flex-1 min-h-0">
        <Outlet />
      </main>
      <CommandPalette open={palette} onClose={() => setPalette(false)} onToggleTheme={toggle} dark={resolved === "dark"} />
    </div>
  );
}
