import { createContext, forwardRef, useContext, useEffect, useId, useLayoutEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode, type InputHTMLAttributes, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { createPortal } from "react-dom";
import { ChevronRight, Info, Maximize2, Minimize2, X } from "lucide-react";
import { usePersisted } from "../lib/persisted";
import { cn } from "../lib/format";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "option";

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md"; icon?: ReactNode }
>(function Button({ variant = "secondary", size = "md", icon, className, children, ...rest }, ref) {
  const v: Record<Variant, string> = {
    primary: "btn-ion text-accent-ink border border-transparent",
    secondary: "proximity proximity-rim bg-raised text-ink border border-line hover:text-ink",
    ghost: "proximity text-ink-2 hover:text-ink hover:bg-sunken/60 border border-transparent",
    danger: "proximity bg-raised text-bad border border-line hover:bg-bad-soft hover:border-bad/40 [--glow:255,90,120]",
    option: "bg-option text-white border border-transparent hover:brightness-110 shadow-[0_6px_22px_-8px_var(--option)]",
  };
  const s = size === "sm" ? "h-7 px-2.5 text-[12.5px] gap-1.5 rounded-[7px]" : "h-8.5 px-3.5 text-[13.5px] gap-2 rounded-[9px]";
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center font-medium whitespace-nowrap transition-[color,background-color,border-color,box-shadow,transform] active:translate-y-px disabled:opacity-45 disabled:pointer-events-none cursor-pointer select-none",
        v[variant],
        s,
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
});

/** True inside a panel that is open in focus view, so charts can use the extra room. */
const FocusCtx = createContext(false);
export const useFocusView = () => useContext(FocusCtx);

export function Panel({
  title,
  aside,
  children,
  className,
  bodyClass,
  sub,
  info,
  id,
  collapsible = true,
  defaultCollapsed = false,
  expandable = false,
}: {
  title?: ReactNode;
  /** Plain-language description shown in a hoverable info icon next to the title. */
  info?: ReactNode;
  aside?: ReactNode;
  sub?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  /** Key for remembering the collapsed state; defaults to the title when it is a string. */
  id?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  /** Adds a focus-view button that lifts the panel to fill the window. */
  expandable?: boolean;
}) {
  const key = id ?? (typeof title === "string" ? title : null);
  const canCollapse = collapsible && !!title;
  const [collapsed, setCollapsed] = usePersisted<boolean>(canCollapse && key ? `panel:${key}` : null, defaultCollapsed);
  const [expanded, setExpanded] = useState(false);
  const isCollapsed = canCollapse && collapsed && !expanded;
  const bodyId = useId();

  useEffect(() => {
    if (!canCollapse) return;
    const on = (e: Event) => setCollapsed((e as CustomEvent<boolean>).detail);
    window.addEventListener("forge:collapse-all", on);
    return () => window.removeEventListener("forge:collapse-all", on);
  }, [canCollapse, setCollapsed]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  return (
    <>
      {expanded && createPortal(<div className="fixed inset-0 z-[80] bg-[color-mix(in_oklab,var(--bg)_60%,transparent)] backdrop-blur-[8px]" onClick={() => setExpanded(false)} />, document.body)}
      <section
        className={cn(
          "glass proximity-rim border border-line rounded-[14px] min-w-0",
          isCollapsed && "self-start",
          expanded && "focus-view fixed inset-4 md:inset-8 z-[90] flex flex-col overflow-hidden",
          className,
        )}
        style={expanded ? { boxShadow: "var(--shadow-pop)", background: "var(--panel-solid)" } : undefined}
      >
        {(title || aside) && (
          <header className={cn("group/ph flex items-start justify-between gap-3 px-4 pt-3", isCollapsed ? "pb-3" : "pb-1")}>
            <div className="min-w-0 flex items-start gap-1.5">
              {canCollapse && !expanded && (
                <button
                  type="button"
                  onClick={() => setCollapsed((c) => !c)}
                  aria-expanded={!isCollapsed}
                  aria-controls={bodyId}
                  aria-label={isCollapsed ? `Expand ${typeof title === "string" ? title : "section"}` : `Collapse ${typeof title === "string" ? title : "section"}`}
                  className="mt-[1px] -ml-1 p-0.5 rounded-[5px] text-muted hover:text-ink hover:bg-sunken cursor-pointer"
                >
                  <ChevronRight size={14} className={cn("transition-transform duration-200", !isCollapsed && "rotate-90")} />
                </button>
              )}
              <div className="min-w-0">
                {title && (
                  <h3
                    className={cn("text-[13.5px] font-semibold text-ink leading-tight", canCollapse && !expanded && "cursor-pointer select-none")}
                    onClick={canCollapse && !expanded ? () => setCollapsed((c) => !c) : undefined}
                  >
                    {title}
                    {info && <InfoTip className="ml-1.5 align-[-2px]">{info}</InfoTip>}
                  </h3>
                )}
                {sub && !isCollapsed && <p className="text-[12px] text-muted mt-0.5 leading-snug">{sub}</p>}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {aside && !isCollapsed && aside}
              {expandable && (
                <button
                  type="button"
                  onClick={() => setExpanded((x) => !x)}
                  aria-label={expanded ? "Exit focus view" : "Open in focus view"}
                  title={expanded ? "Exit focus view (Esc)" : "Focus view"}
                  className={cn(
                    "p-1 rounded-[6px] text-muted hover:text-ink hover:bg-sunken cursor-pointer transition-opacity",
                    !expanded && "opacity-0 group-hover/ph:opacity-100 focus-visible:opacity-100 [@media(pointer:coarse)]:opacity-100",
                  )}
                >
                  {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                </button>
              )}
            </div>
          </header>
        )}
        <div id={bodyId} className={cn("collapse-body", isCollapsed && "is-collapsed", expanded && "flex-1 min-h-0 overflow-auto")} aria-hidden={isCollapsed || undefined}>
          <div className="min-h-0 overflow-hidden">
            {/* keep content mounted while collapsed so charts and inputs keep their state */}
            <div className={cn("px-4 pb-4 pt-2", bodyClass)} {...((isCollapsed ? { inert: "" } : {}) as object)}>
              <FocusCtx.Provider value={expanded}>{children}</FocusCtx.Provider>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

const field =
  "bg-raised border border-line rounded-[9px] text-[13.5px] text-ink placeholder:text-muted/80 transition-[border-color,box-shadow] focus:outline-none focus:border-accent focus:shadow-[0_0_0_3px_rgba(var(--glow),0.18),0_0_18px_-4px_rgba(var(--glow),0.6)] disabled:opacity-60";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...p }, ref) {
  return <input ref={ref} className={cn(field, "h-8.5 px-2.5", !/(^|\s)w-/.test(className ?? "") && "w-full", className)} {...p} />;
});
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea(
  { className, ...p },
  ref,
) {
  return <textarea ref={ref} className={cn(field, "w-full px-2.5 py-2 leading-snug resize-y", className)} {...p} />;
});
export function Select({ className, children, ...p }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn(field, "h-8.5 px-2 pr-7 cursor-pointer", !/(^|\s)w-/.test(className ?? "") && "w-full", className)} {...p}>
      {children}
    </select>
  );
}

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={cn("block min-w-0", className)}>
      <span className="block text-[12.5px] font-medium text-ink-2 mb-1">{label}</span>
      {children}
      {hint && <span className="block text-[11.5px] text-muted mt-1 leading-snug">{hint}</span>}
    </label>
  );
}

export function Chip({ children, tone = "neutral", className, title }: { children: ReactNode; tone?: "neutral" | "accent" | "good" | "warn" | "bad" | "option"; className?: string; title?: string }) {
  const t = {
    neutral: "bg-sunken text-ink-2 border-line",
    accent: "bg-accent-soft text-accent border-accent/25",
    good: "bg-good-soft text-good border-good/25",
    warn: "bg-warn-soft text-warn border-warn/30",
    bad: "bg-bad-soft text-bad border-bad/30",
    option: "bg-option-soft text-option border-option/30",
  }[tone];
  return (
    <span title={title} className={cn("inline-flex items-center gap-1 h-5.5 px-1.5 rounded-[6px] border text-[11.5px] font-medium whitespace-nowrap", t, className)}>
      {children}
    </span>
  );
}

export function Dialog({
  open,
  onClose,
  title,
  children,
  width = 560,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  width?: number;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      className="glass film-edge m-auto text-ink border border-transparent rounded-[18px] p-0 overflow-visible"
      style={{ width: `min(${width}px, calc(100vw - 32px))`, boxShadow: "var(--shadow-pop)" }}
    >
      {open && (
        <div>
          <header className="flex items-center justify-between px-5 pt-4 pb-2">
            <h2 className="font-display text-[16px] font-medium tracking-[-0.01em]">{title}</h2>
            <button aria-label="Close" onClick={onClose} className="p-1 rounded text-muted hover:text-ink hover:bg-sunken cursor-pointer">
              <X size={16} />
            </button>
          </header>
          <div className="px-5 pb-5">{children}</div>
        </div>
      )}
    </dialog>
  );
}

/** Horizontal meter. Track is a lighter step of the fill. */
export function Meter({ value, color = "var(--accent)", height = 6, className, marker }: { value: number; color?: string; height?: number; className?: string; marker?: number }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className={cn("relative w-full rounded-full", className)} style={{ height, background: `color-mix(in oklab, ${color} 16%, transparent)` }}>
      <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500" style={{ width: `${v * 100}%`, background: color }} />
      {marker != null && (
        <div className="absolute -top-[3px] -bottom-[3px] w-[2px] bg-ink rounded" style={{ left: `calc(${Math.max(0, Math.min(1, marker)) * 100}% - 1px)` }} />
      )}
    </div>
  );
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: ReactNode; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-16 px-6">
      {icon && <div className="text-muted mb-3">{icon}</div>}
      <h3 className="text-[15px] font-semibold">{title}</h3>
      {children && <div className="text-[13px] text-ink-2 mt-1.5 max-w-[46ch] leading-relaxed">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" className="animate-spin text-current" aria-hidden>
      <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2" />
      <path d="M14.5 8A6.5 6.5 0 0 0 8 1.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

/** Small segmented control. */
export function Segmented<T extends string>({ value, onChange, options, size = "sm" }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode }[]; size?: "sm" | "md" }) {
  return (
    <div className="inline-flex p-0.5 bg-sunken border border-line rounded-[9px]" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "px-2.5 rounded-[7px] font-medium cursor-pointer transition-[color,background-color,box-shadow]",
            size === "sm" ? "h-6 text-[12px]" : "h-7 text-[13px]",
            value === o.value ? "bg-raised text-ink shadow-[0_0_0_1px_var(--line),0_0_14px_-4px_rgba(var(--glow),0.7)]" : "text-muted hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Small (i) icon that shows a description on hover or keyboard focus. The popover is portalled and
 * fixed-positioned so panels with overflow clipping can't cut it off. */
export function InfoTip({ children, className, label = "More info" }: { children: ReactNode; className?: string; label?: string }) {
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    if (!open || !ref.current) return;
    const place = () => {
      const a = ref.current!.getBoundingClientRect();
      const w = tip.current?.offsetWidth ?? 300;
      const h = tip.current?.offsetHeight ?? 60;
      const left = Math.max(8, Math.min(window.innerWidth - w - 8, a.left + a.width / 2 - w / 2));
      const below = a.bottom + 6;
      const top = below + h > window.innerHeight - 8 ? Math.max(8, a.top - h - 6) : below;
      setPos({ left, top });
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open]);

  return (
    <span
      ref={ref}
      tabIndex={0}
      role="button"
      aria-label={label}
      aria-describedby={open ? id : undefined}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
      className={cn("inline-flex items-center text-muted hover:text-ink-2 focus:text-ink-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 rounded-full cursor-help align-middle", className)}
    >
      <Info size={13} aria-hidden />
      {open &&
        createPortal(
          <div
            ref={tip}
            id={id}
            role="tooltip"
            style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999, boxShadow: "var(--shadow-pop)" }}
            className="glass summon fixed z-[1000] max-w-[300px] w-max border border-line rounded-[10px] px-3 py-2 text-[12px] leading-[1.45] font-normal text-ink whitespace-normal normal-case tracking-normal text-left pointer-events-none"
          >
            {children}
          </div>,
          document.body,
        )}
    </span>
  );
}
