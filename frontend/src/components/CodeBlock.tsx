import { cn } from "../lib/format";

export function CodeBlock({ code, startLine = 1, className, maxHeight = 280 }: { code: string; startLine?: number; className?: string; maxHeight?: number }) {
  const lines = code.replace(/\n$/, "").split("\n");
  const w = String(startLine + lines.length - 1).length;
  return (
    <div className={cn("bg-code border border-line-soft rounded-[6px] overflow-auto", className)} style={{ maxHeight }}>
      <pre className="font-mono text-[11.5px] leading-[1.6] py-2 min-w-max">
        {lines.map((l, i) => (
          <div key={i} className="flex">
            <span className="select-none text-muted/80 text-right pr-3 pl-2 sticky left-0 bg-code" style={{ minWidth: `${w + 2.5}ch` }}>
              {startLine + i}
            </span>
            <code className="pr-4 text-ink whitespace-pre">{l || " "}</code>
          </div>
        ))}
      </pre>
    </div>
  );
}
