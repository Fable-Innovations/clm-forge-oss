import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronRight, ClipboardCopy, Download, File, FileCode2, FileText, Folder, Package } from "lucide-react";
import { api } from "../api/client";
import { qk } from "../api/queries";
import type { PackageFile, PackageManifest } from "../api/types";
import { Button, Chip, Empty, Panel, Segmented, Spinner } from "../components/ui";
import { Markdown } from "../components/Markdown";
import { CodeBlock } from "../components/CodeBlock";
import { bytes, cn, timeAgo } from "../lib/format";
import { useWs } from "./context";

type Node = { name: string; path: string; file?: PackageFile; children: Map<string, Node>; size: number };

function buildTree(files: PackageFile[]): Node {
  const root: Node = { name: "", path: "", children: new Map(), size: 0 };
  for (const f of files) {
    const parts = f.path.split("/");
    let n = root;
    n.size += f.bytes;
    parts.forEach((p, i) => {
      const path = parts.slice(0, i + 1).join("/");
      let c = n.children.get(p);
      if (!c) {
        c = { name: p, path, children: new Map(), size: 0 };
        n.children.set(p, c);
      }
      c.size += f.bytes;
      if (i === parts.length - 1) c.file = f;
      n = c;
    });
  }
  return root;
}

const TEXT_EXT = /\.(md|py|toml|json|jsonl|txt|cfg|yaml|yml)$/i;

function TreeNode({ n, depth, sel, onSel }: { n: Node; depth: number; sel: string | null; onSel: (p: string) => void }) {
  const [open, setOpen] = useState(depth < 1);
  const kids = [...n.children.values()].sort((a, b) => Number(!!a.file) - Number(!!b.file) || a.name.localeCompare(b.name));
  if (n.file) {
    const text = TEXT_EXT.test(n.name);
    const Icon = /\.md$/.test(n.name) ? FileText : /\.(py|toml|json|jsonl)$/.test(n.name) ? FileCode2 : File;
    return (
      <li>
        <button
          disabled={!text}
          onClick={() => onSel(n.path)}
          title={n.file.description}
          className={cn(
            "w-full grid grid-cols-[1fr_auto] gap-2 items-center text-left py-1 pr-2 rounded-[4px] text-[12.5px]",
            text ? "cursor-pointer hover:bg-sunken" : "cursor-default",
            sel === n.path && "bg-accent-soft",
          )}
          style={{ paddingLeft: 8 + depth * 14 }}
        >
          <span className="min-w-0">
            <span className="flex items-center gap-1.5 min-w-0">
              <Icon size={13} className="text-muted shrink-0" />
              <span className="font-mono text-[12px] truncate">{n.name}</span>
            </span>
            <span className="block text-[11px] text-muted truncate pl-[19px]">{n.file.description}</span>
          </span>
          <span className="text-[11.5px] text-muted tnum">{bytes(n.file.bytes)}</span>
        </button>
      </li>
    );
  }
  return (
    <li>
      <button onClick={() => setOpen((o) => !o)} className="w-full grid grid-cols-[1fr_auto] gap-2 items-center text-left py-1 pr-2 rounded-[4px] text-[12.5px] hover:bg-sunken cursor-pointer" style={{ paddingLeft: 8 + depth * 14 }} aria-expanded={open}>
        <span className="flex items-center gap-1.5 min-w-0">
          <ChevronRight size={12} className={cn("text-muted transition-transform shrink-0", open && "rotate-90")} />
          <Folder size={13} className="text-muted shrink-0" />
          <span className="font-mono text-[12px] truncate">{n.name}/</span>
        </span>
        <span className="text-[11.5px] text-muted tnum">{bytes(n.size)}</span>
      </button>
      {open && (
        <ul>
          {kids.map((k) => (
            <TreeNode key={k.path} n={k} depth={depth + 1} sel={sel} onSel={onSel} />
          ))}
        </ul>
      )}
    </li>
  );
}

function agentPrompt(m: PackageManifest, integration: string) {
  return `You are integrating a trained CLM classifier into this codebase.

Package: ${m.name} ${m.version}
Package directory: ${m.dir}
Zip: ${m.zip}

Follow INTEGRATION.md below exactly. Keep the existing code path as the low-confidence fallback, put the change behind the feature flag it names, and add the tests it lists (golden examples are in examples/golden.jsonl inside the package).

---

${integration}`;
}

export function PackageTab() {
  const { pid, store } = useWs();
  const q = useQuery({ queryKey: qk.pkg(pid), queryFn: () => api.package(pid), retry: false, enabled: !!store.manifest || store.state === "delivered" });
  const m = store.manifest ?? q.data ?? null;
  const [view, setView] = useState<string>("integration");
  const [sel, setSel] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [dl, setDl] = useState<"idle" | "busy" | "error">("idle");

  const integrationPath = m?.integration_md ?? "INTEGRATION.md";
  const evalPath = m?.files.find((f) => /EVAL_REPORT\.md$/.test(f.path))?.path ?? "EVAL_REPORT.md";
  const shownPath = view === "integration" ? integrationPath : view === "eval" ? evalPath : sel;

  const file = useQuery({ queryKey: qk.pkgFile(pid, shownPath ?? ""), queryFn: () => api.packageFile(pid, shownPath!), enabled: !!m && !!shownPath, retry: false, staleTime: Infinity });
  const integ = useQuery({ queryKey: qk.pkgFile(pid, integrationPath), queryFn: () => api.packageFile(pid, integrationPath), enabled: !!m, retry: false, staleTime: Infinity });
  const tree = useMemo(() => (m ? buildTree(m.files) : null), [m]);

  if (!m) {
    return (
      <Empty icon={store.state === "packaging" ? <Spinner size={22} /> : <Package size={26} />} title={store.state === "packaging" ? "Building the package" : "Nothing packaged yet"}>
        {store.state === "packaging"
          ? "Writing the runtime package, building the wheel and drafting INTEGRATION.md."
          : "Once the evaluation gate passes, the agent ships: a pip-installable package with the trained head, an evaluation report and an integration plan a coding agent can execute."}
      </Empty>
    );
  }

  const download = async () => {
    setDl("busy");
    try {
      const res = await fetch(api.packageDownloadUrl(pid));
      if (!res.ok) throw new Error(String(res.status));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${m.name}-${m.version}.zip`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      setDl("idle");
    } catch {
      setDl("error");
    }
  };

  const copy = async () => {
    const text = agentPrompt(m, integ.data ?? "");
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const total = m.files.reduce((a, f) => a + f.bytes, 0);
  const isMd = shownPath?.endsWith(".md");

  return (
    <div className="p-4 space-y-4">
      <div className="bg-panel border border-line rounded-[10px] px-5 py-4 flex flex-wrap items-center gap-4">
        <div className="w-10 h-10 rounded-[8px] grid place-items-center bg-good-soft text-good">
          <Package size={20} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="text-[17px] font-semibold font-mono">{m.name}</h2>
            <Chip tone="good">v{m.version}</Chip>
          </div>
          <div className="text-[12px] text-muted mt-0.5 truncate">
            {m.files.length} files, {bytes(total)} · built {timeAgo(m.created_at)} · <span className="font-mono">{m.dir}</span>
          </div>
        </div>
        <Button icon={<Download size={14} />} onClick={download} disabled={dl === "busy"}>
          {dl === "busy" ? "Preparing…" : dl === "error" ? "Download failed, retry" : "Download zip"}
        </Button>
        <Button variant="primary" icon={copied ? <Check size={14} /> : <ClipboardCopy size={14} />} onClick={copy} disabled={!integ.data}>
          {copied ? "Copied" : "Copy for coding agent"}
        </Button>
      </div>

      <div className="grid gap-4 grid-cols-1 xl:grid-cols-[340px_minmax(0,1fr)] items-start">
        <Panel title="Files" sub="Click a text file to preview it." bodyClass="px-2 pb-3 pt-1">
          {tree && (
            <ul>
              {[...tree.children.values()]
                .sort((a, b) => Number(!!a.file) - Number(!!b.file) || a.name.localeCompare(b.name))
                .map((n) => (
                  <TreeNode
                    key={n.path}
                    n={n}
                    depth={0}
                    sel={view === "file" ? sel : view === "integration" ? integrationPath : evalPath}
                    onSel={(p) => {
                      if (p === integrationPath) setView("integration");
                      else if (p === evalPath) setView("eval");
                      else {
                        setSel(p);
                        setView("file");
                      }
                    }}
                  />
                ))}
            </ul>
          )}
        </Panel>
        <Panel
          title={shownPath ?? ""}
          aside={
            <Segmented
              value={view}
              onChange={setView}
              options={[
                { value: "integration", label: "INTEGRATION.md" },
                { value: "eval", label: "EVAL_REPORT.md" },
                ...(sel ? [{ value: "file", label: sel.split("/").pop()! }] : []),
              ]}
            />
          }
        >
          {file.isLoading ? (
            <div className="py-10 text-center text-muted">
              <Spinner />
            </div>
          ) : file.isError ? (
            <p className="text-[13px] text-muted py-6">Couldn't load {shownPath}: {(file.error as Error).message}</p>
          ) : isMd ? (
            <div className="max-w-[78ch]">
              <Markdown>{file.data ?? ""}</Markdown>
            </div>
          ) : (
            <CodeBlock code={file.data ?? ""} maxHeight={640} />
          )}
        </Panel>
      </div>
    </div>
  );
}
