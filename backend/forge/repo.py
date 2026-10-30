"""Locate / clone a repository, scan it, and find candidate CLM sites.

Local paths are read in place and never written to. https git URLs are shallow-cloned into
data/repos/<hash>. Everything here is deterministic (no LLM): it builds the digest that the
Ollama provider analyses and that seeds the Claude provider's exploration.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from forge import config

SKIP_DIRS = {".git", "node_modules", "vendor", "vendors", "dist", "build", ".venv", "venv", "env",
             "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "third_party",
             "site-packages", "target", ".next", ".nuxt", "coverage", ".tox", ".idea", ".vscode",
             "bower_components", ".gradle", "Pods", ".terraform", ".cache", "out"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".tgz",
            ".bz2", ".7z", ".whl", ".so", ".dylib", ".dll", ".exe", ".bin", ".pt", ".pth", ".onnx",
            ".safetensors", ".npy", ".npz", ".pkl", ".parquet", ".lock", ".mp3", ".mp4", ".mov",
            ".woff", ".woff2", ".ttf", ".otf", ".eot", ".class", ".jar", ".pyc", ".map", ".min.js",
            ".svg", ".sqlite", ".db"}
MAX_FILE_BYTES = 400_000
MAX_FILES = 5000

LANG_BY_EXT = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".rb": "Ruby",
    ".php": "PHP", ".cs": "C#", ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++", ".swift": "Swift",
    ".scala": "Scala", ".ex": "Elixir", ".exs": "Elixir", ".md": "Markdown", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".sql": "SQL", ".sh": "Shell", ".html": "HTML",
    ".css": "CSS", ".jsonl": "JSONL", ".csv": "CSV", ".txt": "Text", ".ipynb": "Notebook",
}
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java", ".kt", ".rb", ".php",
            ".cs", ".cpp", ".cc", ".c", ".swift", ".scala", ".ex", ".exs"}

# (category, regex) — matched per line on code files.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("llm_call", re.compile(
        r"(chat\.completions\.create|ChatCompletion\.create|responses\.create|messages\.create|"
        r"\bcompletion\(|acompletion\(|litellm\.|ollama\.(chat|generate)|\.invoke\(|ChatOpenAI|ChatAnthropic|"
        r"ChatOllama|generateContent|anthropic\.Anthropic|OpenAI\(|AsyncOpenAI\(|from openai|import openai|"
        r"from anthropic|import anthropic|from litellm|from langchain|import ollama|with_structured_output)")),
    ("classify_prompt", re.compile(
        r"(?i)[\"'`].*\b(classif|categor|label|route|routing|is this|yes or no|yes/no|rate (this|the)|"
        r"score (this|the|from)|on a scale|which (department|team|category|intent)|intent|sentiment|"
        r"toxic|spam|relevan|triage)")),
    ("regex_router", re.compile(r"(re\.(search|match|fullmatch|compile)\(|new RegExp\(|\.match\(/|/[^/\n]{3,}/[gimsuy]*\.test\()")),
    ("keyword_ladder", re.compile(
        r"^\s*(if|elif|else if|case)\b.*([\"'][^\"']+[\"']\s+in\s+[\w.]+|\bin\b\s+\w*(text|msg|message|body|subject|query|content|lower\(\))|"
        r"\.lower\(\)|\.includes\(|\.contains\(|\.startswith\(|==\s*[\"'][a-z_ ]+[\"'])")),
    ("ml_lib", re.compile(
        r"(?i)(vaderSentiment|textblob|transformers\.pipeline|pipeline\([\"'](sentiment|text-classification|"
        r"zero-shot)|detoxify|perspective api|fasttext|sklearn\.(naive_bayes|linear_model)|"
        r"zero[-_]shot|SentimentIntensityAnalyzer|spacy\.load|setfit|toxicity|moderation)")),
]


def is_url(s: str) -> bool:
    return bool(re.match(r"^(https://|git@|ssh://)", s.strip()))


def resolve(repo: str) -> Path:
    """Return a local directory for `repo` (https git URL or absolute local path)."""
    repo = repo.strip()
    if is_url(repo):
        if not repo.startswith("https://"):
            raise ValueError("only https:// git URLs are supported")
        h = hashlib.sha1(repo.encode()).hexdigest()[:12]
        dest = config.REPOS_DIR / h
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest / ".git").exists():
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            r = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", repo, str(dest)],
                               capture_output=True, text=True, timeout=300, env=env)
            if r.returncode != 0:
                raise RuntimeError(f"git clone failed: {r.stderr.strip()[-500:]}")
        return dest
    p = Path(repo).expanduser()
    if not p.is_absolute():
        raise ValueError("repo must be an https git URL or an absolute local path")
    if not p.is_dir():
        raise FileNotFoundError(f"no such directory: {p}")
    return p.resolve()


def _gitignore_patterns(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.exists():
        return []
    pats = []
    for line in gi.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("!"):
            pats.append(line.rstrip("/").lstrip("/"))
    return pats


def _ignored(rel: str, pats: list[str]) -> bool:
    parts = rel.split("/")
    for pat in pats:
        if "/" in pat:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, pat + "/*"):
                return True
        elif any(fnmatch.fnmatch(part, pat) for part in parts):
            return True
    return False


def _git_files(root: Path) -> list[str] | None:
    if not (root / ".git").exists():
        return None
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
                           capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    return [ln for ln in r.stdout.splitlines() if ln]


def list_files(root: Path) -> list[str]:
    """Relative paths of readable text files, respecting .gitignore and skipping vendored/binary/huge."""
    files = _git_files(root)
    if files is None:
        pats = _gitignore_patterns(root)
        files = []
        for dp, dns, fns in os.walk(root):
            rel_dir = os.path.relpath(dp, root)
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")
                      and not _ignored(os.path.normpath(os.path.join(rel_dir, d)), pats)]
            for fn in fns:
                rel = os.path.normpath(os.path.join(rel_dir, fn))
                if not _ignored(rel, pats):
                    files.append(rel)
            if len(files) > MAX_FILES * 2:
                break
    out = []
    for rel in sorted(files):
        parts = rel.split("/")
        if any(p in SKIP_DIRS for p in parts[:-1]):
            continue
        low = rel.lower()
        if any(low.endswith(e) for e in SKIP_EXT):
            continue
        fp = safe_path(root, rel)
        if fp is None:
            continue
        try:
            st = fp.stat()
        except OSError:
            continue
        if not fp.is_file() or st.st_size > MAX_FILE_BYTES or st.st_size == 0:
            continue
        try:
            with open(fp, "rb") as f:
                if b"\x00" in f.read(2048):
                    continue
        except OSError:
            continue
        out.append(rel)
        if len(out) >= MAX_FILES:
            break
    return out


def safe_path(root: Path, rel: str) -> Path | None:
    """``root / rel`` if it is safe to read, else None.

    Repos may be cloned from untrusted URLs and file paths may come from an LLM, so reject absolute
    paths, ``..``, any symlink on the way from ``root`` to the file, and anything resolving outside
    ``root``. Otherwise a planted ``notes.txt -> ~/.aws/credentials`` would be read and sent on.
    """
    rp = Path(rel)
    if not rel or rp.is_absolute() or ".." in rp.parts:
        return None
    fp = root / rp
    try:
        hops = [root / q for q in [rp, *rp.parents] if str(q) != "."]
        if any(h.is_symlink() for h in hops) or root.resolve() not in fp.resolve().parents:
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return fp


def read_text(root: Path, rel: str, max_chars: int = MAX_FILE_BYTES) -> str:
    """Whole file by default: every listed file is <= MAX_FILE_BYTES, so grep, read_file and grounding
    see the same content list_files offers (a smaller cap silently hid the tail of large files)."""
    fp = safe_path(root, rel)
    if fp is None:
        return ""
    try:
        return fp.read_text(errors="ignore")[:max_chars]
    except OSError:
        return ""


@dataclass
class Candidate:
    path: str
    line: int
    category: str
    snippet: str
    start_line: int
    end_line: int


@dataclass
class RepoScan:
    root: str
    files: list[str]
    languages: dict[str, int]
    readme: str
    candidates: list[Candidate] = field(default_factory=list)

    def tree(self, max_lines: int = 300) -> str:
        lines = self.files[:max_lines]
        more = len(self.files) - len(lines)
        return "\n".join(lines) + (f"\n... (+{more} more files)" if more > 0 else "")

    def digest(self, max_chars: int = 24_000) -> str:
        """Compact text an LLM without tools can analyse: tree, languages, README, candidate sites."""
        langs = ", ".join(f"{k} ({v})" for k, v in sorted(self.languages.items(), key=lambda x: -x[1])[:10])
        parts = [f"# Repository digest\nFiles: {len(self.files)}. Languages: {langs}\n",
                 "## File tree\n" + self.tree(150), "## README (excerpt)\n" + (self.readme or "(none)")[:3000],
                 "## Candidate sites (pattern matches)\n"]
        body = "\n".join(parts)
        for c in self.candidates:
            block = f"\n### {c.path}:{c.start_line}-{c.end_line} [{c.category}]\n```\n{c.snippet}\n```\n"
            if len(body) + len(block) > max_chars:
                body += "\n... (more candidates truncated)"
                break
            body += block
        return body


def _window(lines: list[str], i: int, before: int = 4, after: int = 8) -> tuple[int, int, str]:
    a, b = max(0, i - before), min(len(lines), i + after + 1)
    return a + 1, b, "\n".join(f"{n + 1:>4}  {lines[n][:200]}" for n in range(a, b))


def find_candidates(root: Path, files: list[str], max_candidates: int = 60) -> list[Candidate]:
    """Grep for LLM calls, classify-ish prompts, regex routers, keyword ladders and sentiment/intent libs."""
    hits: list[tuple[int, Candidate]] = []
    for rel in files:
        ext = os.path.splitext(rel)[1].lower()
        if ext not in CODE_EXT:
            continue
        lines = read_text(root, rel).splitlines()
        cats_by_line: dict[int, set[str]] = {}
        ladder_runs: list[int] = []
        for i, line in enumerate(lines):
            for cat, rx in PATTERNS:
                if rx.search(line):
                    cats_by_line.setdefault(i, set()).add(cat)
            if "keyword_ladder" in cats_by_line.get(i, set()):
                ladder_runs.append(i)
        # keep ladders only when ≥3 branches within 40 lines (a real if/elif chain over strings)
        dense_ladder = {i for i in ladder_runs if sum(1 for j in ladder_runs if abs(j - i) <= 40) >= 3}
        file_score = 0
        merged: list[tuple[int, set[str]]] = []
        for i in sorted(cats_by_line):
            cats = {c for c in cats_by_line[i] if c != "keyword_ladder" or i in dense_ladder}
            if not cats:
                continue
            if merged and i - merged[-1][0] <= 6:
                merged[-1][1].update(cats)
                continue
            merged.append((i, set(cats)))
        weight = {"llm_call": 3, "classify_prompt": 3, "ml_lib": 3, "keyword_ladder": 2, "regex_router": 1}
        for i, cats in merged:
            s, e, snip = _window(lines, i)
            score = sum(weight[c] for c in cats)
            file_score += score
            hits.append((score, Candidate(rel, i + 1, ",".join(sorted(cats)), snip, s, e)))
    # prefer high-signal sites; cap per file so one file can't drown the digest
    hits.sort(key=lambda x: (-x[0], x[1].path, x[1].line))
    out, per_file = [], Counter()
    for _, c in hits:
        if per_file[c.path] >= 6:
            continue
        per_file[c.path] += 1
        out.append(c)
        if len(out) >= max_candidates:
            break
    return out


def scan(root: Path) -> RepoScan:
    files = list_files(root)
    langs = Counter(LANG_BY_EXT.get(os.path.splitext(f)[1].lower(), "Other") for f in files)
    langs.pop("Other", None)
    readme = ""
    for f in files:
        if os.path.basename(f).lower().startswith("readme") and "/" not in f:
            readme = read_text(root, f, 6000)
            break
    return RepoScan(str(root), files, dict(langs), readme, find_candidates(root, files))


def relevant_files(scan_: RepoScan, extra: list[str] | None = None, limit: int = 12) -> list[str]:
    """Files worth mining for examples: those named by the opportunity, then fixtures/tests/data/prompts."""
    chosen: list[str] = []
    for p in extra or []:
        if p in scan_.files and p not in chosen:
            chosen.append(p)
    rx = re.compile(r"(?i)(test|fixture|sample|example|seed|prompt|label|categor|enum|data)")
    data_ext = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".md"}
    for f in scan_.files:
        if len(chosen) >= limit:
            break
        if f in chosen:
            continue
        ext = os.path.splitext(f)[1].lower()
        if rx.search(f) and (ext in CODE_EXT or ext in data_ext) and not f.lower().startswith("readme"):
            chosen.append(f)
    for c in scan_.candidates:
        if len(chosen) >= limit:
            break
        if c.path not in chosen:
            chosen.append(c.path)
    return chosen[:limit]
