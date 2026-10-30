"""Codebase analysis -> Proposal. The analyst *proposes*; whether the project moves is the rules'
call (`propose` requires verdict == fit, `declare_no_fit` requires verdict == no_fit)."""
from __future__ import annotations

import re
from pathlib import Path

from forge import repo as repo_mod
from forge.llm.base import LLM, OnEvent
from forge.models import FileRef, Opportunity, Proposal

CLM_PRIMER = """\
You are the analyst for CLM Forge. You decide whether, and where, a **CLM** belongs in a codebase.

## What a CLM is
A Contrastive Language Model: a frozen Qwen3-8B encoder plus two small trained projection heads. It
embeds a *state* (text describing the situation) and each *option* (text describing one answer), and
scores options by scaled cosine similarity; a softmax gives a calibrated probability per option.
One CLM head answers ONE typed question over a fixed option set:
- `choice`: pick one key from criteria {key: description}  (routing, triage, intent, category)
- `noul`:   yes/no ("is this X?")                          (verification, guardrails, policy checks)
- `score`:  an ordered level from a list low->high          (severity, priority, quality rating)
It also reranks arbitrary candidate lists (best-of-N selection, retrieval reranking).

## What CLM is good at (say fit)
- Fast (~tens of ms after embedding, no API call), cheap, local, calibrated scoring over an
  ENUMERABLE option set: routing/triage, intent or category classification, yes/no verification,
  guardrails/moderation, relevance/quality scoring, reranking, best-of-N selection.
- Replacing an LLM call that only returns a label, a brittle regex/keyword router, or an if/elif
  ladder over string keywords, where the label set is stable.
- It needs roughly ≤ 20 options, and a state text of ≤ ~2k tokens (longer text gets truncated from
  the head; keep the tail).

## What CLM is bad at (say no_fit, honestly)
- Generating text (replies, summaries, code, SQL), extraction of free-form spans, translation.
- Multi-step reasoning, math, tool use, planning, open-ended outputs.
- Label sets that are huge, open-ended, or change per request without being expressible as candidates.
- Tasks needing knowledge the state text does not contain (e.g. "is this account overdue?" when the
  ledger is not in the text).
If nothing qualifies, return verdict "no_fit" with concrete no_fit_reasons. A wrong "fit" wastes the
user's compute and trust; an honest no_fit is a good outcome.

## How to write the proposal
- Rank opportunities best first (at most 3). Each MUST have a non-empty `files` list citing REAL
  files (repo-relative paths) with start_line/end_line and a short snippet copied from the code —
  the call site to replace first. Never invent paths.
- `current_approach`: what the code does today (LLM call / regex / heuristic / manual / none).
- `task` is the TaskSchema the head will answer at that call site:
  - `name`: snake_case slug (becomes the package clm_<name>); `question_id`: short snake_case key.
  - `instructions`: the question appended after the state, phrased as a question.
  - `criteria`: choice -> {key: one-sentence description that distinguishes it from the others};
    score -> ordered list of level descriptions low->high; noul -> {"true": ..., "false": ...}.
    Keys must be the labels the code actually uses (e.g. the existing enum values).
  - `state_description`: exactly what text is available at the call site; `state_example`: a
    realistic example of it.
- `fit_score` 0..1; `expected_benefits` with concrete numbers where defensible (e.g. "p50 latency
  ~1-2 s LLM call -> ~50 ms local", "no per-call API cost"); `risks`.
- `data_strategy`: where training examples come from (mined fixtures/prompts/enums + synthetic).
- `target_examples`: 300-1200 typically (cap 3000); `success_metric`: "accuracy" for balanced labels,
  "macro_f1" if imbalanced; `success_threshold`: realistic absolute floor (often 0.80-0.92).
- `integration_sketch`: how the call site changes (call the head, fall back to the existing path
  when confidence < τ).
- `recommended_id` = id of the best opportunity (null when no_fit).
"""


def _task_prompt(description: str | None, has_repo: bool) -> str:
    parts = []
    if has_repo:
        parts.append("Analyse the repository in your working directory. Look for places where code asks a "
                     "model or a heuristic to pick from a fixed set of answers: LLM calls whose output is a "
                     "label/score/yes-no, prompts mentioning classify/categorise/route/label/rate/score, "
                     "regex routers, if/elif or match ladders over string keywords, sentiment/intent/"
                     "toxicity libraries. Read the relevant files to confirm what the label set is and what "
                     "text is available at the call site. Be efficient: grep first, read only what matters.")
    if description:
        parts.append(f"The user describes the task as:\n{description}")
    if not has_repo:
        parts.append("There is no repository; base the analysis on the description alone (files may be empty).")
    parts.append("Return a Proposal.")
    return "\n\n".join(parts)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "task"


def _ground(p: Proposal, repo_path: str | None) -> tuple[Proposal, list[str]]:
    """Make file refs real: drop paths that do not exist, fill/replace snippets from the file."""
    warnings: list[str] = []
    root = Path(repo_path) if repo_path else None
    seen_ids: set[str] = set()
    for i, o in enumerate(p.opportunities):
        if not o.id or o.id in seen_ids:
            o.id = f"opp{i + 1}"
        seen_ids.add(o.id)
        o.task.name = _slug(o.task.name)
        o.task.question_id = _slug(o.task.question_id)
        o.target_examples = max(150, min(3000, int(o.target_examples or 600)))
        o.success_threshold = max(0.5, min(0.99, float(o.success_threshold or 0.85)))
        if o.task.type == "choice" and isinstance(o.task.criteria, dict):
            o.task.criteria = {_slug(k): v for k, v in o.task.criteria.items()}
        if root is None:
            continue
        kept: list[FileRef] = []
        for f in o.files:
            rel = f.path.strip()
            if rel.startswith(str(root)):
                rel = str(Path(rel).relative_to(root))
            while rel.startswith("./"):
                rel = rel[2:]
            rel = rel.lstrip("/")
            fp = repo_mod.safe_path(root, rel)
            if fp is None or not fp.is_file():
                warnings.append(f"dropped non-existent or unsafe file ref {f.path!r} in {o.id}")
                continue
            lines = fp.read_text(errors="ignore").splitlines()
            s = f.start_line or 1
            e = f.end_line or min(len(lines), s + 15)
            s, e = max(1, min(s, len(lines))), max(1, min(max(e, s), len(lines), s + 40))
            snippet = "\n".join(lines[s - 1:e])
            kept.append(FileRef(path=rel, start_line=s, end_line=e, snippet=snippet))
        if not kept:
            kept = _infer_refs(o, root)
            if kept:
                warnings.append(f"{o.id} cited no valid files; attached scan matches: "
                                + ", ".join(f"{k.path}:{k.start_line}" for k in kept))
        o.files = kept
    ids = [o.id for o in p.opportunities]
    if p.verdict == "fit" and not p.opportunities:
        p.verdict = "no_fit"
        p.no_fit_reasons = p.no_fit_reasons or ["analysis returned no concrete opportunity"]
        warnings.append("verdict 'fit' with no opportunities was turned into no_fit")
    if p.verdict == "no_fit":
        p.recommended_id = None
    elif p.recommended_id not in ids:
        p.recommended_id = ids[0] if ids else None
    p.opportunities.sort(key=lambda o: (o.id != p.recommended_id, -o.fit_score))
    return p, warnings


def _infer_refs(o: Opportunity, root: Path) -> list[FileRef]:
    """Fallback grounding: candidate sites from the deterministic scan, preferring paths the
    opportunity text mentions."""
    try:
        sc = repo_mod.scan(root)
    except Exception:  # noqa: BLE001
        return []
    text = " ".join([o.title, o.rationale, o.current_approach, o.integration_sketch, o.data_strategy])
    mentioned = [c for c in sc.candidates if c.path in text or Path(c.path).name in text]
    pool = mentioned or [c for c in sc.candidates if "classify_prompt" in c.category or "keyword_ladder" in c.category]
    out, seen = [], set()
    for c in pool:
        if c.path in seen:
            continue
        seen.add(c.path)
        lines = repo_mod.read_text(root, c.path).splitlines()
        out.append(FileRef(path=c.path, start_line=c.start_line, end_line=c.end_line,
                           snippet="\n".join(lines[c.start_line - 1:c.end_line])))
        if len(out) >= 3:
            break
    return out


async def analyze(llm: LLM, repo_path: str | None, description: str | None, on_event: OnEvent,
                  previous: Proposal | None = None, feedback: list[str] | None = None) -> Proposal:
    task = _task_prompt(description, bool(repo_path))
    if previous is not None:
        fb = "\n".join(f"- {f}" for f in (feedback or []))
        task += (f"\n\nThis is a REVISION. Previous proposal (revision {previous.revision}):\n"
                 f"{previous.model_dump_json(indent=1)}\n\nUser feedback to address:\n{fb}\n"
                 "Keep what the user did not object to; change what they asked for. Re-check files only if needed.")
    p = await llm.explore(task, repo_path, Proposal, on_event, system=CLM_PRIMER)
    p, warnings = _ground(p, repo_path)
    p.revision = (previous.revision + 1) if previous is not None else 0
    for w in warnings:
        r = on_event({"type": "log", "stage": "analysis", "level": "warn", "message": w})
        if hasattr(r, "__await__"):
            await r
    return p


def opportunity(p: Proposal, oid: str) -> Opportunity | None:
    return next((o for o in p.opportunities if o.id == oid), None)
