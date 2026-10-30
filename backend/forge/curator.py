"""Training-data curation: import -> mine -> generate (+hard negatives) -> independent relabel -> dedup ->
balance -> stratified split -> dataset.jsonl. See SPEC "Curation pipeline".

Import (``importer.py``) turns tables the user exported into the repo (``clm_data/*.csv`` etc.) into
real examples; generation only fills what real data does not cover, and real rows fill test first.

LLM batches run concurrently under a semaphore. Generation fans out to one worker agent per label
(batches from all workers share a wider generation semaphore); before labelling, an orchestrator
agent reviews a sample of each worker's output and sends a label back for one regeneration round,
with its feedback, when too much of the sample fails.

On a recurate the previous dataset is kept (test split frozen so attempts stay comparable) and new,
guidance-directed rows go to train/val only. Tables exported since the last pass are imported then
too, into train/val.

Exported rows pinned to ``split=test`` (see ``importer``) form a frozen test split from the first pass:
all of them are kept, none are sampled, balanced or re-split, train rows duplicating them are dropped in
dedup, and the train/val split is drawn only from the remaining rows.

Real rows without a label (imported without a label column value, mined without a stated label) follow the
project's ``label_policy`` (SPEC "Labelling real rows"): ``agent`` (the default, also when no policy was
set) and ``user`` queue them as ``needs_label`` without a teacher call — the ``labelling`` stage's agent
disposes them (``forge.labelling.agent_label``) or, under ``user``, the human review gate does; ``teacher``
labels them with the teacher LLM (and mirrors them into the queue so the user can review and override),
``mixed`` asks the teacher for a suggestion and still queues them as ``needs_label``.
Rows still ``needs_label`` never reach ``dataset.jsonl``; ``forge.labelling`` upserts them once labelled.

Real-shaped generation (SPEC "Real-shaped synthetic data"): when real labelled rows exist, every generation
prompt carries 4 scrubbed real exemplars (``forge.shaping``) with their key template, length band and
register, and asks for the same structure with new people, topics and wording. Those rows carry
``provenance=real_shaped_synthetic`` and are train-only. After the split every generated train row is
leak-checked against val/test (``forbidden``), and real-shaped rows also against the whole real pool
(shared 8-gram, 4-gram Jaccard, real identifiers); drops are counted per guard in ``stats["shaped_drops"]``.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from forge import shaping
from forge.labelling import DEFAULT_POLICY, POLICIES, LabelQueue, from_example
from forge.llm.base import LLM, LLMError
from forge.models import REAL_SHAPED, REAL_SOURCES, Example, Opportunity, RealRow, TaskSchema

PERSONAS = ["a frustrated long-time customer", "a terse power user", "a non-native English speaker",
            "a polite first-time user", "an enterprise admin", "a developer integrating the API",
            "someone writing from their phone", "a confused elderly user", "an angry user threatening to churn",
            "a small-business owner", "a support agent forwarding a note", "a user pasting an error message"]
STYLES = ["one short sentence", "two or three sentences", "a detailed paragraph with specifics",
          "informal with typos", "formal and structured", "a bulleted list of facts",
          "mixed concerns where one clearly dominates", "indirect phrasing that never names the category"]


# ---- structured shapes the LLM fills ------------------------------------------------------------
class MinedItem(BaseModel):
    state: str = Field(description="a realistic state text as it would appear at the call site")
    label: str | None = Field(default=None, description="label key only if the source states it explicitly, else null")
    evidence: str = Field(default="", description="where in the file this came from")


class MinedBatch(BaseModel):
    items: list[MinedItem]


class Voices(BaseModel):
    personas: list[str] = Field(description="8-12 distinct sources/speakers who realistically produce this state text")
    styles: list[str] = Field(description="6-10 distinct forms/lengths/registers the state text realistically takes")


class GenBatch(BaseModel):
    states: list[str] = Field(description="distinct, realistic state texts")


class QAItem(BaseModel):
    i: int = Field(description="index of the state in the list")
    ok: bool = Field(description="true if this text is usable training data for the stated label")
    issue: str = Field(default="", description="short reason when ok is false")


class QAReview(BaseModel):
    items: list[QAItem]
    feedback: str = Field(description="concrete instructions for the generator to fix recurring problems; "
                                      "empty if the sample is good")


class LabelItem(BaseModel):
    i: int = Field(description="index of the state in the list")
    label: str = Field(description="exactly one of the allowed label keys")
    confidence: float = Field(ge=0, le=1)
    reason: str


class LabelBatch(BaseModel):
    labels: list[LabelItem]


# ---- helpers -------------------------------------------------------------------------------------
# real data wins over generated data in dedup, balance and the test split
SOURCE_PRIORITY = {"user": 0, "imported": 1, "mined": 2, "hard_negative": 3, "synthetic": 4}


def label_menu(task: TaskSchema) -> str:
    if task.type == "choice":
        return "\n".join(f"- {k}: {v}" for k, v in (task.criteria or {}).items())
    if task.type == "score":
        return "\n".join(f"- {i}: {v}" for i, v in enumerate(task.criteria or []))
    crit = task.criteria if isinstance(task.criteria, dict) else {}
    return f"- true: {crit.get('true', 'yes')}\n- false: {crit.get('false', 'no')}"


def label_desc(task: TaskSchema, label: str) -> str:
    if task.type == "choice":
        return f"{label} ({(task.criteria or {}).get(label, '')})"
    if task.type == "score":
        lv = task.criteria or []
        return f"level {label} ({lv[int(label)] if label.isdigit() and int(label) < len(lv) else ''})"
    crit = task.criteria if isinstance(task.criteria, dict) else {}
    return f"{label} ({crit.get(label, 'yes' if label == 'true' else 'no')})"


def norm_label(task: TaskSchema, raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().strip("\"'`").lower()
    labels = task.labels()
    if s in labels:
        return s
    if task.type == "noul":
        return {"yes": "true", "no": "false", "1": "true", "0": "false"}.get(s)
    if task.type == "choice":
        for k in labels:
            if s.replace(" ", "_") == k.lower() or s == k.lower():
                return k
    if task.type == "score":
        m = re.match(r"(level\s*)?(\d+)", s)
        if m and m.group(2) in labels:
            return m.group(2)
    return None


def _norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def _eid(prefix: str, state: str) -> str:
    return prefix + hashlib.sha1(state.encode()).hexdigest()[:12]


async def _call(fn, *a, **kw):
    r = fn(*a, **kw)
    if inspect.isawaitable(r):
        r = await r
    return r


def _task_brief(task: TaskSchema) -> str:
    return (f"Task: {task.instructions}\nType: {task.type}\nLabels:\n{label_menu(task)}\n\n"
            f"The state text is: {task.state_description}\nExample state:\n{task.state_example}")


async def label_states(llm: LLM, task: TaskSchema, states: list[str], batch: int = 15,
                       sem: asyncio.Semaphore | None = None) -> list[tuple[str | None, float, str]]:
    """Teacher labels, independent of how the states were produced. -> [(label|None, conf, reason)]."""
    sem = sem or asyncio.Semaphore(4)
    out: list[tuple[str | None, float, str]] = [(None, 0.0, "")] * len(states)
    system = ("You are a careful annotator. Label each state with exactly one allowed label key, judging "
              "only from the state text. Do not guess from style; follow the label descriptions.")

    async def one(lo: int):
        chunk = states[lo:lo + batch]
        listing = "\n\n".join(f"[{i}]\n{s[:3000]}" for i, s in enumerate(chunk))
        prompt = (f"{_task_brief(task)}\n\nLabel every state below. Return one entry per index "
                  f"(0..{len(chunk) - 1}); `label` must be one of: {task.labels()}.\n\n{listing}")
        async with sem:
            try:
                res = await llm.structured(prompt, LabelBatch, system=system)
            except LLMError:
                return
        for it in res.labels:
            if 0 <= it.i < len(chunk):
                out[lo + it.i] = (norm_label(task, it.label), float(it.confidence), it.reason)

    await asyncio.gather(*(one(lo) for lo in range(0, len(states), batch)))
    return out


# ---- the pipeline --------------------------------------------------------------------------------
class Curator:
    def __init__(self, llm: LLM, task: TaskSchema, emit: Callable[[dict], Any], *, target: int,
                 repo_path: str | None = None, opportunity: Opportunity | None = None,
                 guidance: str | None = None, previous: list[Example] | None = None, concurrency: int = 4,
                 seed: int = 1234, should_stop: Callable[[], bool] = lambda: False,
                 dedup_threshold: float = 0.975, use_embeddings: bool = True, encoder_spec: str | None = None,
                 gen_concurrency: int | None = None, qa_min_pass: float = 0.75, qa_sample: int = 24,
                 label_policy: str | None = None, queue: LabelQueue | None = None):
        if label_policy is not None and label_policy not in POLICIES:
            raise ValueError(f"label_policy must be one of {'|'.join(POLICIES)}, got {label_policy!r}")
        self.llm, self.task, self.emit = llm, task, emit
        self.label_policy = label_policy or DEFAULT_POLICY
        self.queue = queue
        self.queued: list[RealRow] = []   # real rows for the labelling queue (pending, or teacher-labelled mirrors)
        self.target = target
        self.repo_path, self.opp, self.guidance = repo_path, opportunity, (guidance or "").strip()
        self.previous = previous or []
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.gen_sem = asyncio.Semaphore(max(1, gen_concurrency or concurrency * 2))
        self.qa_min_pass, self.qa_sample = qa_min_pass, qa_sample
        self.rng = random.Random(seed)
        self.should_stop = should_stop
        self.dedup_threshold = dedup_threshold
        self.use_embeddings = use_embeddings
        if use_embeddings and not encoder_spec:
            raise ValueError("embedding dedup needs the project's encoder spec (encoder_spec)")
        self.encoder_spec = encoder_spec
        self.labels = task.labels()
        self.stats: dict[str, Any] = {"disagreements": 0, "invalid": 0, "exact_dups": 0, "near_dups": 0,
                                      "imported": 0, "imported_labelled": 0, "mined": 0, "generated": 0,
                                      "capped": 0, "qa_rejected": 0, "qa": {}, "needs_label": 0,
                                      "shaped": 0, "shaped_drops": {}}
        # generated state -> {exemplar_hashes, batch}: rows shaped on real exemplars (train-only)
        self.shaped_meta: dict[str, dict] = {}
        # (the exemplar list, the names its real rows show): computed once per exemplar list (shaping.pool_names)
        self._pool_names: tuple[list, set[str]] | None = None
        self.imported_files: set[str] = set()
        self.fixed_test: list[Example] = []   # rows the export pins to the test split (first pass only)
        self.personas, self.styles = list(PERSONAS), list(STYLES)

    # progress ------------------------------------------------------------------------------------
    async def _progress(self, phase: str, done: int, total: int, rows: list[Example] | None = None, **extra):
        counts = {"by_label": {}, "by_source": {}, "by_split": {}}
        for r in rows or []:
            counts["by_label"][r.label] = counts["by_label"].get(r.label, 0) + 1
            counts["by_source"][r.source] = counts["by_source"].get(r.source, 0) + 1
            counts["by_split"][r.split] = counts["by_split"].get(r.split, 0) + 1
        counts["needs_label"] = self.stats.get("needs_label", 0)
        await _call(self.emit, {"type": "curation", "phase": phase, "done": done, "total": total,
                                "counts": counts, **extra})

    async def _log(self, msg: str, level: str = "info"):
        await _call(self.emit, {"type": "log", "stage": "curating", "level": level, "message": msg})

    def _check(self):
        if self.should_stop():
            raise asyncio.CancelledError("curation stopped")

    # real rows without a label: the label policy -------------------------------------------------------
    async def label_real(self, items: list[tuple[str, Example]], what: str) -> list[Example]:
        """Label unlabelled real rows per ``label_policy``. ``items`` = [(state, Example template with
        label="")]. -> Examples that may go into the dataset now (teacher policy only); every other row is
        queued as ``needs_label`` in ``self.queued``."""
        if not items:
            return []
        policy = self.label_policy
        if policy in ("agent", "user"):
            await self._log(f"{len(items)} unlabelled {what} go to "
                            + ("the labelling agent" if policy == "agent" else "your label review"))
            for _s, ex in items:
                self.queued.append(from_example(ex, label_state="needs_label", head=self.task.name))
            return []
        await self._log(f"teacher {'labelling' if policy == 'teacher' else 'suggesting labels for'} "
                        f"{len(items)} unlabelled {what}")
        results = await label_states(self.llm, self.task, [s for s, _ in items], sem=self.sem)
        out: list[Example] = []
        for (_s, ex), (got, conf, why) in zip(items, results):
            if policy == "teacher":
                if got is None:
                    self.stats["invalid"] += 1
                    continue
                out.append(ex.model_copy(update={"label": got, "rationale": why, "label_source": "teacher",
                                                 "meta": {**ex.meta, "label_origin": "teacher",
                                                          "teacher_confidence": conf}}))
            else:   # mixed: a suggestion; the row still needs a label
                row = from_example(ex, label_state="needs_label", head=self.task.name)
                self.queued.append(row.model_copy(update={"suggested_label": got, "teacher_conf": conf}))
        return out

    # 1. mine ---------------------------------------------------------------------------------------
    async def import_data(self, per_label: int) -> list[Example]:
        """Real rows from exported tables. Labelled rows keep the export's label; unlabelled rows get the
        teacher's. Sampled to ~1.3x the per-label target so dedup and balance still have room.

        Rows pinned to the test split are not returned: all of them go to ``self.fixed_test``."""
        if not self.repo_path:
            return []
        from forge import importer

        async def tool(rel: str, summary: str):
            await _call(self.emit, {"type": "tool", "name": "import", "input": rel, "output": summary})

        root = Path(self.repo_path)
        found = importer.discover(root)
        if not found:
            return []
        await self._progress("importing", 0, len(found))
        res = await importer.import_tables(self.llm, self.task, root, self.rng, self._log, tool)
        self.imported_files = set(res.files)
        self._check()
        want = math.ceil(per_label * 1.3)
        by_label: dict[str, list] = defaultdict(list)
        unlabelled = []
        await self._import_fixed_test([r for r in res.rows if r.split == "test"])
        for r in (r for r in res.rows if r.split != "test"):
            (by_label[r.label] if r.label else unlabelled).append(r)
        picked = []
        for lab, rs in by_label.items():
            self.rng.shuffle(rs)
            picked.extend(rs[:want])
        rows: list[Example] = [
            Example(id=_eid("i", r.state), state=r.state, label=r.label, source="imported", split="train",
                    rationale=f"label from export ({r.raw_label})", label_source="export",
                    meta={"file": r.file, "row": r.row, "label_origin": "export", "raw_label": r.raw_label})
            for r in picked]
        room = max(0, want * len(self.labels) - len(rows))
        if unlabelled and room:
            self.rng.shuffle(unlabelled)
            todo = unlabelled[:room]
            rows += await self.label_real(
                [(r.state, Example(id=_eid("i", r.state), state=r.state, label="", source="imported", split="train",
                                   meta={"file": r.file, "row": r.row})) for r in todo], "imported rows")
        self.stats["imported"] = len(rows) + len(self.fixed_test)
        self.stats["imported_labelled"] = sum(1 for r in rows + self.fixed_test if r.meta.get("label_origin") == "export")
        await self._progress("importing", len(found), len(found), rows)
        await self._log(f"imported {len(rows)} real rows from {sum(1 for v in res.files.values() if 'rows' in v)} "
                        f"of {len(found)} tables: {dict(Counter(r.label for r in rows))}")
        return rows

    async def _import_fixed_test(self, pinned: list) -> None:
        """Every row the export pins to the test split, split=test. Unlabelled ones follow the label policy;
        queued ones keep split=test in the queue, so they join the frozen test split once labelled."""
        out: list[Example] = []
        todo = [r for r in pinned if not r.label]
        for r in pinned:
            if r.label:
                out.append(Example(id=_eid("t", r.state), state=r.state, label=r.label, source="imported", split="test",
                                   rationale=f"label from export ({r.raw_label})", label_source="export",
                                   meta={"file": r.file, "row": r.row, "label_origin": "export",
                                         "raw_label": r.raw_label, "frozen_test": True}))
        out += await self.label_real(
            [(r.state, Example(id=_eid("t", r.state), state=r.state, label="", source="imported", split="test",
                               meta={"file": r.file, "row": r.row, "frozen_test": True})) for r in todo],
            "frozen-test rows")
        self.fixed_test = out
        if out:
            self.stats["fixed_test"] = len(out)
            await self._log(f"frozen test split from the export: {len(out)} rows {dict(Counter(r.label for r in out))}")

    async def mine(self) -> list[tuple[str, str | None]]:
        if not self.repo_path:
            return []
        from forge import repo as repo_mod
        root = Path(self.repo_path)
        sc = await asyncio.to_thread(repo_mod.scan, root)
        files = repo_mod.relevant_files(sc, [f.path for f in (self.opp.files if self.opp else [])], limit=40)
        files = [f for f in files if f not in self.imported_files][:10]  # imported tables are already used row by row
        await self._progress("mining", 0, len(files))
        found: list[tuple[str, str | None]] = []
        done = 0
        system = ("You extract realistic examples of the state text for a classifier from source files: test "
                  "fixtures, few-shot examples in prompts, sample data, docstrings, enum definitions. Copy real "
                  "text where it exists; do not invent. Return an empty list if the file has none.")

        async def one(rel: str):
            nonlocal done
            text = repo_mod.read_text(root, rel, 14000)
            prompt = (f"{_task_brief(self.task)}\n\nFile `{rel}`:\n```\n{text}\n```\n\n"
                      f"Extract up to 30 state texts found in this file. Set `label` to one of {self.labels} only "
                      "when the file explicitly pairs the example with that label (e.g. a test's expected value).")
            async with self.sem:
                try:
                    res = await self.llm.structured(prompt, MinedBatch, system=system)
                    items = res.items
                except LLMError as e:
                    await self._log(f"mining {rel} failed: {e}", "warn")
                    items = []
            for it in items:
                st = (it.state or "").strip()
                if 12 <= len(st) <= 6000:
                    found.append((st, norm_label(self.task, it.label)))
            done += 1
            await _call(self.emit, {"type": "tool", "name": "mine", "input": rel, "output": f"{len(items)} states"})
            await self._progress("mining", done, len(files))

        await asyncio.gather(*(one(f) for f in files))
        self.stats["mined"] = len(found)
        await self._log(f"mined {len(found)} candidate states from {len(files)} files")
        return found

    # 2. generate (one worker agent per label, in parallel) ------------------------------------------
    async def plan_voices(self) -> None:
        """Personas and styles for the generators, derived from the task's state description.

        The built-in PERSONAS/STYLES describe support-ticket writers; for any other kind of state
        (speech transcripts, log lines, product listings ...) they ask for unrealistic texts that QA
        then rejects. One LLM call adapts them to the state; on failure the built-ins are kept."""
        system = ("You plan the diversity of a synthetic training set. Describe who or what produces the "
                  "state text and the forms it takes in reality; never mention or hint at any label.")
        prompt = (f"{_task_brief(self.task)}\n\nList 8-12 personas (who or what produces this state text, as it "
                  "is actually produced at the call site) and 6-10 styles (realistic forms, lengths and registers "
                  "of the state text). Every persona and style must make sense for every label.")
        async with self.sem:
            try:
                v = await self.llm.structured(prompt, Voices, system=system)
            except LLMError as e:
                await self._log(f"voice planning failed ({e}); using the built-in personas and styles", "warn")
                return
        personas = [p.strip() for p in v.personas if p.strip()]
        styles = [x.strip() for x in v.styles if x.strip()]
        if len(personas) >= 3 and len(styles) >= 3:
            self.personas, self.styles = personas, styles
            await self._log(f"generator voices: {len(personas)} personas, {len(styles)} styles "
                            f"(e.g. {personas[0]!r}; {styles[0]!r})")

    def _gen_jobs(self, lab: str, n: int) -> list[tuple[str | None, int]]:
        """Batches for one label: ~25% hard negatives, each resembling a different other label."""
        hard = int(round(n * 0.25)) if len(self.labels) > 1 else 0
        easy = n - hard
        jobs: list[tuple[str | None, int]] = [(None, min(20, easy - lo)) for lo in range(0, easy, 20)]
        others = [x for x in self.labels if x != lab]
        jobs += [(others[j % len(others)], min(10, hard - lo)) for j, lo in enumerate(range(0, hard, 10))]
        return jobs

    def _exemplar_block(self, exemplars: list[tuple[str, str]], batch_id: str) -> tuple[str, list[str]]:
        """4 scrubbed real exemplars for one prompt -> (prompt text, their hashes)."""
        picks = shaping.pick_exemplars(exemplars, 4, self.rng)
        if self._pool_names is None or self._pool_names[0] is not exemplars:
            self._pool_names = (exemplars, shaping.pool_names(st for st, _ in exemplars))
        parts = []
        for j, (st, lab) in enumerate(picks):
            clean = shaping.scrub(st, self._pool_names[1])     # first: every descriptor below reads the scrubbed text
            parts.append(f"[exemplar {j + 1}] label {lab}; structure {shaping.key_template(clean)}; length "
                         f"{shaping.length_band(clean)}\n{clean[:1500]}")
        text = ("Real examples of this state text, scrubbed (typed placeholders like <NAME>, <EMAIL>, <NUM> "
                "replace identifying details). Match their structure, field order, register and length, but "
                "write new people, organisations, topics and wording; never copy a phrase from them and never "
                "write a placeholder token:\n\n" + "\n\n".join(parts) + "\n")
        return text, [shaping.exemplar_hash(st) for st, _ in picks]

    async def generate(self, need_per_label: dict[str, int],
                       feedback: dict[str, str] | None = None,
                       exemplars: list[tuple[str, str]] | None = None) -> list[tuple[str, str, str]]:
        """-> [(state, intended_label, source)]. Each label is its own worker; all workers run at once
        and their batches share `gen_sem`. `feedback` (per label, from QA) is added to that worker's prompts.
        ``exemplars`` ([(real state, label)]) shape every prompt on 4 scrubbed real rows; the resulting
        states are recorded in ``self.shaped_meta`` (train-only, leak-checked after the split)."""
        plan = {lab: self._gen_jobs(lab, n) for lab, n in need_per_label.items() if n > 0}
        total = sum(len(j) for j in plan.values())
        workers = {lab: {"done": 0, "total": len(j), "generated": 0} for lab, j in plan.items()}
        await self._progress("generating", 0, total, workers=workers)
        out: list[tuple[str, str, str]] = []
        system = ("You write realistic, diverse input texts for training a classifier. Every text must clearly "
                  "belong to the requested label to a careful reader. Vary vocabulary, length, tone and details; "
                  "never number them, never mention the label name verbatim unless natural.")
        guidance = f"\nCuration guidance from the reviewer (follow it): {self.guidance}\n" if self.guidance else ""
        kw = {"temperature": 0.9} if "temperature" in inspect.signature(self.llm.structured).parameters else {}

        async def batch(lab: str, near: str | None, n: int):
            persona, style = self.rng.choice(self.personas), self.rng.choice(self.styles)
            if near is None:
                ask = (f"Write {n} distinct state texts whose correct label is {label_desc(self.task, lab)}. "
                       f"Voice: {persona}. Style: {style}.")
                src = "synthetic"
            else:
                ask = (f"Write {n} distinct HARD state texts whose correct label is {label_desc(self.task, lab)} "
                       f"but which superficially resemble {label_desc(self.task, near)} (shared vocabulary, "
                       f"mixed signals) — a careful reader still picks {lab}. Voice: {persona}. Style: {style}.")
                src = "hard_negative"
            qa = (feedback or {}).get(lab)
            fix = f"\nA quality reviewer rejected earlier texts for this label. Fix this: {qa}\n" if qa else ""
            shape, hashes, batch_id = "", [], f"g{self.rng.getrandbits(40):010x}"
            if exemplars:
                shape, hashes = self._exemplar_block(exemplars, batch_id)
                ask += f" Every text must be labelled {lab}, whatever the exemplars' labels are."
            prompt = f"{_task_brief(self.task)}\n{guidance}{fix}\n{shape}\n{ask}"
            async with self.gen_sem:
                self._check()
                try:
                    states = (await self.llm.structured(prompt, GenBatch, system=system, **kw)).states
                except LLMError as e:
                    # the error may quote the model's output, which saw the exemplars: logged scrubbed and short
                    await self._log(f"generation batch for {lab} failed: "
                                    f"{shaping.scrub(str(e), (self._pool_names or ((), ()))[1])[:300]}", "warn")
                    states = []
            got = [(st, lab, src) for st in (x.strip() for x in states[: n + 2]) if 8 <= len(st) <= 6000]
            if exemplars:
                placeholder = re.compile(r"<(NAME|EMAIL|PHONE|URL|ID|NUM)>")
                kept = [g for g in got if not placeholder.search(g[0])]
                self.stats["shaped_drops"]["placeholder"] = (self.stats["shaped_drops"].get("placeholder", 0)
                                                            + len(got) - len(kept))
                got = kept
                for st, _l, _s in got:
                    self.shaped_meta[st] = {"exemplar_hashes": hashes, "batch": batch_id}
            out.extend(got)
            w = workers[lab]
            w["done"] += 1
            w["generated"] += len(got)
            await self._progress("generating", sum(v["done"] for v in workers.values()), total,
                                 generated=len(out), workers=workers)

        await self._log(f"generating with {len(plan)} parallel workers (one per label), "
                        f"up to {self.gen_sem._value} batches at once")
        for lab, jobs in plan.items():
            await _call(self.emit, {"type": "tool", "name": "generate", "input": lab,
                                    "output": f"worker started: {sum(n for _, n in jobs)} texts in {len(jobs)} batches"})
        # gen_sem wakes waiters FIFO, so batches are queued round-robin across labels: queued label by label,
        # the first label's batches would take every slot until they drained and the workers would run in turn.
        queues = [list(jobs) for jobs in plan.values()]
        order = [(lab, q[i]) for i in range(max(map(len, queues), default=0))
                 for lab, q in zip(plan, queues) if i < len(q)]
        await asyncio.gather(*(batch(lab, near, n) for lab, (near, n) in order))
        self.stats["generated"] += len(out)
        return out

    # 2b. orchestrator QA gate ------------------------------------------------------------------------
    async def review(self, lab: str, items: list[tuple[str, str, str]]) -> tuple[set[int], float, str]:
        """Orchestrator reviews a sample of one worker's output. -> (rejected indexes into items, pass rate,
        feedback). Unsampled items are kept; the pass rate decides whether the label is regenerated."""
        if not items:
            return set(), 1.0, ""
        idx = list(range(len(items)))
        self.rng.shuffle(idx)
        sample = sorted(idx[: self.qa_sample])
        listing = "\n\n".join(f"[{k}] ({items[i][2].replace('_', ' ')})\n{items[i][0][:2000]}"
                               for k, i in enumerate(sample))
        system = ("You are the orchestrator of a training-data team, reviewing a generator agent's work before "
                  "it goes to labelling. Be strict but fair: reject only texts that would hurt a classifier.")
        prompt = (f"{_task_brief(self.task)}\n\nEvery text below was written to have the label "
                  f"{label_desc(self.task, lab)}. 'hard negative' texts deliberately share vocabulary with "
                  "another label but must still belong to this one.\n\nMark each text ok=false if it: belongs to "
                  "a different label or is ambiguous; is unrealistic for the state described (wrong format, "
                  "meta-commentary, placeholders); names the label as a giveaway; or near-duplicates another text "
                  "in the list (reject the later copy). Then give the generator concrete feedback on recurring "
                  f"problems (empty if none).\n\nReturn one entry per index 0..{len(sample) - 1}.\n\n{listing}")
        async with self.sem:
            self._check()
            try:
                res = await self.llm.structured(prompt, QAReview, system=system)
            except LLMError as e:
                await self._log(f"QA review of {lab} failed ({e}); passing it through unreviewed", "warn")
                return set(), 1.0, ""
        bad = {sample[it.i] for it in res.items if not it.ok and 0 <= it.i < len(sample)}
        issues = [it.issue for it in res.items if not it.ok and it.issue][:3]
        rate = 1 - len(bad) / len(sample)
        await _call(self.emit, {"type": "tool", "name": "qa", "input": lab,
                                "output": f"{len(sample) - len(bad)}/{len(sample)} passed"
                                          + (f"; e.g. {'; '.join(issues)}" if issues else "")})
        return bad, rate, res.feedback.strip()

    async def quality_gate(self, generated: list[tuple[str, str, str]],
                           need: dict[str, int]) -> list[tuple[str, str, str]]:
        """Review every worker's output. Rejected sample items are dropped; a label whose pass rate is below
        `qa_min_pass` is regenerated once with the orchestrator's feedback and reviewed again."""
        by: dict[str, list] = defaultdict(list)
        for g in generated:
            by[g[1]].append(g)
        await self._progress("reviewing", 0, len(by))
        kept: list[tuple[str, str, str]] = []
        retry: dict[str, str] = {}
        done = 0

        async def check(lab: str, items: list, attempt: int):
            nonlocal done
            bad, rate, fb = await self.review(lab, items)
            self.stats["qa_rejected"] += len(bad)
            passed = rate >= self.qa_min_pass
            self.stats["qa"][lab] = {"pass_rate": round(rate, 3), "rejected": len(bad), "attempt": attempt,
                                     "passed": passed, "feedback": fb}
            if passed or attempt > 1:
                kept.extend(it for i, it in enumerate(items) if i not in bad)
                if not passed:
                    await self._log(f"QA {lab}: still {rate:.0%} after regeneration; keeping the passing "
                                    "texts, the teacher relabel will filter the rest", "warn")
            else:
                retry[lab] = fb or "texts were off-label, unrealistic or repetitive; make each one clearly fit the label"
                await self._log(f"QA {lab}: only {rate:.0%} of the sample passed (< {self.qa_min_pass:.0%}); "
                                f"sending back for regeneration: {retry[lab]}", "warn")
            done += 1
            await self._progress("reviewing", done, len(by) + len(retry))

        await asyncio.gather(*(check(lab, items, 1) for lab, items in by.items()))
        if retry:
            self._check()
            regen = await self.generate({lab: need.get(lab, len(by[lab])) for lab in retry}, feedback=retry)
            again: dict[str, list] = defaultdict(list)
            for g in regen:
                again[g[1]].append(g)
            await self._progress("reviewing", done, len(by) + len(retry))
            await asyncio.gather(*(check(lab, again.get(lab, []), 2) for lab in retry))
        summary = ", ".join(f"{lab} {v['pass_rate']:.0%}" + ("" if v["passed"] else " ✗")
                            for lab, v in sorted(self.stats["qa"].items()))
        await self._log(f"orchestrator QA: {summary}; dropped {self.stats['qa_rejected']} texts"
                        + (f"; regenerated {', '.join(sorted(retry))}" if retry else ""))
        return kept

    # 4. dedup --------------------------------------------------------------------------------------
    async def dedup(self, rows: list[Example], frozen: int = 0) -> list[Example]:
        """Exact + embedding near-dup removal. The first `frozen` rows are always kept."""
        prio = SOURCE_PRIORITY
        head, tail = rows[:frozen], sorted(rows[frozen:], key=lambda r: prio.get(r.source, 9))
        seen, kept = set(), []
        for idx, r in enumerate(head + tail):
            key = _norm_text(r.state)
            if key in seen and idx >= frozen:
                self.stats["exact_dups"] += 1
                continue
            seen.add(key)
            kept.append(r)
        await self._progress("dedup", 0, len(kept), kept)
        if not self.use_embeddings or len(kept) < 2:
            return kept
        try:
            from forge.encoder import get_encoder  # ML side; may not exist in tests
            enc = get_encoder(self.encoder_spec)
        except Exception as e:  # noqa: BLE001
            await self._log(f"embedding dedup skipped (encoder unavailable: {e})", "warn")
            return kept
        import numpy as np

        loop = asyncio.get_running_loop()
        texts = [r.state for r in kept]

        def progress(d, t):
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(
                _call(self.emit, {"type": "embed", "done": d, "total": t})))

        def status(level: str, msg: str) -> None:   # model lock / memory waits and back-offs, as log events
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._log(msg, level)))

        await self._log(f"embedding {len(texts)} states for near-duplicate removal")
        vecs = await asyncio.to_thread(enc.embed, texts, progress, on_status=status, should_stop=self.should_stop)
        vecs = np.asarray(vecs, dtype=np.float32)
        sims = vecs @ vecs.T
        # Qwen3 last-token embeddings are anisotropic: unrelated short texts already sit at cosine
        # ~0.93-0.96, paraphrases at ~0.98-0.99. Start at the configured threshold and raise it if it
        # would discard more than 30% of the new rows (that is "same topic", not "duplicate").
        # Only rows with the SAME label can be near-duplicates: two close states with different labels
        # are a minimal pair (e.g. one long shared context, a short differing part), the most
        # informative rows there are, never redundant.
        labs = np.array([r.label for r in kept], dtype=object)
        thr = self.dedup_threshold
        while True:
            keep_mask = np.ones(len(kept), dtype=bool)
            for i in range(len(kept)):
                if not keep_mask[i]:
                    continue
                dup = np.where((sims[i, i + 1:] > thr) & (labs[i + 1:] == labs[i]))[0] + i + 1
                keep_mask[dup[dup >= frozen]] = False
            removed = int((~keep_mask).sum())
            if removed <= 0.3 * max(1, len(kept) - frozen) or thr >= 0.995:
                break
            thr = round(thr + 0.005, 3)
        self.stats["near_dups"] += removed
        self.stats["dedup_threshold"] = thr
        await self._log(f"near-duplicate removal at cosine > {thr}: dropped {removed}")
        out = [r for r, k in zip(kept, keep_mask) if k]
        await self._progress("dedup", len(out), len(kept), out)
        return out

    # 5. balance + split ----------------------------------------------------------------------------
    def balance(self, rows: list[Example], cap: int) -> list[Example]:
        prio = SOURCE_PRIORITY
        by: dict[str, list[Example]] = defaultdict(list)
        for r in rows:
            by[r.label].append(r)
        out = []
        for lab, rs in by.items():
            self.rng.shuffle(rs)
            rs.sort(key=lambda r: prio.get(r.source, 9))
            out.extend(rs[:cap])
            self.stats["capped"] += max(0, len(rs) - cap)
        return out

    def split(self, rows: list[Example], allow_test: bool = True) -> list[Example]:
        by: dict[str, list[Example]] = defaultdict(list)
        for r in rows:
            by[r.label].append(r)
        out = []
        for lab, rs in by.items():
            self.rng.shuffle(rs)
            n = len(rs)
            n_test = int(round(n * 0.15)) if allow_test and n >= 4 else 0
            n_val = int(round(n * (0.15 if allow_test else 0.15 / 0.85))) if n >= 3 else 0
            # real rows (imported, then mined) preferred for test
            test_pool = sorted(rs, key=lambda r: SOURCE_PRIORITY.get(r.source, 9))
            test = test_pool[:n_test]
            test_ids = {id(r) for r in test}
            rest = [r for r in rs if id(r) not in test_ids]
            val, train = rest[:n_val], rest[n_val:]
            if not train and val:
                train, val = val[:1], val[1:]
            for r in test:
                r.split = "test"
            for r in val:
                r.split = "val"
            for r in train:
                r.split = "train"
            out.extend(train + val + test)
        return out

    # orchestration ---------------------------------------------------------------------------------
    async def run(self) -> list[Example]:
        k = len(self.labels)
        per_label = math.ceil(self.target / k)
        keep_prev = [r for r in self.previous if r.label in self.labels]
        is_recurate = bool(keep_prev)
        if is_recurate:
            await self._log(f"recurating: keeping {len(keep_prev)} existing rows (test split frozen)"
                            + (f"; guidance: {self.guidance}" if self.guidance else ""))

        # 0. import exported tables (also on recurate, so newly exported data is used), 1. mine (first pass only)
        imported = await self.import_data(per_label)
        if is_recurate:
            prev_states = {_norm_text(r.state) for r in keep_prev}
            imported = [r for r in imported if _norm_text(r.state) not in prev_states]
            new_test = [r for r in self.fixed_test if _norm_text(r.state) not in prev_states]
            if new_test:
                await self._log(f"ignoring {len(new_test)} newly exported test rows: the test split is frozen "
                                "from the first pass", "warn")
            self.fixed_test = []
        fixed_test = self.fixed_test
        self._check()
        mined = [] if is_recurate else await self.mine()
        self._check()

        # 2. generate to fill each label (over-generate ~40% for relabel drops + dedup)
        have = Counter(r.label for r in keep_prev + imported)
        for _, lab in mined:
            if lab:
                have[lab] += 1
        if is_recurate:
            extra = max(per_label // 2, 20)
            need = {lab: extra for lab in self.labels}
        else:
            need = {lab: max(0, math.ceil((per_label - have[lab]) * 1.4)) for lab in self.labels}
        if any(need.values()):
            await self.plan_voices()
        # real labelled rows (never the frozen test rows) are the shape guide for generation
        exemplars = [(r.state, r.label) for r in keep_prev + imported
                     if r.source in REAL_SOURCES and r.label and r.split == "train"]
        exemplars += [(s, lab) for s, lab in mined if lab]
        if exemplars:
            await self._log(f"generation is shaped on {len(exemplars)} real labelled rows (scrubbed exemplars, "
                            "4 per prompt); shaped rows stay in train")
        generated = await self.generate(need, exemplars=exemplars or None)
        self._check()
        generated = await self.quality_gate(generated, need)
        self._check()

        # 3. real mined rows without a stated label follow the label policy; generated rows get the
        #    independent teacher relabel
        mined_real = await self.label_real(
            [(s, Example(id=_eid("m", s), state=s, label="", source="mined", split="train")) for s, lab in mined
             if lab is None], "mined states")
        to_label: list[tuple[str, str | None, str]] = [(s, lab, src) for s, lab, src in generated]
        await self._progress("labelling", 0, len(to_label))
        labelled_states = [s for s, _, _ in to_label]
        results: list[tuple[str | None, float, str]] = []
        B = 15
        for lo in range(0, len(labelled_states), B * 4):
            self._check()
            chunk = labelled_states[lo:lo + B * 4]
            results += await label_states(self.llm, self.task, chunk, batch=B, sem=self.sem)
            await self._progress("labelling", min(len(labelled_states), lo + len(chunk)), len(labelled_states))
        rows: list[Example] = list(imported) + mined_real
        for (s, lab) in [(s, lab) for s, lab in mined if lab is not None]:
            rows.append(Example(id=_eid("m", s), state=s, label=lab, source="mined", split="train",
                                rationale="label stated in source", label_source="export"))
        for (s, intended, src), (got, conf, why) in zip(to_label, results):
            if got is None:
                self.stats["invalid"] += 1
                continue
            if intended is not None and got != intended:
                self.stats["disagreements"] += 1
                continue
            shaped = self.shaped_meta.get(s)
            rows.append(Example(id=_eid(src[0], s), state=s, label=got, source=src, split="train",
                                rationale=why, provenance=REAL_SHAPED if shaped else None,
                                meta={"teacher_confidence": conf, "intended": intended, **(shaped or {})}))
        await self._progress("labelling", len(labelled_states), len(labelled_states), rows,
                             disagreements=self.stats["disagreements"])
        await self._log(f"teacher relabel: kept {len(rows)}, dropped {self.stats['disagreements']} disagreements, "
                        f"{self.stats['invalid']} unlabelled")

        # 4. dedup (previous rows and the export's frozen test rows first), 5. balance + split
        all_rows = await self.dedup(keep_prev + fixed_test + rows, frozen=len(keep_prev) + len(fixed_test))
        self._check()
        prev_ids = {r.id for r in keep_prev} | {r.id for r in fixed_test}
        new_rows = [r for r in all_rows if r.id not in prev_ids]
        old_rows = [r for r in all_rows if r.id in prev_ids]
        await self._progress("splitting", 0, len(all_rows), all_rows)
        if is_recurate:
            cap = per_label * 2  # recurate grows the dataset
            new_rows = self.balance(new_rows, cap)
            final = old_rows + self._split_shaped(new_rows, allow_test=False)
        elif fixed_test:
            final = old_rows + self._split_shaped(self.balance(new_rows, per_label), allow_test=False)
        else:
            final = self._split_shaped(self.balance(new_rows, per_label))
        final = await self._leak_check(final)
        # unique ids
        seen: set[str] = set()
        for r in final:
            while r.id in seen:
                r.id = r.id + "x"
            seen.add(r.id)
        test = [r for r in final if r.split == "test"]
        if test:
            real = sum(1 for r in test if r.source in REAL_SOURCES)
            self.stats["test_real"] = real
            await self._log(f"test split: {real}/{len(test)} rows are real data"
                            + ("" if real == len(test) else " (the rest is generated, so test scores partly measure "
                                                             "fit to generated text)"), "info" if real == len(test) else "warn")
        await self._queue_real(final)
        await self._progress("done", len(final), len(final), final, stats=self.stats)
        return final

    def _split_shaped(self, rows: list[Example], allow_test: bool = True) -> list[Example]:
        """Stratified split of everything except real-shaped rows, which always go to train."""
        shaped = [r for r in rows if r.provenance == REAL_SHAPED]
        for r in shaped:
            r.split = "train"
        return self.split([r for r in rows if r.provenance != REAL_SHAPED], allow_test=allow_test) + shaped

    async def _leak_check(self, final: list[Example]) -> list[Example]:
        """Drop generated train rows that leak a val/test row, and real-shaped rows that copy the real pool."""
        gen = [r for r in final if r.split == "train" and r.source in ("synthetic", "hard_negative")]
        if not gen:
            return final
        held_out = [r.state for r in final if r.split in ("val", "test")]
        real = [r.state for r in final if r.source in REAL_SOURCES] + [r.state for r in self.queued]
        guard = await asyncio.to_thread(shaping.LeakGuard, held_out, real)
        drops: Counter = Counter()
        bad: set[int] = set()
        for r in gen:
            why = guard.check(r.state, real_shaped=r.provenance == REAL_SHAPED)
            if why:
                drops[why] += 1
                bad.add(id(r))
        for k, v in drops.items():
            self.stats["shaped_drops"][k] = self.stats["shaped_drops"].get(k, 0) + v
        self.stats["shaped"] = sum(1 for r in final if r.provenance == REAL_SHAPED and id(r) not in bad)
        if drops:
            await self._log(f"leak check dropped {sum(drops.values())} generated train rows: {dict(drops)}", "warn")
        return [r for r in final if id(r) not in bad]

    async def _queue_real(self, final: list[Example]) -> None:
        """Hand real rows to the labelling queue: the ones still needing a label, plus the teacher-labelled
        real rows that made it into the dataset (so the user can review and override them)."""
        have = {_norm_text(r.state) for r in final}
        pending = []
        for r in self.queued:   # a pending row duplicating a dataset row adds nothing
            k = _norm_text(r.state)
            if k not in have:
                have.add(k)
                pending.append(r)
        mirrors = [from_example(r, head=self.task.name) for r in final
                   if r.source in REAL_SOURCES and r.label_source == "teacher"]
        if self.queue is not None:
            added = await asyncio.to_thread(self.queue.add, pending + mirrors)
            self.stats["queued"] = added
            self.stats["needs_label"] = len(self.queue.pending())
        else:
            self.stats["queued"] = len(pending) + len(mirrors)
            self.stats["needs_label"] = len(pending)
        if self.stats["needs_label"]:
            await self._log(f"{self.stats['needs_label']} real rows need a label before training", "warn")


def write_task(pdir: Path, task: TaskSchema) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "task.json").write_text(json.dumps(task.model_dump(), indent=2))
