"""Import real examples from data the user exported into the repo (CSV / TSV / JSONL tables).

Discovery: every table under ``clm_data/`` (read even when gitignored, since exports of production
data usually are) plus any tracked/untracked-but-not-ignored ``.csv``/``.tsv``/``.jsonl`` elsewhere.

Per table the LLM sees only the column profile and a few sample rows, and returns a mapping: whether
the table is relevant, a template that renders a row into the call-site state text, and optionally
which column holds the label and how its raw values map to label keys. Applying the mapping to the
rows is deterministic, so every imported state is real text copied from the export, never retyped.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from forge import repo as repo_mod
from forge.llm.base import LLM, LLMError
from forge.models import TaskSchema

DATA_DIR_NAME = "clm_data"
TABLE_EXT = {".csv", ".tsv", ".jsonl", ".ndjson"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "site-packages"}
MAX_TABLES = 20
MAX_TABLE_BYTES = 1_000_000_000
MAX_READ_ROWS = 500_000      # stop reading a table after this many rows
SAMPLE_ROWS = 20_000         # reservoir kept per table for mapping + import
EMPTY = {"", "null", "none", "nan", "n/a", "na", "-"}
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
SPLIT_COLUMN = "split"
TEST_SPLIT_VALUES = {"test", "testing", "holdout", "held_out", "held-out"}


class TableMapping(BaseModel):
    relevant: bool = Field(description="true only if rows of this table can be rendered into the classifier's state text")
    reason: str = Field(description="one sentence: why relevant or not, and what the label column means")
    template: str = Field(default="", description="state text template; {Column Name} placeholders are replaced "
                                                  "by that row's value; must mirror the call-site state format")
    label_column: str | None = Field(default=None, description="column holding the ground-truth answer, if any")
    label_map: dict[str, str] = Field(default_factory=dict,
                                      description="raw label_column value -> label key; omit ambiguous values")


@dataclass
class Table:
    rel: str
    columns: list[str]
    rows: list[dict[str, str]]
    total_rows: int
    truncated: bool = False


@dataclass
class ImportedRow:
    state: str
    label: str | None
    file: str
    row: int
    raw_label: str | None = None
    split: str | None = None   # "test" when the export pins the row to the frozen test split


@dataclass
class ImportResult:
    rows: list[ImportedRow] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)   # rel -> outcome summary


def _blank(v: str | None) -> bool:
    return v is None or v.strip().lower() in EMPTY


def discover(root: Path) -> list[str]:
    """Relative paths of candidate tables, ``clm_data/`` first."""
    out: list[str] = []
    data_dir = root / DATA_DIR_NAME
    if data_dir.is_dir():
        for dp, dns, fns in os.walk(data_dir):
            dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() in TABLE_EXT:
                    out.append(os.path.relpath(os.path.join(dp, fn), root))
    tracked = repo_mod._git_files(root) or []
    for rel in sorted(tracked):
        if rel in out or os.path.splitext(rel)[1].lower() not in TABLE_EXT:
            continue
        if any(p in SKIP_DIRS for p in rel.split("/")[:-1]):
            continue
        out.append(rel)
    # never follow a symlink out of the repo (see repo.safe_path)
    sized = []
    for rel in out:
        fp = repo_mod.safe_path(root, rel)
        if fp is None:
            continue
        try:
            size = fp.stat().st_size
        except OSError:
            continue
        if 0 < size <= MAX_TABLE_BYTES:
            sized.append(rel)
    return sized[:MAX_TABLES]


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def read_table(root: Path, rel: str, rng: random.Random) -> Table | None:
    """Read a table, keeping a uniform reservoir sample of up to SAMPLE_ROWS rows."""
    path = repo_mod.safe_path(root, rel)
    if path is None:
        return None
    ext = os.path.splitext(rel)[1].lower()
    sample: list[dict[str, str]] = []
    columns: list[str] = []
    n = 0

    def keep(row: dict[str, str]):
        nonlocal n
        n += 1
        if len(sample) < SAMPLE_ROWS:
            sample.append(row)
        else:
            j = rng.randrange(n)
            if j < SAMPLE_ROWS:
                sample[j] = row

    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            if ext in (".jsonl", ".ndjson"):
                seen: dict[str, None] = {}
                for line in f:
                    if n >= MAX_READ_ROWS:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    for k in obj:
                        seen.setdefault(str(k))
                    keep({str(k): _cell(v) for k, v in obj.items()})
                columns = list(seen)
            else:
                head = f.read(65536)
                f.seek(0)
                delim = "\t" if ext == ".tsv" else ","
                if ext != ".tsv":
                    try:
                        delim = csv.Sniffer().sniff(head, delimiters=",\t;|").delimiter
                    except csv.Error:
                        pass
                csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
                reader = csv.DictReader(f, delimiter=delim)
                columns = [c for c in (reader.fieldnames or []) if c]
                for row in reader:
                    if n >= MAX_READ_ROWS:
                        break
                    keep({c: _cell(row.get(c)) for c in columns})
    except (OSError, csv.Error, UnicodeError):
        return None
    if not columns or not sample:
        return None
    return Table(rel=rel, columns=columns, rows=sample, total_rows=n, truncated=n >= MAX_READ_ROWS)


def split_column(table: Table) -> str | None:
    """The table's split column (case-insensitive ``split``), if it has one."""
    return next((c for c in table.columns if c.strip().lower() == SPLIT_COLUMN), None)


def input_columns(table: Table) -> list[str]:
    """Columns the mapping may use: everything except the split column."""
    sc = split_column(table)
    return [c for c in table.columns if c != sc]


def profile(table: Table, n_samples: int = 3) -> str:
    """Column summary + a few sample rows: all the LLM sees of the table (the split column is hidden)."""
    cols = input_columns(table)
    lines = [f"Table `{table.rel}`: {table.total_rows}{'+' if table.truncated else ''} rows, "
             f"{len(cols)} columns."]
    for c in cols:
        vals = [r.get(c, "") for r in table.rows]
        filled = [v for v in vals if not _blank(v)]
        counts = Counter(v.strip() for v in filled)
        avg = sum(len(v) for v in filled) / max(1, len(filled))
        desc = f"- {c!r}: {len(filled)}/{len(vals)} non-empty, {len(counts)} distinct, avg {avg:.0f} chars"
        if 0 < len(counts) <= 40:
            desc += "; values: " + ", ".join(f"{v[:60]!r}×{k}" for v, k in counts.most_common(25))
        elif filled:
            desc += "; e.g. " + " | ".join(repr(v[:120]) for v in filled[:2])
        lines.append(desc)
    lines.append("\nSample rows:")
    for r in table.rows[:n_samples]:
        lines.append(json.dumps({c: (r.get(c, "")[:400]) for c in cols}, ensure_ascii=False))
    return "\n".join(lines)


def render(template: str, row: dict[str, str]) -> str | None:
    """Fill {Column} placeholders. Lines whose placeholders are all blank are dropped; None if nothing filled."""
    out_lines, any_filled = [], False
    for line in template.splitlines():
        names = _PLACEHOLDER.findall(line)
        if names and all(_blank(row.get(n)) for n in names):
            continue
        filled = _PLACEHOLDER.sub(lambda m: "" if _blank(row.get(m.group(1))) else row[m.group(1)].strip(), line)
        any_filled = any_filled or bool(names)
        out_lines.append(filled.rstrip())
    text = "\n".join(out_lines).strip()
    return text if any_filled and text else None


def validate(mapping: TableMapping, table: Table, task: TaskSchema) -> tuple[TableMapping | None, str]:
    """Deterministic checks on the LLM's mapping. -> (mapping with normalised label_map, problem)."""
    from forge.curator import norm_label
    if not mapping.relevant:
        return None, f"not relevant: {mapping.reason}"
    names = _PLACEHOLDER.findall(mapping.template)
    if not names:
        return None, "mapping has no {column} placeholders"
    cols = input_columns(table)
    unknown = [n for n in names if n not in cols]
    if unknown:
        return None, f"template names unknown columns {unknown}"
    lab_map: dict[str, str] = {}
    if mapping.label_column:
        if mapping.label_column not in cols:
            return None, f"label column {mapping.label_column!r} does not exist"
        if mapping.label_column in names:
            return None, f"template includes the label column {mapping.label_column!r} (would leak the answer)"
        for raw, lab in mapping.label_map.items():
            key = norm_label(task, lab)
            if key is not None:
                lab_map[raw.strip().lower()] = key
        if not lab_map:
            return None, f"no values of {mapping.label_column!r} map to a label"
    return mapping.model_copy(update={"label_map": lab_map}), ""


async def map_table(llm: LLM, task: TaskSchema, table: Table) -> TableMapping:
    from forge.curator import _task_brief
    system = ("You map exported database tables onto a classifier's input. You never invent data: you only "
              "say which columns to copy into the state text and which column holds the ground-truth label.")
    prompt = (f"{_task_brief(task)}\n\n{profile(table)}\n\n"
              "Decide whether rows of this table are real instances of the state this classifier sees.\n"
              "If so, write `template`: text with {Column Name} placeholders (exact column names) that renders a "
              "row into the same shape and field order as the example state above. Use only input columns that "
              "exist at the call site before the decision is made. Never put the label column, or any column "
              "that reveals the answer (status, outcome, decision, result, reviewer notes), in the template.\n"
              "If a column holds the ground-truth answer, set `label_column` and `label_map` from each of its raw "
              f"values (as written above) to one of {task.labels()}. Leave out values that are ambiguous or mean "
              "something else. Leave `label_column` null if no column holds the answer.")
    return await llm.structured(prompt, TableMapping, system=system)


async def import_tables(llm: LLM, task: TaskSchema, root: Path, rng: random.Random, log, tool) -> ImportResult:
    """Discover, map and render every table. ``log(msg, level)`` and ``tool(rel, summary)`` report progress."""
    res = ImportResult()
    for rel in discover(root):
        table = read_table(root, rel, rng)
        if table is None:
            res.files[rel] = "unreadable or empty"
            await tool(rel, res.files[rel])
            continue
        try:
            mapping = await map_table(llm, task, table)
        except LLMError as e:
            res.files[rel] = f"mapping failed: {e}"
            await log(f"import {rel}: {res.files[rel]}", "warn")
            await tool(rel, res.files[rel])
            continue
        mapping, problem = validate(mapping, table, task)
        if mapping is None:
            res.files[rel] = f"skipped ({problem})"
            await tool(rel, res.files[rel])
            continue
        n_rows, n_lab, n_unmapped, n_test = 0, 0, 0, 0
        sc = split_column(table)
        if sc and table.total_rows > len(table.rows):
            # sampling would silently drop frozen test rows; refuse rather than import a partial test set
            res.files[rel] = (f"skipped (has a {sc!r} column but {table.total_rows} rows exceed the "
                              f"{SAMPLE_ROWS}-row read sample; split the export into smaller tables)")
            await log(f"import {rel}: {res.files[rel]}", "warn")
            await tool(rel, res.files[rel])
            continue
        for i, row in enumerate(table.rows):
            state = render(mapping.template, row)
            if not state or not 12 <= len(state) <= 6000:
                continue
            raw = row.get(mapping.label_column, "") if mapping.label_column else None
            label = None
            if mapping.label_column and not _blank(raw):
                label = mapping.label_map.get((raw or "").strip().lower())
                if label is None:
                    n_unmapped += 1
                    continue  # the export has an answer for this row, but it maps to none of our labels
                n_lab += 1
            # a blank label cell is no answer: the row is real but unlabelled (the label policy decides)
            split = "test" if sc and (row.get(sc) or "").strip().lower() in TEST_SPLIT_VALUES else None
            n_test += split == "test"
            res.rows.append(ImportedRow(state=state, label=label, file=rel, row=i, raw_label=raw, split=split))
            n_rows += 1
        sampled = f" (sampled from {table.total_rows}{'+' if table.truncated else ''})" if table.total_rows > len(table.rows) else ""
        res.files[rel] = (f"{n_rows} rows{sampled}, "
                          + (f"labels from {mapping.label_column!r}" + (f", {n_unmapped} unmapped skipped" if n_unmapped else "")
                             + (f", {n_rows - n_lab} without a label" if n_rows > n_lab else "")
                             if mapping.label_column else "unlabelled (the label policy decides who labels them)")
                          + (f", {n_test} pinned to the frozen test split by {sc!r}" if n_test else ""))
        await log(f"import {rel}: {res.files[rel]}; template {mapping.template!r}")
        await tool(rel, res.files[rel])
    return res
