"""Real-shaped synthetic data: scrubbed real exemplars as a guide, and the guards that keep real text out.

Generation can follow the *shape* of real rows (structure, register, length, channel) without copying
their content. This module owns the pieces, used by ``Curator.generate(exemplars=...)``:

- ``scrub(text, names)``: names, emails, phone numbers, URLs, ids/uuids and numbers become typed placeholders
  (``<NAME>``, ``<EMAIL>``, ``<PHONE>``, ``<URL>``, ``<ID>``, ``<NUM>``) before an exemplar reaches a prompt;
  ``names`` (``pool_names``) adds the names the rest of the real pool shows, so one standing at a sentence start
  is replaced too;
- ``key_template`` / ``length_band``: the structural description the prompt carries with each exemplar, always
  taken from the **scrubbed** text (a chat line ``Jordan Pike: ...`` would otherwise put a name in the template);
- ``exemplar_block`` (``forge.curator``): nothing from an exemplar reaches a prompt, a log or an event unscrubbed;
- ``pick_exemplars``: a stratified sample (by label, then template) of real rows per prompt;
- ``LeakGuard``: the checks a generated row must pass —
  ``forbidden`` (4-gram Jaccard >= 0.8 against any val/test row), and against the whole real pool
  ``shared_8gram`` (any shared 8-gram), ``jaccard`` (4-gram Jaccard >= 0.5) and ``identifier`` (a token
  from the real identifier set: names, emails, ids, long numbers, URLs). ``check`` returns the first
  failing guard's name, or None.

Every function is deterministic; nothing here calls a model.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Iterable

FORBIDDEN_JACCARD = 0.8
REAL_JACCARD = 0.5
REAL_NGRAM = 8

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEXID = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{10,}\b", re.I)
_TOKID = re.compile(r"\b[A-Za-z]+[_-]?\d[\w-]*\b|\b\d+[A-Za-z][\w-]*\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_CAP = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\b")
_WORD = re.compile(r"<[A-Z]+>|\w+")
_KEY = re.compile(r"^\s*([A-Za-z][\w ./-]{0,30}?)\s*:\s", re.M)
# a sentence start: text start, after . ! ? or a line break, or a list bullet. Not after ':' — "From: Jordan" is a
# field value, and a name there must be scrubbed (first names after a colon reached prompts).
_SENT_START = re.compile(r"(^|[.!?\n]\s*|^\s*[-*]\s*)$")
# Capitalised words that are not names: sentence-initial words are skipped by position; these are too common.
_NOT_NAMES = frozenset("""I I'm I've I'll I'd Monday Tuesday Wednesday Thursday Friday Saturday Sunday January
February March April May June July August September October November December Subject From To Re Fwd Cc Hi
Hello Hey Dear Thanks Thank Regards Best Cheers Please OK Ok Yes No The A An This That These Those It We You
They He She Our Your My Their His Her""".split())


def _names(text: str) -> list[str]:
    """Capitalised words that are not at a sentence start and not common capitalised words."""
    out = []
    for m in _CAP.finditer(text):
        w = m.group(0)
        if w in _NOT_NAMES or _SENT_START.search(text[:m.start()]):
            continue
        out.append(w)
    return out


def pool_names(texts: Iterable[str]) -> set[str]:
    """Every name ``_names`` finds anywhere in a pool of real texts: a name that opens a sentence in one text is
    scrubbed there too when another text shows it mid-sentence (``scrub(names=...)``)."""
    return {w for t in texts for w in _names(t)}


def scrub(text: str, names: Iterable[str] = ()) -> str:
    """Replace identifying spans with typed placeholders (order matters: URLs and emails before numbers, ids
    before numbers, names last over what is left). ``names``: names known from the rest of the real pool
    (``pool_names``), replaced wherever they stand, sentence starts included."""
    t = _URL.sub("<URL>", text)
    t = _EMAIL.sub("<EMAIL>", t)
    t = _UUID.sub("<ID>", t)
    t = _HEXID.sub("<ID>", t)
    t = _PHONE.sub("<PHONE>", t)
    t = _TOKID.sub(lambda m: m.group(0) if m.group(0).startswith("<") else "<ID>", t)
    t = _NUM.sub("<NUM>", t)
    names = set(_names(t)) | {n for n in names if n and n not in _NOT_NAMES}
    if names:
        t = re.sub(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b", "<NAME>", t)
    return t


def identifiers(text: str) -> set[str]:
    """Tokens that identify a real person or record: names, emails, URLs, ids and numbers of 4+ digits."""
    ids = {m.group(0).lower() for rx in (_EMAIL, _URL, _UUID, _HEXID) for m in rx.finditer(text)}
    ids |= {m.group(0).lower() for m in _TOKID.finditer(text)}
    ids |= {m.group(0) for m in _NUM.finditer(text) if len(re.sub(r"\D", "", m.group(0))) >= 4}
    ids |= {w.lower() for w in _names(text) if len(w) >= 3}
    return ids


def key_template(text: str) -> str:
    """The row's structure: its leading ``Key:`` fields in order (``Subject|From|Body``), else ``free``."""
    keys = [k.strip().lower() for k in _KEY.findall(text)]
    return "|".join(dict.fromkeys(keys)) if keys else "free"


def length_band(text: str) -> str:
    n = len(text)
    return "short (< 200 chars)" if n < 200 else "medium (200-800 chars)" if n < 800 else "long (> 800 chars)"


def exemplar_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def pick_exemplars(pool: list[tuple[str, str]], k: int, rng: random.Random) -> list[tuple[str, str]]:
    """``k`` (state, label) exemplars: round-robin over labels, each from a different key template when
    possible, so one prompt sees the real spread of structures, not k copies of the commonest one."""
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for st, lab in pool:
        by_label[lab].append((st, lab))
    labels = sorted(by_label)
    rng.shuffle(labels)
    out: list[tuple[str, str]] = []
    seen_tpl: set[str] = set()
    i = 0
    while len(out) < min(k, len(pool)) and i < 4 * k + len(labels):
        lab = labels[i % len(labels)]
        cands = [x for x in by_label[lab] if x not in out]
        fresh = [x for x in cands if key_template(x[0]) not in seen_tpl] or cands
        if fresh:
            pick = rng.choice(fresh)
            out.append(pick)
            seen_tpl.add(key_template(pick[0]))
        i += 1
    return out


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _grams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else (
        {tuple(tokens)} if tokens else set())


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class _Index:
    """4-gram sets with an inverted index, so a Jaccard check only compares rows that share a 4-gram."""

    def __init__(self, texts: Iterable[str]):
        self.sets: list[set] = []
        self.inv: dict[tuple, list[int]] = defaultdict(list)
        for t in texts:
            g = _grams(_tokens(t), 4)
            for x in g:
                self.inv[x].append(len(self.sets))
            self.sets.append(g)

    def max_jaccard(self, g: set) -> float:
        cands = {i for x in g for i in self.inv.get(x, ())}
        return max((_jaccard(g, self.sets[i]) for i in cands), default=0.0)


class LeakGuard:
    """``forbidden`` = val/test texts; ``real`` = every real text the generator may have seen (or that a
    copy of would leak user data). ``extra_identifiers``: identifiers known from elsewhere."""

    GUARDS = ("forbidden", "shared_8gram", "jaccard", "identifier")

    def __init__(self, forbidden: Iterable[str], real: Iterable[str] = (),
                 extra_identifiers: Iterable[str] = ()):
        real = list(real)
        self.forbidden = _Index(forbidden)
        self.real = _Index(real)
        self.real8: set = set()
        self.ids: set[str] = {x.lower() for x in extra_identifiers}
        for t in real:
            self.real8 |= _grams(_tokens(t), REAL_NGRAM) if len(_tokens(t)) >= REAL_NGRAM else set()
            self.ids |= identifiers(t)

    def check(self, text: str, *, real_shaped: bool = True) -> str | None:
        """The first guard ``text`` fails, or None. Rows not shaped on real exemplars only face
        ``forbidden`` (the val/test leak check)."""
        toks = _tokens(text)
        g4 = _grams(toks, 4)
        if self.forbidden.max_jaccard(g4) >= FORBIDDEN_JACCARD:
            return "forbidden"
        if not real_shaped:
            return None
        if len(toks) >= REAL_NGRAM and _grams(toks, REAL_NGRAM) & self.real8:
            return "shared_8gram"
        if self.real.max_jaccard(g4) >= REAL_JACCARD:
            return "jaccard"
        if self.ids and (identifiers(text) & self.ids):
            return "identifier"
        return None
