"""Shared domain models. The contract between the agent, the ML stages and the API (see SPEC.md)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["noul", "choice", "score"]
Source = Literal["imported", "mined", "synthetic", "hard_negative", "user"]
Split = Literal["train", "val", "test"]
# Who gave a row its label. None on a synthetic row means the generator's intended label (kept only when
# the teacher relabel agreed). Real rows always carry one: the user, the teacher LLM, the labelling agent
# (teacher passes + adjudication, ``forge.labelling.agent_label``) or the export itself.
LabelSource = Literal["user", "teacher", "agent", "export"]
# A real row's disposition: still waiting, labelled, or dropped by the labelling agent (with a reason).
LabelState = Literal["needs_label", "labelled", "excluded_by_agent"]
LabelPolicy = Literal["agent", "user", "teacher", "mixed"]
# Example.provenance for a generated row shaped on scrubbed real exemplars (``forge.shaping``).
REAL_SHAPED = "real_shaped_synthetic"
# Sources whose state text is real data, not generated (curator.REAL_SOURCES mirrors this).
REAL_SOURCES: tuple[str, ...] = ("imported", "mined", "user")


class TaskSchema(BaseModel):
    """The single typed question the trained CLM head answers."""

    name: str = Field(description="slug, e.g. ticket_router; becomes package clm_<name>")
    question_id: str = Field(description="key in the System One questions map, e.g. department")
    type: QuestionType
    instructions: str = Field(description="the question text, appended after the state")
    criteria: dict[str, str] | list[str] | None = Field(
        default=None,
        description="choice: {key: description}; score: ordered levels low->high; noul: optional {true, false}",
    )
    state_description: str = Field(description="what the state text contains at the call site")
    state_example: str

    def question(self) -> dict:
        """System One wire-format question dict (clm_core.schema)."""
        q: dict = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            q["criteria"] = self.criteria
        return q

    def labels(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria or {})
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria or []))]
        return ["false", "true"]


class FileRef(BaseModel):
    path: str
    start_line: int | None = None
    end_line: int | None = None
    snippet: str | None = None


class Opportunity(BaseModel):
    id: str
    title: str
    files: list[FileRef] = []
    current_approach: str
    task: TaskSchema
    fit_score: float = Field(ge=0, le=1)
    rationale: str
    expected_benefits: list[str] = []
    risks: list[str] = []
    data_strategy: str
    target_examples: int = 600
    success_metric: Literal["accuracy", "macro_f1"] = "accuracy"
    success_threshold: float = 0.85
    integration_sketch: str


class Proposal(BaseModel):
    verdict: Literal["fit", "no_fit"]
    summary: str
    opportunities: list[Opportunity] = []
    recommended_id: str | None = None
    no_fit_reasons: list[str] = []
    revision: int = 0


class Example(BaseModel):
    id: str
    state: str
    label: str
    probabilities: dict[str, float] | None = None
    source: Source
    split: Split
    rationale: str | None = None
    meta: dict = {}
    label_source: LabelSource | None = None
    # None for ordinary rows; REAL_SHAPED for generated rows modelled on scrubbed real exemplars
    # (meta.exemplar_hashes, meta.batch say which). Always train-only.
    provenance: str | None = None

    @model_validator(mode="after")
    def _backfill_label_source(self) -> "Example":
        """Rows written before ``label_source`` existed: derive it from what they recorded. JSONL files are
        never rewritten for this; every read backfills."""
        if self.label_source is None:
            origin = self.meta.get("label_origin")
            if origin in ("teacher", "export"):
                self.label_source = origin
            elif self.source == "mined" and self.rationale == "label stated in source":
                self.label_source = "export"   # the mined file itself paired the text with this label
            elif self.source == "user" or self.meta.get("edited"):
                self.label_source = "user"
            elif self.source in REAL_SOURCES:
                self.label_source = "teacher"
        return self


RealRowSource = Literal["imported", "mined", "user", "export"]


class RealRow(BaseModel):
    """One real (not generated) row in a labelling queue (``forge.labelling``).

    ``label`` is None until someone labels it. ``label_state`` says whether the row may be trained on
    (``labelled``), still waits (``needs_label``) or was dropped by the labelling agent
    (``excluded_by_agent``, with ``exclusion_reason``); ``label_source`` who labelled it. The teacher's view
    of a row it may not label (an export or a user row, or under the ``mixed`` policy) lives in
    ``suggested_label`` / ``teacher_conf`` / ``teacher_votes``.

    The head-set adapter writes the same file; keys it adds that this model does not
    name are kept (``extra="allow"``), so a rewrite by the queue never drops them."""
    model_config = ConfigDict(extra="allow")
    rid: str
    state: str
    question_meta: dict = {}             # {"question": {...}} for heads whose options differ per row; provenance
    source: RealRowSource
    split: Split = "train"
    label: str | None = None
    label_state: LabelState = "needs_label"
    label_source: LabelSource | None = None
    suggested_label: str | None = None
    teacher_conf: float | None = None
    teacher_votes: list[dict] = []       # two-pass teacher: one {"pass", "label", "conf"} per pass
    head: str | None = None              # head id (head dirs) or task name (projects)
    skipped: bool = False                # the user passed over it for now; still pending
    excluded: bool = False               # the user chose to train without it; not pending until labelled
    adjudicated: bool = False            # the agent's label came from the adjudicator (the passes disagreed)
    exclusion_reason: str | None = None  # why the agent excluded it (label_state == excluded_by_agent)
    adjudication: dict | None = None     # the adjudicator's verdict {model, decision, reason} (agent_label.settle)
    updated_at: str = ""

    @property
    def pending(self) -> bool:
        """No final disposition yet: neither labelled, excluded by the agent, nor set aside by the user."""
        return self.label_state == "needs_label" and not self.excluded


class ModelScore(BaseModel):
    name: str  # finetuned | reference | majority | teacher
    accuracy: float
    macro_f1: float
    n: int
    note: str | None = None


class Confusion(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class ReliabilityBin(BaseModel):
    bin_lo: float
    bin_hi: float
    confidence: float
    accuracy: float
    count: int


class LabelScore(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class Latency(BaseModel):
    embed_ms_p50: float
    embed_ms_p95: float
    head_ms_p50: float
    head_ms_p95: float
    warm_cached_ms_p50: float
    n: int = 0                                   # timed calls (after the warm-up calls)
    encoder_peak_bytes: int | None = None        # embedding worker peak footprint (weights + activations) during evaluation


class ErrorCase(BaseModel):
    id: str
    state: str
    gold: str
    pred: str
    probabilities: dict[str, float]


class ValDiagnostics(BaseModel):
    """The fine-tuned head on the VAL split: what curation guidance may look at. Test rows are only scored
    for the gate and never shown to the reviewer that writes guidance for new training data."""
    n: int
    accuracy: float
    macro_f1: float
    confusion: Confusion
    errors: list[ErrorCase]


class EvalReport(BaseModel):
    metric: str
    threshold: float
    # why ``threshold`` differs from the project default, and who approved it (a per-head exception)
    threshold_note: str | None = None
    gate_passed: bool
    gate_reasons: list[str]
    models: list[ModelScore]
    confusion: Confusion
    reliability: list[ReliabilityBin]
    ece: float
    per_label: list[LabelScore]
    latency: Latency
    errors: list[ErrorCase]
    n_test: int
    val: ValDiagnostics | None = None
    # decisiveness: the share of test rows whose top probability reaches ``confident_at`` (a caller's
    # "decided" threshold) and the accuracy on those rows; gated when ``min_confident_share`` is set
    confident_at: float = 0.8
    confident_share: float | None = None
    confident_accuracy: float | None = None
    min_confident_share: float | None = None
    # "labels": gold/pred are option keys; "none_match": dynamic option sets, scored as none / match
    # (pred == the gold candidate) / wrong (another candidate), macro-F1 over none and match
    scoring: str = "labels"


class PackageFile(BaseModel):
    path: str
    bytes: int
    description: str


class PackageManifest(BaseModel):
    name: str
    version: str
    dir: str
    zip: str
    files: list[PackageFile]
    integration_md: str
    created_at: str
