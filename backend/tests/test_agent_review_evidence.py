"""The reviewer's curation guidance must never see frozen test rows (they would leak into train)."""
from forge.models import (Confusion, ErrorCase, EvalReport, LabelScore, Latency, ModelScore, ValDiagnostics)
from forge.orchestrator import _review_evidence


def _report(val: ValDiagnostics | None) -> EvalReport:
    err = lambda i, s: ErrorCase(id=i, state=s, gold="a", pred="b", probabilities={"a": 0.1, "b": 0.9})  # noqa: E731
    return EvalReport(
        metric="macro_f1", threshold=0.8, gate_passed=False, gate_reasons=[],
        models=[ModelScore(name="finetuned", accuracy=0.7, macro_f1=0.7, n=10)],
        confusion=Confusion(labels=["a", "b"], matrix=[[3, 2], [0, 5]]), reliability=[], ece=0.0,
        per_label=[LabelScore(label="a", precision=1, recall=0.6, f1=0.75, support=5)],
        latency=Latency(embed_ms_p50=1, embed_ms_p95=1, head_ms_p50=1, head_ms_p95=1, warm_cached_ms_p50=1),
        errors=[err("t1", "SECRET TEST ROW")], n_test=10, val=val)


def test_review_evidence_uses_val_rows_only():
    val = ValDiagnostics(n=20, accuracy=0.9, macro_f1=0.9, confusion=Confusion(labels=["a", "b"], matrix=[[9, 1], [1, 9]]),
                         errors=[ErrorCase(id="v1", state="a val row", gold="a", pred="b", probabilities={})])
    text = _review_evidence(_report(val))
    assert "SECRET TEST ROW" not in text
    assert "a val row" in text and "a→b 1" in text and "b→a 1" in text


def test_review_evidence_without_test_split_shows_its_own_errors():
    # no separate test split: the report was computed on val, so its errors are val errors
    assert "SECRET TEST ROW" in _review_evidence(_report(None))
