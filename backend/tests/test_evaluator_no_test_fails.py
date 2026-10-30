"""The gate is scored on test rows only: no test split is an error, never a silent switch to val."""
from pathlib import Path

import pytest

from forge.evaluator import evaluate
from forge.models import Example

from test_agent_fakes import TASK


def test_evaluate_without_test_split_raises(tmp_path):
    rows = [Example(id=f"{s}{i}", state=f"{lab} {i}", label=lab, source="synthetic", split=s)
            for i, lab in enumerate(TASK.labels()) for s in ("train", "val")]
    with pytest.raises(ValueError, match="no test split"):
        evaluate(TASK, rows, Path(tmp_path / "missing.pt"), tmp_path, lambda e: None)
