"""Base-model selection with fake encoders/trainer: winner by val accuracy, failures skipped, cancel."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from forge import base_models
from forge.base_models import select_base_model
from forge.models import Example

from test_agent_fakes import TASK


class FakeEncoder:
    def __init__(self, spec, fail=False, dim=8):
        self.spec, self.fail, self.dim, self.unloaded = spec, fail, dim, False

    def load(self, on_status=None, should_stop=None):
        if self.fail:
            raise RuntimeError("llama-server not found")

    def embed(self, texts, use_cache=True, on_progress=None, should_stop=None):
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def unload(self):
        self.unloaded = True


ROWS = [Example(id=f"e{i}", state=f"state {i}", label=["billing", "technical", "account"][i % 3], source="synthetic",
                split=["train", "val", "test"][i % 3]) for i in range(30)]


def setup(monkeypatch, accs, fail=()):
    encs = {s: FakeEncoder(s, fail=s in fail) for s in accs}
    monkeypatch.setattr(base_models, "get_encoder", lambda spec=None: encs[spec])
    calls = []

    def fake_train(task, rows, out_dir, cfg, emit, should_stop):
        calls.append((cfg.encoder, {r.split for r in rows}))
        emit({"type": "train_epoch", "epoch": 1})            # must not be forwarded
        emit({"type": "log", "stage": "training", "level": "info", "message": "hello"})
        acc = accs[cfg.encoder]
        return SimpleNamespace(best_val_acc=acc, init_val_acc=0.3, best_epoch=1, cancelled=False,
                               history=[{"val_soft_ce": 1.0}, {"val_soft_ce": 0.5}])
    return encs, calls, fake_train


def test_picks_highest_val_accuracy_and_unloads_losers(tmp_path, monkeypatch):
    accs = {"mlx:qwen": 0.81, "llamacpp:winnow": 0.88}
    encs, calls, fake_train = setup(monkeypatch, accs)
    events = []
    out = select_base_model(TASK, ROWS, tmp_path, list(accs), events.append, train_fn=fake_train)
    assert out["chosen"] == "llamacpp:winnow"
    assert [c for c, _ in calls] == ["mlx:qwen", "llamacpp:winnow"]
    assert encs["mlx:qwen"].unloaded and not encs["llamacpp:winnow"].unloaded
    assert json.loads((tmp_path / "base_model.json").read_text())["chosen"] == "llamacpp:winnow"
    assert not any(e["type"] == "train_epoch" for e in events)
    assert any(e["type"] == "log" and e["stage"] == "selecting" and "hello" in e["message"] for e in events)
    last = [e for e in events if e["type"] == "base_model"][-1]
    assert last["chosen"] == "llamacpp:winnow" and [c["chosen"] for c in last["candidates"]] == [False, True]


def test_failed_candidate_is_skipped(tmp_path, monkeypatch):
    accs = {"mlx:qwen": 0.81, "llamacpp:winnow": 0.99}
    _, _, fake_train = setup(monkeypatch, accs, fail={"llamacpp:winnow"})
    out = select_base_model(TASK, ROWS, tmp_path, list(accs), lambda e: None, train_fn=fake_train)
    assert out["chosen"] == "mlx:qwen"
    w = next(c for c in out["candidates"] if c["spec"] == "llamacpp:winnow")
    assert w["status"] == "failed" and "llama-server" in w["error"]


def test_all_failed_raises(tmp_path, monkeypatch):
    accs = {"mlx:qwen": 0.8, "llamacpp:winnow": 0.9}
    _, _, fake_train = setup(monkeypatch, accs, fail=set(accs))
    with pytest.raises(RuntimeError, match="every base model failed"):
        select_base_model(TASK, ROWS, tmp_path, list(accs), lambda e: None, train_fn=fake_train)


def test_single_candidate_skips_comparison(tmp_path, monkeypatch):
    _, calls, fake_train = setup(monkeypatch, {"mlx:qwen": 0.8})
    out = select_base_model(TASK, ROWS, tmp_path, ["mlx:qwen"], lambda e: None, train_fn=fake_train)
    assert out["chosen"] == "mlx:qwen" and calls == []



def test_qwen3_4b_specs_are_known():
    for spec, name in [(base_models.QWEN4_MLX, "Qwen3-4B (MLX 4-bit)"), (base_models.QWEN4_GGUF, "Qwen3-4B (GGUF Q4_K_M)")]:
        d = base_models.describe(spec)
        assert d["spec"] == spec and d["name"] == name
        assert "2560-d" in d["detail"] and "from scratch" in d["detail"]
        assert d["url"].startswith("https://huggingface.co/")
    assert base_models.QWEN4_GGUF == "llamacpp:Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"
    assert base_models.QWEN4_MLX == "mlx:mlx-community/Qwen3-4B-4bit"
    assert not hasattr(base_models, "DEFAULT_CANDIDATES")      # no app-wide candidates: each experiment has its own


def test_selection_needs_the_experiment_encoder(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="needs the experiment's encoder"):
        select_base_model(None, [], tmp_path, [], lambda ev: None)
