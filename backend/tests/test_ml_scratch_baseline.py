"""Non-4096-d base models (Qwen3-4B is 2560-d): the evaluator scores an init head as the reference
baseline and never touches the 8B reference head/encoder; the trainer uses the longer scratch schedule;
export_head.py round-trips a head to head.bin/head.json. Tiny fake encoder, real torch heads."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge import evaluator, inference  # noqa: E402
from forge.clm_core.heads import make_head  # noqa: E402
from forge.models import Example  # noqa: E402
from forge.trainer import (EPOCHS_SCRATCH, EPOCHS_WARM, PATIENCE_SCRATCH, PATIENCE_WARM,  # noqa: E402
                           TrainConfig, resolve_schedule)

from test_agent_fakes import TASK  # noqa: E402

DIM = 24
SPEC = "llamacpp:Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"


class TinyEncoder:
    """Deterministic hash embeddings, L2-normalised, ``DIM``-d."""
    backend, model_name, max_len = "fake", "tiny", 2048

    def __init__(self, spec=SPEC):
        self.spec, self.dim, self.loads = spec, DIM, 0

    def load(self, on_status=None, should_stop=None):
        self.loads += 1

    def embed(self, texts, use_cache=True, on_progress=None, on_status=None, should_stop=None):
        out = np.stack([np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).standard_normal(DIM)
                        for t in texts]).astype(np.float32) if texts else np.zeros((0, DIM), np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def info(self):
        return {"worker": {}}


def _ckpt(path: Path, hidden=DIM, seed=7) -> Path:
    torch.manual_seed(seed)
    arch = dict(width=16, depth=3, activation="gelu", layernorm=True, residual=False)
    sh = make_head(arch["width"], arch["depth"], 8, arch["activation"], arch["layernorm"], arch["residual"], hidden=hidden)
    ah = make_head(arch["width"], arch["depth"], 8, arch["activation"], arch["layernorm"], arch["residual"], hidden=hidden)
    # plain dicts, as the packager writes head.pt (an OrderedDict's _metadata would trip the restricted unpickler)
    torch.save({"state_head": dict(sh.state_dict()), "action_head": dict(ah.state_dict()),
                "logit_scale": torch.tensor(math.log(1 / 0.07)),
                "cfg": {**arch, "projection_dim": 8, "hidden_size": hidden, "embed_model": SPEC,
                        "labels": list(TASK.criteria), "task_schema": TASK.model_dump(), "max_len": 2048}}, path)
    return path


@pytest.fixture
def tiny(monkeypatch):
    enc = TinyEncoder()
    monkeypatch.setattr(inference, "get_encoder", lambda spec=None: enc)

    def no_download(*a, **k):
        raise AssertionError("the 8B reference head must not be fetched for a non-4096-d encoder")
    monkeypatch.setattr(inference, "download", no_download)
    return enc


def test_evaluate_uses_init_head_reference_without_8b(tiny, tmp_path):
    ck = _ckpt(tmp_path / "best_head.pt")
    labels = list(TASK.criteria)
    rows = [Example(id=f"r{i}", state=f"ticket {i}", label=labels[i % 3], source="synthetic",
                    split=["train", "val", "test", "test"][i % 4]) for i in range(40)]
    events = []
    rep = evaluator.evaluate(TASK, rows, ck, tmp_path / "eval", events.append, metric="macro_f1", threshold=0.1)
    ref = next(m for m in rep.models if m.name == "reference")
    assert "zero-shot init head on " + SPEC in ref.note and "24-d" in ref.note
    assert (tmp_path / "eval" / "reference_init_head.pt").exists()
    assert any("reference baseline: zero-shot init head" in e.get("message", "") for e in events if e["type"] == "log")
    assert rep.metric == "macro_f1" and len(rep.gate_reasons) == 3


def test_reference_predictor_is_deterministic_and_shares_encoder(tiny, tmp_path):
    ft = inference.Predictor(_ckpt(tmp_path / "ft.pt"), TASK)
    r1, _ = evaluator.reference_predictor(ft, TASK, tmp_path / "a")
    r2, _ = evaluator.reference_predictor(ft, TASK, tmp_path / "b")
    assert r1.encoder is ft.encoder and r1.name == "reference"
    np.testing.assert_allclose(r1.probs(["x", "y"]), r2.probs(["x", "y"]), atol=1e-6)


def test_reference_predictor_keeps_8b_head_for_4096(monkeypatch, tmp_path):
    seen = []

    class FakePred:
        def __init__(self, ckpt, task, *a, **k):
            seen.append(ckpt)
            self.head = type("H", (), {"cfg": {"hidden_size": 4096}})()
    monkeypatch.setattr(evaluator, "Predictor", FakePred)
    _, note = evaluator.reference_predictor(FakePred("ft", TASK), TASK, tmp_path)
    assert seen == ["ft", None] and note == evaluator.REF_NOTE_8B


def test_scratch_schedule():
    assert (resolve_schedule(TrainConfig(), scratch=True).epochs, resolve_schedule(TrainConfig(), scratch=True).patience) \
        == (EPOCHS_SCRATCH, PATIENCE_SCRATCH) == (40, 8)
    assert (resolve_schedule(TrainConfig(), scratch=False).epochs, resolve_schedule(TrainConfig(), scratch=False).patience) \
        == (EPOCHS_WARM, PATIENCE_WARM) == (30, 6)
    c = resolve_schedule(TrainConfig(epochs=5, patience=2), scratch=True)
    assert (c.epochs, c.patience) == (5, 2)


def _export_mod():
    p = Path(__file__).resolve().parents[1] / "scripts" / "export_head.py"
    spec = importlib.util.spec_from_file_location("export_head", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_export_head_round_trip(tmp_path):
    ex = _export_mod()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _ckpt(pkg / "head.pt", hidden=DIM)
    opts = [{"key": k, "text": v} for k, v in TASK.criteria.items()]
    (pkg / "task.json").write_text(json.dumps({"task": TASK.model_dump(), "options": opts,
                                               "encoder": {"backend": "llamacpp", "model": "Qwen/Qwen3-4B-GGUF",
                                                           "revision": "abc123"}}))
    oe = TinyEncoder().embed([o["text"] for o in opts])
    np.save(pkg / "options.npy", oe)
    out = tmp_path / "out"
    meta = ex.export(pkg, out)
    assert meta["format"] == "clm-head-v1" and meta["hidden_size"] == DIM and meta["projection_dim"] == 8
    assert meta["encoder"]["revision"] == "abc123" and [o["key"] for o in meta["options"]] == list(TASK.criteria)
    assert {"head.bin", "options.bin", "options_proj.bin"} == set(meta["files"])
    assert (out / "head.bin").stat().st_size == sum(l["count"] for l in meta["layers"]) * 4
    assert ex.verify(pkg, out, n=20) < 1e-5
    # the exported forward matches the torch heads too
    ck = torch.load(pkg / "head.pt", weights_only=True)
    sh = make_head(16, 3, 8, "gelu", True, False, hidden=DIM)
    sh.load_state_dict(ck["state_head"]); sh.eval()
    x = np.random.default_rng(1).standard_normal((20, DIM)).astype(np.float32)
    with torch.no_grad():
        want = torch.nn.functional.normalize(sh(torch.from_numpy(x)), dim=-1).numpy()
    m, t = ex.load_exported(out)
    np.testing.assert_allclose(ex.forward(m, t, "state", x), want, atol=1e-5)
