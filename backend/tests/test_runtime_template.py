"""The packaged runtime (``forge/runtime_template/pkg``) must use the encoder the head was trained on.

Fast and model-free: the encoder is never loaded. Regression for project 6f15ea9a9d, whose head was
trained on Qwen3-4B (2560-d) while the runtime defaulted to Qwen3-8B (4096-d) and crashed in matmul.
Run: ``cd backend && uv run pytest tests/test_runtime_template.py -q``
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from forge import packager

enc_mod = packager._load_template_module("encoders")
head_mod = packager._load_template_head_module()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("CLM_ENCODER", "CLM_MLX_MODEL", "CLM_MLX_CACHE_LIMIT_MB", "CLM_EMBED_URL", "CLM_EMBED_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_mlx_encoder_defaults_to_the_packaged_model_and_revision():
    e = enc_mod.encoder_from_env(2048, "mlx", model="mlx-community/Qwen3-4B-4bit", revision="abc123")
    assert (e.model_name, e.revision, e.cache_limit_mb) == ("mlx-community/Qwen3-4B-4bit", "abc123", 256)


def test_mlx_override_drops_the_packaged_revision(monkeypatch):
    monkeypatch.setenv("CLM_MLX_MODEL", "/models/local-copy")
    monkeypatch.setenv("CLM_MLX_CACHE_LIMIT_MB", "128")
    e = enc_mod.encoder_from_env(2048, "mlx", model="mlx-community/Qwen3-4B-4bit", revision="abc123")
    assert (e.model_name, e.revision, e.cache_limit_mb) == ("/models/local-copy", None, 128)


def test_no_model_and_bad_cache_limit_fail_fast(monkeypatch):
    with pytest.raises(RuntimeError, match="encoder.model"):
        enc_mod.encoder_from_env(2048, "mlx")
    monkeypatch.setenv("CLM_MLX_CACHE_LIMIT_MB", "lots")
    with pytest.raises(RuntimeError, match="CLM_MLX_CACHE_LIMIT_MB"):
        enc_mod.encoder_from_env(2048, "mlx", model="m")


def test_openai_needs_the_served_model_name(monkeypatch):
    monkeypatch.setenv("CLM_ENCODER", "openai")
    monkeypatch.setenv("CLM_EMBED_URL", "http://h/v1/embeddings")
    with pytest.raises(RuntimeError, match="CLM_EMBED_MODEL"):
        enc_mod.encoder_from_env(2048, "mlx", model="m")


def test_head_rejects_embeddings_of_the_wrong_width():
    head = types.SimpleNamespace(in_dim=2560)
    x = np.zeros((1, 4096), dtype=np.float32)
    with pytest.raises(ValueError, match="4096-d .* 2560-d"):
        head_mod.HeadPair._check(head, x)
    assert head_mod.HeadPair._check(head, np.zeros((1, 2560))).dtype == np.float32


def test_encoder_facts_come_from_the_encoder(monkeypatch):
    monkeypatch.setattr(packager, "_mlx_snapshot", lambda repo: ("4dcb3d1", 2_260_000_000))
    enc = types.SimpleNamespace(backend="mlx", model_name="mlx-community/Qwen3-4B-4bit",
                                spec="mlx:mlx-community/Qwen3-4B-4bit", max_len=2048)
    meta = packager.encoder_facts(enc, 2560)
    assert meta["model"] == "mlx-community/Qwen3-4B-4bit" and meta["revision"] == "4dcb3d1"
    assert meta["dim"] == 2560 and meta["base_model"] == "Qwen3-4B (MLX 4-bit)"   # KNOWN display name
    assert meta["mlx_cache_limit_mb"] == enc_mod.DEFAULT_MLX_CACHE_LIMIT_MB
    assert "8B" not in meta["recipe"]
