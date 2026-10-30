"""Shared pytest wiring for the backend suite.

``model_loading``: a test module that loads a real encoder (``tests/test_ml_e2e.py``, ``tests/test_encoder_worker.py``)
marks itself ``pytestmark = pytest.mark.model_loading``. When Forge's own memory admission refuses the model
(``forge.memory.MemoryAdmissionError``: the model can never fit the configured budget, or the machine did not free
enough memory within the admission wait, e.g. because another model is resident), such a test is **skipped** with the
refusal as the reason, not failed: the machine could not host the model then, and the test says exactly why. A test
that expects the refusal (``pytest.raises(MemoryAdmissionError)``) still sees it; any other error still fails.
"""
from __future__ import annotations

import os
import tempfile

import pytest

# Before any ``forge`` import (``forge.paths`` reads it once): the suite writes its output root, caches included, to a
# throwaway folder, never the repo's ``output/``.
os.environ["FORGE_DATA"] = tempfile.mkdtemp(prefix="forge-tests-")


@pytest.fixture(autouse=True)
def _no_headset_presets(tmp_path_factory, monkeypatch):
    """No test reads this machine's head-set presets file: the default is a file that does not exist (no presets);
    a test that needs presets writes its own (``headset_presets.use``)."""
    monkeypatch.setenv("FORGE_HEADSET_PRESETS", str(tmp_path_factory.getbasetemp() / "no-headset-presets.json"))


def pytest_configure(config):
    config.addinivalue_line("markers", "model_loading: loads a real encoder; skipped when memory admission refuses it")


def _admission_refusal(exc: BaseException | None) -> BaseException | None:
    from forge.memory import MemoryAdmissionError
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, MemoryAdmissionError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    try:
        return (yield)
    except Exception as e:
        refusal = _admission_refusal(e) if pyfuncitem.get_closest_marker("model_loading") else None
        if refusal is None:
            raise
        pytest.skip(f"model memory refused on this machine: {refusal}")
