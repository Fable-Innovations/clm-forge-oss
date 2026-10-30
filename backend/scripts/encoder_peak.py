"""Measure an encoder's peak memory against the configured budget (Settings.memory).

Embeds N synthetic texts spread over every padding bucket (uncached, so the model really runs)
and prints the worker's peak (per-batch estimate and the parent watchdog's polled footprint), the
budget, and the batch back-offs. Exit code 1 if the peak exceeded the budget.

    cd backend && uv run python scripts/encoder_peak.py mlx:mlx-community/Qwen3-8B-4bit --n 512
    cd backend && uv run python scripts/encoder_peak.py llamacpp:EldanRing/Winnow-12B:gguf/Winnow-12B-Q8_0.gguf
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from forge.encoder import DEFAULT_MAX_LEN, Encoder
from forge.memory import GiB, available_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    a = ap.parse_args()
    words = ("invoice refund crash login password latency export webhook seat billing outage token "
             "schedule deploy rollback migration cluster quota").split()
    lengths = [4, 24, 60, 120, 250, 500, 1000, 1900]       # one bucket each, up to max_len
    texts = [f"case {i}: " + " ".join(words[(i + j) % len(words)] for j in range(lengths[i % len(lengths)]))
             for i in range(a.n)]
    enc = Encoder(a.spec, a.max_len, cache_dir=Path(tempfile.mkdtemp()))
    pol = enc.policy()
    print(f"budget {pol.budget_bytes / GiB:.1f} GB, reserve {pol.reserve_bytes / GiB:.1f} GB, "
          f"free now {available_bytes() / GiB:.1f} GB")
    t0 = time.perf_counter()
    try:
        E = enc.embed(texts, use_cache=False, on_status=lambda lvl, m: print(f"[{lvl}] {m}"))
        watch = enc._client.watchdog.peak_bytes
    finally:
        enc.unload()
    peak = max(enc.last_peak_bytes or 0, watch)
    print(f"{a.spec}: {E.shape} in {time.perf_counter() - t0:.0f}s | peak {peak / GiB:.2f} GB "
          f"(worker {enc.last_peak_bytes / GiB:.2f}, watchdog {watch / GiB:.2f}) | budget {pol.budget_bytes / GiB:.2f} GB")
    return 0 if peak <= pol.budget_bytes else 1


if __name__ == "__main__":
    sys.exit(main())
