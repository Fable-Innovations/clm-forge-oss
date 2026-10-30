"""Provider interface shared by Claude (Agent SDK, subscription) and Ollama."""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
OnEvent = Callable[[dict], Any]          # receives {"type": "tool"|"log"|..., ...} (unstamped)
OnDelta = Callable[[str], Any]


class LLM(Protocol):
    provider: str
    model: str

    async def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T: ...

    async def chat(self, messages: list[dict], system: str | None = None,
                   on_delta: OnDelta | None = None) -> str: ...

    async def explore(self, task: str, repo_path: str | None, schema: type[T], on_event: OnEvent,
                      system: str | None = None) -> T: ...


class LLMError(RuntimeError):
    pass


def short(x: Any, n: int = 240) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def schema_for(model: type[BaseModel]) -> dict:
    return model.model_json_schema()


# ---- opt-in tracing (FORGE_MLFLOW=1) -----------------------------------------------------------
_mlflow = None


def tracing_enabled() -> bool:
    global _mlflow
    if os.environ.get("FORGE_MLFLOW") != "1":
        return False
    if _mlflow is None:
        try:
            import mlflow  # type: ignore

            from forge import config
            mlflow.set_tracking_uri(f"sqlite:///{config.DATA_DIR / 'mlflow.db'}")
            # artifacts beside the database, not in ./mlruns of whatever folder the server started in
            if mlflow.get_experiment_by_name("clm-forge") is None:
                mlflow.create_experiment("clm-forge", artifact_location=(config.DATA_DIR / "mlruns").as_uri())
            mlflow.set_experiment("clm-forge")
            _mlflow = mlflow
        except Exception:  # noqa: BLE001
            _mlflow = False
    return bool(_mlflow)


@contextlib.contextmanager
def span(name: str, inputs: dict | None = None):
    """An MLflow span when tracing is on, otherwise nothing. Yields a setter for outputs."""
    if not tracing_enabled():
        yield lambda _out: None
        return
    with _mlflow.start_span(name=name) as sp:  # type: ignore[union-attr]
        if inputs:
            sp.set_inputs({k: short(v, 4000) for k, v in inputs.items()})
        yield lambda out: sp.set_outputs({"output": short(out, 4000)})


def traced(kind: str):
    def deco(fn: Callable[..., Awaitable]):
        async def wrapper(self, *a, **kw):
            with span(f"{self.provider}.{kind}", {"args": a[:1], "model": self.model}) as set_out:
                out = await fn(self, *a, **kw)
                set_out(out.model_dump() if isinstance(out, BaseModel) else out)
                return out
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco
