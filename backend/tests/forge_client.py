"""A ``TestClient`` that talks to Forge the way the Forge page does (``forge.web_guard``): addressed to
127.0.0.1:<port>, from the Forge origin, with the server's session."""
from __future__ import annotations

from fastapi.testclient import TestClient


def client(app, headers: dict | None = None, **kw) -> TestClient:
    g = app.state.web_guard
    return TestClient(app, base_url=g.base_url(), headers={**g.client_headers(), **(headers or {})}, **kw)
