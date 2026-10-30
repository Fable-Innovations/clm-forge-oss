"""Which requests may reach and change Forge: a localhost web app is reachable by every website the user opens
(cross-site requests, DNS rebinding) and by every local process.

One owner for the request checks ``forge.api`` installs as middleware:

* **Host**: every request must be addressed to Forge itself (``127.0.0.1:<port>`` / ``localhost:<port>``, or the
  UI dev server's ``:<ui port>`` that proxies ``/api``). A DNS-rebinding page names its own host and is refused,
  reads included.
* **Another site's page**: an ``/api/`` request the browser marks ``Sec-Fetch-Site`` other than ``same-origin`` /
  ``none`` is refused, reads included (some reads start work, e.g. a preset's ``inspect``).
* **Mutating requests** (POST, PUT, PATCH, DELETE) additionally need
  - an ``Origin`` that is the Forge UI (the same hosts, ``http://``); a browser always sends it, so a page on any
    other site is refused, and so is a client that sends none;
  - the per-server session token in ``X-Forge-Session``. The token is random per server start and handed out only
    by ``GET /api/session`` (same Host rules; a cross-origin page cannot read the response: no CORS grant), which
    the UI calls once. It is never written to disk or logged.
* **Framing**: every response carries ``X-Frame-Options: DENY`` and ``Content-Security-Policy: frame-ancestors
  'none'``, and a page request (not ``/api/``) the browser marks as loaded into a frame (``Sec-Fetch-Dest``
  ``iframe`` / ``frame``) by anything but the Forge page itself is refused. Without this another site could frame
  the Forge page, whose own scripts hold the session and send same-origin changes, and trick the user into
  clicking Run, label, promote or delete (clickjacking).
* ``browser_origin(request)``: the UI origin and user agent of an allowed mutating request, which the labelling
  queue records with every ``actor=user`` entry (``labelling_log.jsonl`` ``via``), so a label "by the user" says
  where it came from.

Every refusal is a 403 whose ``detail`` says in plain words what was missing. The server binds 127.0.0.1 only
(``forge.api.main``).
"""
from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse

MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
HEADER = "X-Forge-Session"
UI_DEV_PORT = int(os.environ.get("FORGE_UI_PORT", "5173"))
NO_FRAMING = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'"}
FRAME_DESTS = {"iframe", "frame", "embed", "object"}


@dataclass
class SessionGuard:
    port: int
    ui_port: int = UI_DEV_PORT
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))

    @property
    def hosts(self) -> set[str]:
        return {f"{h}:{p}" for h in ("127.0.0.1", "localhost") for p in (self.port, self.ui_port)}

    @property
    def origins(self) -> set[str]:
        return {f"http://{h}" for h in self.hosts}

    def refuse(self, request: Request) -> str | None:
        """Why ``request`` is refused, in plain words, or None."""
        host = (request.headers.get("host") or "").lower()
        if host not in self.hosts:
            return (f"Forge only answers requests addressed to it (127.0.0.1:{self.port} or localhost:{self.port}); "
                    f"this one was addressed to {host or 'no host'}")
        site = request.headers.get("sec-fetch-site")
        if site and site not in ("same-origin", "none") and request.url.path.startswith("/api/"):
            # another site's page (a fetch, an <img>, a form): not even a read, some reads start work (inspect)
            return f"the browser reports this request as {site}, not from the Forge page; refused"
        dest = (request.headers.get("sec-fetch-dest") or "").lower()
        if dest in FRAME_DESTS and site != "same-origin" and not request.url.path.startswith("/api/"):
            return (f"the Forge page cannot be shown inside another page (this request is for an {dest} on a "
                    f"{site or 'unknown'} page); open Forge directly")
        if request.method not in MUTATING:
            return None
        origin = request.headers.get("origin")
        if not origin:
            return ("changes to Forge are accepted only from the Forge page in a browser; this request did not say "
                    "where it came from (no Origin header)")
        if origin.lower() not in self.origins:
            return f"changes to Forge are accepted only from the Forge page; this request came from {origin}"
        given = request.headers.get(HEADER) or ""
        if not given:
            return "this request carries no Forge session; reload the Forge page and try again"
        if not hmac.compare_digest(given, self.token):
            return "this request's Forge session is not the running server's (the server restarted?); reload the page"
        return None

    def session_refusal(self, request: Request) -> str | None:
        """``GET /api/session``: the Host rule, and never to a page the browser says is another site's."""
        if why := self.refuse(request):
            return why
        site = request.headers.get("sec-fetch-site")
        if site and site not in ("same-origin", "none"):
            return f"the browser reports this request as {site}; the Forge session is only for the Forge page"
        return None

    def client_headers(self) -> dict[str, str]:
        """What an allowed in-process client sends (tests' ``TestClient``)."""
        return {"Origin": f"http://127.0.0.1:{self.port}", HEADER: self.token}

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def install(app, guard: SessionGuard) -> None:
    app.state.web_guard = guard

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        if request.url.path == "/api/session":
            response = await call_next(request)  # the route applies session_refusal itself
        elif why := guard.refuse(request):
            response = JSONResponse({"detail": why}, status_code=403)
        else:
            response = await call_next(request)
        response.headers.update(NO_FRAMING)      # no other page may frame Forge (clickjacking)
        return response

    @app.get("/api/session")
    async def session(request: Request):
        why = guard.session_refusal(request)
        if why:
            return JSONResponse({"detail": why}, status_code=403)
        return JSONResponse({"token": guard.token, "header": HEADER}, headers={"Cache-Control": "no-store"})


def browser_origin(request: Request) -> dict:
    """``{origin, user_agent}`` of an allowed mutating request (recorded with user actions)."""
    return {"origin": request.headers.get("origin"), "user_agent": (request.headers.get("user-agent") or "")[:200]}
