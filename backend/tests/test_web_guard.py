"""forge/web_guard.py: a localhost web app is reachable by every website the user opens and by every local process.
Requests must be addressed to Forge; changes need the Forge page's origin and this server's session; a user label
records the browser session it came from."""
import json

from fastapi.testclient import TestClient

import forge_client
from test_agent_fakes import FakeLLM, TASK, isolated  # noqa: F401  (fixture)


def _app():
    from forge import config
    from forge.api import create_app
    from forge.orchestrator import Orchestrator
    from forge.store import Store
    orch = Orchestrator(store=Store(config.DB_PATH))
    orch._llm_override = FakeLLM()
    return create_app(orch)


def _head(tmp_path, monkeypatch):
    from forge.labelling import LabelQueue, head_task
    from forge.models import RealRow
    d = tmp_path / "heads" / "task-x"
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"head_id": "task-x", **TASK.model_dump()}))
    LabelQueue(d, head_task(d)).add([RealRow(rid="q1", state="billing issue", source="export", head="task-x")])
    monkeypatch.setenv("FORGE_LABELSETS", str(tmp_path / "heads"))
    return d


def test_changes_need_the_forge_origin_and_this_servers_session(isolated, tmp_path, monkeypatch):  # noqa: F811
    d = _head(tmp_path, monkeypatch)
    app = _app()
    g = app.state.web_guard
    url = "/api/labelsets/h:task-x/rows/q1"
    with TestClient(app, base_url=g.base_url()) as raw:
        # reads from the Forge host are fine; a request addressed to another host (DNS rebinding) is not
        assert raw.get("/api/labelsets").status_code == 200
        r = raw.get("/api/labelsets", headers={"Sec-Fetch-Site": "cross-site"})            # another site's page
        assert r.status_code == 403 and "cross-site" in r.json()["detail"]
        r = raw.get("/api/labelsets", headers={"Host": "evil.example:8765"})
        assert r.status_code == 403 and "addressed to evil.example:8765" in r.json()["detail"]
        # an automated client posting a "user" label without a browser session is refused, in plain words
        r = raw.post(url, json={"label": "account"})
        assert r.status_code == 403 and "no Origin header" in r.json()["detail"]
        r = raw.post(url, json={"label": "account"}, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403 and "came from https://evil.example" in r.json()["detail"]
        origin = {"Origin": "http://127.0.0.1:8765"}
        r = raw.post(url, json={"label": "account"}, headers=origin)
        assert r.status_code == 403 and "no Forge session" in r.json()["detail"]
        r = raw.post(url, json={"label": "account"}, headers={**origin, "X-Forge-Session": "guess"})
        assert r.status_code == 403 and "not the running server's" in r.json()["detail"]
        r = raw.post(url, json={"label": "account"},
                     headers={**origin, "X-Forge-Session": g.token, "Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403 and "cross-site" in r.json()["detail"]
        # the session comes from a same-origin bootstrap; a page the browser marks as another site's gets nothing
        assert raw.get("/api/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        assert raw.get("/api/session", headers={"Host": "evil.example:8765"}).status_code == 403
        s = raw.get("/api/session", headers={"Sec-Fetch-Site": "same-origin"})
        assert s.status_code == 200 and s.json() == {"token": g.token, "header": "X-Forge-Session"}
        assert s.headers["cache-control"] == "no-store"
        log = (d / "labelling_log.jsonl")
        assert not log.read_text().count('"actor": "user"')                 # nothing above reached the queue
    with forge_client.client(app, headers={**g.client_headers(), "User-Agent": "Mozilla/5.0 test"}) as c:
        assert c.post(url, json={"label": "account"}).status_code == 200
    entry = [json.loads(x) for x in log.read_text().splitlines()][-1]
    assert entry["actor"] == "user" and entry["via"] == {"origin": "http://127.0.0.1:8765",
                                                         "user_agent": "Mozilla/5.0 test"}


def test_the_ui_dev_server_proxy_is_the_forge_page_too(isolated):  # noqa: F811
    from forge.web_guard import SessionGuard
    g = SessionGuard(port=8765, ui_port=5173)
    assert {"127.0.0.1:5173", "localhost:5173", "127.0.0.1:8765", "localhost:8765"} == g.hosts
    assert "http://localhost:5173" in g.origins and "http://evil.example" not in g.origins


def test_no_other_page_may_frame_forge(isolated):  # noqa: F811
    """Clickjacking: a framed Forge page's own scripts hold the session, so framing itself must be refused."""
    app = _app()
    g = app.state.web_guard
    with TestClient(app, base_url=g.base_url()) as raw:
        for r in (raw.get("/api/labelsets"), raw.get("/"), raw.get("/api/session"),
                  raw.get("/api/labelsets", headers={"Host": "evil.example:8765"})):     # refusals too
            assert r.headers["x-frame-options"] == "DENY"
            assert r.headers["content-security-policy"] == "frame-ancestors 'none'"
        r = raw.get("/", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "iframe"})
        assert r.status_code == 403 and "cannot be shown inside another page" in r.json()["detail"]
        assert r.headers["x-frame-options"] == "DENY"
        r = raw.get("/", headers={"Sec-Fetch-Site": "same-site", "Sec-Fetch-Dest": "frame"})
        assert r.status_code == 403
        # opening Forge directly (a top-level navigation) is not a frame
        r = raw.get("/", headers={"Sec-Fetch-Site": "none", "Sec-Fetch-Dest": "document"})
        assert r.status_code != 403


def test_no_cors_grant_and_the_dev_proxy_still_works(isolated, tmp_path, monkeypatch):  # noqa: F811
    """No page of another origin may read Forge's responses (no Access-Control-Allow-* on any answer, not even for
    the Vite dev origin), while the UI dev server, which proxies /api same-origin, reads and changes as the page."""
    _head(tmp_path, monkeypatch)
    app = _app()
    g = app.state.web_guard
    dev = "http://127.0.0.1:5173"
    with TestClient(app, base_url=g.base_url()) as raw:
        for origin in (dev, "https://evil.example"):
            r = raw.get("/api/labelsets", headers={"Origin": origin})
            assert not any(h.lower().startswith("access-control-") for h in r.headers)
            pre = raw.options("/api/labelsets/h:task-x/rows/q1", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-forge-session,content-type"})
            assert "access-control-allow-origin" not in pre.headers
        # the Vite proxy (changeOrigin: false) forwards Host and Origin of the :5173 page, same-origin
        page = {"Host": "127.0.0.1:5173", "Origin": dev, "Sec-Fetch-Site": "same-origin"}
        s = raw.get("/api/session", headers=page)
        assert s.status_code == 200
        r = raw.post("/api/labelsets/h:task-x/rows/q1", json={"label": "account"},
                     headers={**page, "X-Forge-Session": s.json()["token"]})
        assert r.status_code == 200, r.text
