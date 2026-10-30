"""Codex provider: confinement flags, env scrubbing, JSONL event parsing (against a fake `codex`
binary), structured re-ask, and the repo MCP server it reads code through."""
import json
import os
import stat
import sys

import pytest
from pydantic import BaseModel


class Finding(BaseModel):
    summary: str
    files: list[str]


FAKE = r'''#!{py}
import json, os, sys
if "--version" in sys.argv:
    print("codex-cli 9.9.9-fake"); sys.exit(0)
if "features" in sys.argv:
    with open(os.environ["FAKE_CODEX_LOG"] + ".features", "w") as f:
        f.write(json.dumps({{"argv": sys.argv[1:], "codex_home": os.environ.get("CODEX_HOME"),
                             "home_files": os.listdir(os.environ.get("CODEX_HOME") or ".")}}))
    print(os.environ.get("FAKE_FEATURES", "")); sys.exit(int(os.environ.get("FAKE_FEATURES_RC", "0")))
log = os.environ["FAKE_CODEX_LOG"]
prompt = sys.stdin.read()
with open(log, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "prompt": prompt,
                         "env_has_key": "OPENAI_API_KEY" in os.environ}}) + "\n")
n = sum(1 for _ in open(log))
reply = os.environ.get("FAKE_REPLY_%d" % n) or os.environ["FAKE_REPLY"]
extra = json.loads(os.environ.get("FAKE_EXTRA_EVENT", "null"))
for ev in ([extra] if extra else []) + [
    {{"type": "thread.started"}},
    {{"type": "item.completed", "item": {{"type": "agent_message", "text": "Looking at the router."}}}},
    {{"type": "item.completed", "item": {{"type": "mcp_tool_call", "server": "repo", "tool": "read_file",
        "arguments": {{"path": "app/router.py"}}, "result": {{"content": [{{"type": "text", "text": "1 def route"}}]}}}}}},
    {{"type": "item.completed", "item": {{"type": "agent_message", "text": reply}}}},
    {{"type": "turn.completed"}},
]:
    print(json.dumps(ev), flush=True)
'''


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text(FAKE.format(py=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    from forge.llm import codex as mod
    monkeypatch.setattr(mod, "binary", lambda: str(exe))
    monkeypatch.setattr(mod, "_preflight", {})
    monkeypatch.setenv("FAKE_FEATURES", "\n".join(f"{f}  stable  false" for f in mod.DISABLED_FEATURES)
                       + "\ncode_mode_host  stable  true\nunified_exec  stable  true\nsqlite  removed  true")
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("FAKE_REPLY", json.dumps({"summary": "router", "files": ["app/router.py"]}))
    return lambda: [json.loads(x) for x in log.read_text().splitlines()]


async def test_explore_is_confined_and_parses_events(fake_codex, tmp_path):
    from forge.llm.codex import DISABLED_FEATURES, CodexLLM
    repo = tmp_path / "repo"
    repo.mkdir()
    events = []
    out = await CodexLLM(model="gpt-test").explore("find it", str(repo), Finding, events.append)
    assert out == Finding(summary="router", files=["app/router.py"])
    (call,) = fake_codex()
    argv = call["argv"]
    assert call["env_has_key"] is False                      # subscription login only
    assert argv[:2] == ["exec", "--json"]
    for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"):
        assert flag in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    assert {"shell_tool", "unified_exec", "view_image", "browser_use", "computer_use", "hooks", "multi_agent"} <= disabled
    assert disabled == set(DISABLED_FEATURES)
    assert 'approval_policy="never"' in argv and 'web_search="disabled"' in argv
    assert argv[argv.index("-m") + 1] == "gpt-test"
    # the repo is reachable only through our MCP server, and Codex itself runs in an empty temp dir
    mcp_args = next(a for a in argv if a.startswith("mcp_servers.repo.args="))
    assert "forge.llm.repo_mcp" in mcp_args and str(repo) in mcp_args
    workdir = argv[argv.index("-C") + 1]
    assert workdir != str(repo) and "forge-codex-" in workdir
    schema = json.loads(open(argv[argv.index("--output-schema") + 1]).read()) if os.path.exists(
        argv[argv.index("--output-schema") + 1]) else None
    assert schema is None  # temp dir is cleaned up after the run
    tools = [e for e in events if e["type"] == "tool"]
    assert tools == [{"type": "tool", "name": "read_file", "input": "app/router.py", "output": "1 def route"}]


async def test_structured_reasks_on_invalid_json(fake_codex, monkeypatch):
    from forge.llm.codex import CodexLLM
    monkeypatch.setenv("FAKE_REPLY_1", '{"summary": 3}')
    out = await CodexLLM().structured("q", Finding)
    assert out.summary == "router"
    calls = fake_codex()
    assert len(calls) == 2 and "failed validation" in calls[1]["prompt"]
    assert "-m" not in calls[0]["argv"]                      # "default" lets the CLI choose
    assert not any(a.startswith("mcp_servers.") for a in calls[0]["argv"])


async def test_chat_returns_reply_as_one_delta(fake_codex, monkeypatch):
    from forge.llm.codex import CodexLLM
    monkeypatch.setenv("FAKE_REPLY", "Hello there")
    deltas = []
    assert await CodexLLM().chat([{"role": "user", "content": "hi"}], on_delta=deltas.append) == "Hello there"
    assert deltas == ["Hello there"]
    assert "--output-schema" not in fake_codex()[0]["argv"]


async def test_failed_turn_raises(tmp_path, monkeypatch):
    from forge.llm import codex as mod
    from forge.llm import LLMError
    exe = tmp_path / "codex"
    exe.write_text(f"#!{sys.executable}\nimport json,sys\nsys.stdin.read()\n"
                   "print(json.dumps({'type':'turn.failed','error':{'message':'usage limit reached'}}))\nsys.exit(1)\n")
    exe.chmod(0o755)
    monkeypatch.setattr(mod, "binary", lambda: str(exe))
    monkeypatch.setattr(mod, "preflight", lambda: _none())
    with pytest.raises(LLMError, match="usage limit reached"):
        await mod.CodexLLM().structured("q", Finding)


def test_repo_mcp_server_protocol_and_confinement(tmp_path):
    from forge.llm import repo_mcp
    from forge.llm.openai import RepoTools
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path.parent / "outside_secret.txt").write_text("SECRET")
    tools = RepoTools(tmp_path)
    init = repo_mcp.handle(tools, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert repo_mcp.handle(tools, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in repo_mcp.handle(tools, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
    assert names == ["list_files", "grep", "read_file"]
    ok = repo_mcp.handle(tools, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "read_file", "arguments": {"path": "a.py"}}})["result"]
    assert "x = 1" in ok["content"][0]["text"] and not ok["isError"]
    for path in ("../outside_secret.txt", str(tmp_path.parent / "outside_secret.txt")):
        bad = repo_mcp.handle(tools, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                      "params": {"name": "read_file", "arguments": {"path": path}}})["result"]
        assert bad["isError"] and "SECRET" not in bad["content"][0]["text"]
    unknown = repo_mcp.handle(tools, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "shell"}})
    assert unknown["error"]["code"] == -32602


def test_get_llm_codex_default(isolated):  # noqa: F811
    from forge.llm import get_llm
    llm = get_llm(provider="codex-subscription")
    assert llm.provider == "codex-subscription" and llm.model == "default"


async def _none():
    return None


@pytest.mark.parametrize("item", [
    {"type": "command_execution", "command": "cat ~/.ssh/id_ed25519"},
    {"type": "file_change", "changes": []},
    {"type": "web_search", "query": "x"},
    {"type": "brand_new_tool"},
    {"type": "mcp_tool_call", "server": "someone_elses", "tool": "read"},
])
async def test_runtime_guard_kills_run_on_disallowed_tool(fake_codex, monkeypatch, item):
    from forge.llm import LLMError
    from forge.llm.codex import CodexLLM
    monkeypatch.setenv("FAKE_EXTRA_EVENT", json.dumps({"type": "item.started", "item": item}))
    with pytest.raises(LLMError, match="outside the allowed set"):
        await CodexLLM().structured("q", Finding)


async def test_explore_without_any_repo_tool_call_retries_then_fails(fake_codex, tmp_path, monkeypatch):
    from forge.llm import LLMError
    from forge.llm import codex as mod
    # strip the fake's mcp_tool_call event: Codex "answered blind"
    exe = mod.binary()
    src = open(exe).read().replace('"type": "mcp_tool_call"', '"type": "reasoning"')
    open(exe, "w").write(src)
    events = []
    with pytest.raises(LLMError, match="could not reach the repository tools"):
        await mod.CodexLLM().explore("find it", str(tmp_path), Finding, events.append)
    calls = fake_codex()
    assert len(calls) == 2 and "mcp__repo__read_file" in calls[1]["prompt"]
    assert any(e.get("level") == "warn" for e in events)


def test_check_features_is_an_exact_allowlist():
    from forge.llm.codex import ALLOWED_ENABLED, DISABLED_FEATURES, check_features
    ok = "\n".join(f"{f} stable false" for f in DISABLED_FEATURES) + "\ncode_mode_host stable true"
    assert check_features(ok) is None
    # codex 0.161 ignores --disable unified_exec: tolerated only while shell_tool is off
    ue = ok.replace("unified_exec stable false", "unified_exec stable true")
    assert check_features(ue) is None
    assert "ignored --disable" in check_features(ue.replace("shell_tool stable false", "shell_tool stable true"))
    assert check_features(ok + "\nsqlite removed true") is None            # removed features are inert
    # a disabled feature vanished (renamed/removed)
    assert "no longer has" in check_features(ok.replace("shell_tool stable false", ""))
    # --disable silently ignored
    assert "ignored --disable" in check_features(ok.replace("shell_tool stable false", "shell_tool stable true"))
    # ANY new enabled feature refuses, whatever its name (no denylist regex to slip past)
    for name in ("shell_tool_v2", "native_file_reader", "quiet_helper", "multi_agent_v2"):
        assert "hasn't vetted" in check_features(ok + f"\n{name} stable true"), name
        assert "hasn't vetted" in check_features(ok + f"\n{name} under development true"), name
        assert check_features(ok + f"\n{name} stable false") is None
    assert not ALLOWED_ENABLED & set(DISABLED_FEATURES)


async def test_preflight_uses_same_flags_and_empty_codex_home(fake_codex, monkeypatch, tmp_path):
    from forge.llm.codex import DISABLED_FEATURES, CodexLLM
    await CodexLLM().structured("q", Finding)
    import os as _os
    rec = json.loads(open(_os.environ["FAKE_CODEX_LOG"] + ".features").read())
    disabled = [rec["argv"][i + 1] for i, a in enumerate(rec["argv"]) if a == "--disable"]
    assert disabled == DISABLED_FEATURES                     # identical to what exec runs with
    assert rec["codex_home"] and "forge-codex-home-" in rec["codex_home"] and rec["home_files"] == []
    exec_argv = fake_codex()[0]["argv"]
    assert [exec_argv[i + 1] for i, a in enumerate(exec_argv) if a == "--disable"] == DISABLED_FEATURES


async def test_preflight_refuses_when_codex_rejects_flags(fake_codex, monkeypatch):
    from forge.llm import LLMError
    from forge.llm.codex import CodexLLM
    monkeypatch.setenv("FAKE_FEATURES", "error: unknown feature `shell_tool`")
    monkeypatch.setenv("FAKE_FEATURES_RC", "1")
    with pytest.raises(LLMError, match="rejected forge's sandbox flags"):
        await CodexLLM().structured("q", Finding)
    assert not os.path.exists(os.environ["FAKE_CODEX_LOG"])     # no exec run was started


from test_agent_fakes import isolated  # noqa: E402,F401  (fixture)
