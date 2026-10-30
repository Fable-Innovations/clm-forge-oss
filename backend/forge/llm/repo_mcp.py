"""A minimal stdio MCP server exposing the read-only repo tools (list_files, grep, read_file).

The Codex provider runs Codex with its own shell and file tools switched off and this server as its
only way to look at code. Codex's read-only sandbox blocks writes and network but not reads, so a
prompt-injected repository could otherwise have it ``cat ~/.ssh/id_ed25519``; these tools are
confined to the repository by ``repo.safe_path`` (see ``openai.RepoTools``).

Usage: ``python -m forge.llm.repo_mcp <repo-root>``. Speaks newline-delimited JSON-RPC 2.0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from forge.llm.openai import TOOLS, RepoTools

PROTOCOL = "2025-06-18"


def _tool_defs() -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "inputSchema": t["function"]["parameters"],
             "annotations": {"readOnlyHint": True, "openWorldHint": False}} for t in TOOLS]


def handle(tools: RepoTools, req: dict) -> dict | None:
    mid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
    if mid is None:  # notification
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": "forge-repo", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": _tool_defs()}
    elif method == "tools/call":
        name = params.get("name", "")
        if name not in {t["function"]["name"] for t in TOOLS}:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        out, _label = tools.run(name, params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": out}], "isError": out.startswith(("error:", "invalid regex"))}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1 or not Path(args[0]).is_dir():
        sys.exit("usage: python -m forge.llm.repo_mcp <repo-root>")
    tools = RepoTools(Path(args[0]))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle(tools, req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
