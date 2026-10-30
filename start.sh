#!/usr/bin/env bash
# Start CLM Forge. Ctrl+C (or SIGTERM/SIGHUP) stops it and kills every process it started,
# including grandchildren (Claude Agent SDK CLI sessions, git clones, wheel builds, Vite).
#
#   ./start.sh          backend on :8765 serving the built UI (builds frontend/dist if stale)
#   ./start.sh --dev    backend on :8765 + Vite dev server with hot reload on :5173
#
# Ollama is a system service the app only talks to; it is not started or stopped here.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${FORGE_PORT:-8765}"
UI_PORT=5173
DEV=0
[[ "${1:-}" == "--dev" ]] && DEV=1

PIDS=()

log() { printf '\033[1;34m[forge]\033[0m %s\n' "$*"; }

# All descendants of a pid, deepest first, so children die before parents can respawn them.
descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    descendants "$child"
    echo "$child"
  done
}

port_in_use() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1; }

STOPPING=0
cleanup() {
  [[ $STOPPING == 1 ]] && return
  STOPPING=1
  trap - INT TERM HUP EXIT
  log "stopping..."
  local all=() pid
  for pid in "${PIDS[@]}"; do
    all+=($(descendants "$pid") "$pid")
  done
  [[ ${#all[@]} -gt 0 ]] && kill -TERM "${all[@]}" 2>/dev/null || true
  # give uvicorn and Vite up to 5 s to exit cleanly, then force-kill anything left
  for _ in {1..50}; do
    local alive=0
    for pid in "${all[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1 && break; done
    [[ $alive == 0 ]] && break
    sleep 0.1
  done
  for pid in "${all[@]}"; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true; done
  # anything still holding our ports was ours
  for port in "$API_PORT" $([[ $DEV == 1 ]] && echo "$UI_PORT"); do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | xargs kill -KILL 2>/dev/null || true
  done
  log "stopped"
  exit 0
}
trap cleanup INT TERM HUP EXIT

for port in "$API_PORT" $([[ $DEV == 1 ]] && echo "$UI_PORT"); do
  if port_in_use "$port"; then
    echo "Port $port is already in use (pid $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t | tr '\n' ' ')). Stop that first." >&2
    trap - EXIT; exit 1
  fi
done

command -v uv >/dev/null || { echo "uv is required: brew install uv" >&2; exit 1; }
command -v npm >/dev/null || { echo "npm is required: brew install node" >&2; exit 1; }

log "syncing backend dependencies"
(cd "$ROOT/backend" && uv sync -q)

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  log "installing frontend dependencies"
  (cd "$ROOT/frontend" && npm install --silent)
fi

if [[ $DEV == 0 ]]; then
  # rebuild when any source file is newer than the last build
  if [[ ! -f "$ROOT/frontend/dist/index.html" ]] || \
     [[ -n "$(find "$ROOT/frontend/src" "$ROOT/frontend/index.html" -newer "$ROOT/frontend/dist/index.html" -print -quit)" ]]; then
    log "building frontend"
    (cd "$ROOT/frontend" && npm run build --silent >/dev/null)
  fi
fi

curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || log "note: Ollama is not reachable; the Ollama provider will be unavailable"

log "starting backend on http://127.0.0.1:$API_PORT"
(cd "$ROOT/backend" && exec env FORGE_PORT="$API_PORT" uv run forge-api) &
PIDS+=($!)

if [[ $DEV == 1 ]]; then
  log "starting Vite dev server on http://127.0.0.1:$UI_PORT"
  (cd "$ROOT/frontend" && exec env FORGE_PORT="$API_PORT" npm run dev -- --port "$UI_PORT" --strictPort) &
  PIDS+=($!)
fi

for _ in {1..60}; do
  curl -s -m 1 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done
URL="http://127.0.0.1:$([[ $DEV == 1 ]] && echo "$UI_PORT" || echo "$API_PORT")"
log "ready: $URL   (Ctrl+C to stop)"

# exit (and clean up everything) as soon as any of our processes dies
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "process $pid exited; shutting down the rest"
      cleanup
    fi
  done
  sleep 1 & wait $! 2>/dev/null || true
done
