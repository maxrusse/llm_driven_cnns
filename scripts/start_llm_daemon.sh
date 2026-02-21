#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/config/daemon_config.json"
if [[ -f "$REPO_ROOT/config/daemon_config.linux.json" ]]; then
  CONFIG_PATH="$REPO_ROOT/config/daemon_config.linux.json"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-path)
      if [[ "$2" = /* ]]; then
        CONFIG_PATH="$2"
      else
        CONFIG_PATH="$REPO_ROOT/$2"
      fi
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "python3/python not found in PATH." >&2; exit 1; }

readarray -t CFG_OUT < <(
  "$PYTHON_BIN" - "$CONFIG_PATH" "$REPO_ROOT" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1]).resolve()
repo_root = pathlib.Path(sys.argv[2]).resolve()
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

def resolve(raw: str, base: pathlib.Path) -> str:
    p = pathlib.Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)

workspace_raw = str(cfg.get("workspace_root", "")).strip()
workspace = resolve(workspace_raw, repo_root) if workspace_raw else str(repo_root)
py_raw = str(cfg.get("python_exe", "")).strip()
py = resolve(py_raw, pathlib.Path(workspace)) if py_raw else ""
run_id = str(cfg.get("run_id", "llm_driven_cnns_day01"))
poll = int(cfg.get("daemon_poll_seconds", 20))
if poll < 5:
    poll = 5

print(workspace)
print(py)
print(run_id)
print(str(poll))
PY
)

WORKSPACE_ROOT="${CFG_OUT[0]}"
PYTHON_EXE_CFG="${CFG_OUT[1]}"
RUN_ID="${CFG_OUT[2]}"
POLL_SECONDS="${CFG_OUT[3]}"

if [[ -n "$PYTHON_EXE_CFG" && -x "$PYTHON_EXE_CFG" ]]; then
  PYTHON_EXE="$PYTHON_EXE_CFG"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python)"
  else
    echo "Python not found and config.python_exe is invalid." >&2
    exit 1
  fi
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH. Install/update Codex CLI first." >&2
  exit 1
fi
CODEX_EXE="$(command -v codex)"

LOOP_DIR="$WORKSPACE_ROOT/.llm_loop"
LOGS_DIR="$LOOP_DIR/logs"
ARTIFACTS_DIR="$LOOP_DIR/artifacts"
mkdir -p "$LOOP_DIR" "$LOGS_DIR" "$ARTIFACTS_DIR"

STATE_FILE="$LOOP_DIR/state.json"
EVENTS_FILE="$LOGS_DIR/events.jsonl"
HEARTBEAT_FILE="$LOGS_DIR/daemon_heartbeat.json"
CODEX_HOME_DIR="$LOOP_DIR/codex_home"
STOP_DAEMON_FLAG="$LOOP_DIR/STOP_DAEMON"
STOP_CURRENT_RUN_FLAG="$LOOP_DIR/STOP_CURRENT_RUN"
THREAD_ID_FILE="$ARTIFACTS_DIR/codex_thread_id.txt"
CYCLE_SCRIPT="$REPO_ROOT/scripts/llm_cycle.py"

[[ -f "$CYCLE_SCRIPT" ]] || { echo "Missing cycle script: $CYCLE_SCRIPT" >&2; exit 1; }
mkdir -p "$CODEX_HOME_DIR"
rm -f "$STOP_DAEMON_FLAG" "$STOP_CURRENT_RUN_FLAG"
if [[ ! -f "$STATE_FILE" ]]; then
  printf '%s\n' '{"active_run": null}' > "$STATE_FILE"
fi

if ! CODEX_HOME="$CODEX_HOME_DIR" "$CODEX_EXE" login status >/dev/null 2>&1; then
  echo "Codex is not logged in for loop CODEX_HOME ($CODEX_HOME_DIR). Run ./scripts/login_loop_codex.sh first." >&2
  exit 1
fi

cycle=0
echo "Starting LLM daemon run_id=$RUN_ID"
echo "Workspace: $WORKSPACE_ROOT"
echo "CODEX_HOME: $CODEX_HOME_DIR"
echo "Stop flag: $STOP_DAEMON_FLAG"

while true; do
  if [[ -f "$STOP_DAEMON_FLAG" ]]; then
    echo "STOP_DAEMON found. Exiting daemon loop."
    break
  fi

  cycle=$((cycle + 1))
  started_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  set +e
  raw="$("$PYTHON_EXE" "$CYCLE_SCRIPT" \
    --workspace-root "$WORKSPACE_ROOT" \
    --config-path "$CONFIG_PATH" \
    --codex-exe "$CODEX_EXE" \
    --codex-home "$CODEX_HOME_DIR" \
    --thread-id-file "$THREAD_ID_FILE" \
    --state-file "$STATE_FILE" \
    --events-file "$EVENTS_FILE" \
    --stop-daemon-flag "$STOP_DAEMON_FLAG" \
    --stop-current-run-flag "$STOP_CURRENT_RUN_FLAG" 2>&1)"
  exit_code=$?
  set -e
  ended_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  daemon_status="running"
  if [[ "$exit_code" -ne 0 ]]; then
    daemon_status="degraded"
  fi

  printf '%s' "$raw" | "$PYTHON_BIN" - "$HEARTBEAT_FILE" "$daemon_status" "$RUN_ID" "$cycle" "$started_utc" "$ended_utc" "$exit_code" "$STATE_FILE" "$EVENTS_FILE" "$CODEX_HOME_DIR" "$STOP_DAEMON_FLAG" "$STOP_CURRENT_RUN_FLAG" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

raw = sys.stdin.read()
if len(raw) > 4000:
    raw = raw[-4000:]

payload = {
    "daemon_status": sys.argv[2],
    "run_id": sys.argv[3],
    "cycle": int(sys.argv[4]),
    "cycle_started_utc": sys.argv[5],
    "cycle_ended_utc": sys.argv[6],
    "last_exit_code": int(sys.argv[7]),
    "last_cycle_output": raw,
    "state_file": sys.argv[8],
    "events_file": sys.argv[9],
    "codex_home": sys.argv[10],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "stop_daemon_flag": sys.argv[11],
    "stop_current_run_flag": sys.argv[12],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
PY

  if [[ "$exit_code" -ne 0 ]]; then
    echo "WARNING: Cycle failed with exit code $exit_code"
  fi
  sleep "$POLL_SECONDS"
done

"$PYTHON_BIN" - "$HEARTBEAT_FILE" "$RUN_ID" "$cycle" "$STOP_DAEMON_FLAG" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

payload = {
    "daemon_status": "stopped",
    "run_id": sys.argv[2],
    "cycle": int(sys.argv[3]),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "stop_daemon_flag": sys.argv[4],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
PY

echo "LLM daemon stopped."
