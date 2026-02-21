#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

CFG_PATH="$WORKSPACE_ROOT/config/daemon_config.json"
HB="$WORKSPACE_ROOT/.llm_loop/logs/daemon_heartbeat.json"
STATE="$WORKSPACE_ROOT/.llm_loop/state.json"
EVENTS="$WORKSPACE_ROOT/.llm_loop/logs/events.jsonl"
STORYLINE="$WORKSPACE_ROOT/.llm_loop/artifacts/storyline.md"
WORKPAD="$WORKSPACE_ROOT/.llm_loop/artifacts/workpad.md"
FINISHUP="$WORKSPACE_ROOT/.llm_loop/FINISH_UP.json"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "python3/python not found in PATH." >&2; exit 1; }

if [[ -f "$HB" ]]; then
  echo "Heartbeat:"
  cat "$HB"
  "$PYTHON_BIN" - "$HB" "$CFG_PATH" "$STATE" <<'PY'
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

hb_path = pathlib.Path(sys.argv[1])
cfg_path = pathlib.Path(sys.argv[2])
state_path = pathlib.Path(sys.argv[3])

try:
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

poll_seconds = 20
if cfg_path.exists():
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        poll_seconds = int(cfg.get("daemon_poll_seconds", 20))
    except Exception:
        pass

stale_after = max(60, poll_seconds * 3)
updated_raw = str(hb.get("updated_utc", "")).strip()
if not updated_raw:
    raise SystemExit(0)
if updated_raw.endswith("Z"):
    updated_raw = updated_raw[:-1] + "+00:00"
try:
    updated = datetime.fromisoformat(updated_raw)
except Exception:
    raise SystemExit(0)
if updated.tzinfo is None:
    updated = updated.replace(tzinfo=timezone.utc)
updated = updated.astimezone(timezone.utc)
age_seconds = int((datetime.now(timezone.utc) - updated).total_seconds())

active_pid = None
active_alive = False
if state_path.exists():
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        active = state.get("active_run")
        if isinstance(active, dict):
            pid = active.get("pid")
            if isinstance(pid, int):
                active_pid = pid
                try:
                    os.kill(pid, 0)
                    active_alive = True
                except Exception:
                    active_alive = False
    except Exception:
        pass

daemon_status = str(hb.get("daemon_status", "")).strip().lower()
if age_seconds > stale_after and daemon_status in {"running", "degraded"}:
    if active_alive and active_pid is not None:
        print(f"WARNING: Heartbeat is stale ({age_seconds}s old), but active_run PID {active_pid} is alive.")
    else:
        print(f"WARNING: Heartbeat is stale ({age_seconds}s old). Daemon likely not running anymore.")
if daemon_status == "degraded" and active_alive and active_pid is not None:
    print(f"Effective status: running (active_run PID {active_pid} alive; last completed cycle is degraded).")
PY
else
  echo "Heartbeat file missing."
fi

if [[ -f "$STATE" ]]; then
  echo
  echo "State:"
  cat "$STATE"
fi

if [[ -f "$FINISHUP" ]]; then
  echo
  echo "Finish-up request:"
  cat "$FINISHUP"
fi

if [[ -f "$STORYLINE" ]]; then
  echo
  echo "Storyline (latest 40 lines):"
  tail -n 40 "$STORYLINE"
fi

if [[ -f "$WORKPAD" ]]; then
  echo
  echo "Workpad (latest 80 lines):"
  tail -n 80 "$WORKPAD"
fi

if [[ -f "$EVENTS" ]]; then
  echo
  echo "Recent events:"
  tail -n 30 "$EVENTS"
fi
