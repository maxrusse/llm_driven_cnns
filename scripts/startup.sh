#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/config/daemon_config.json"
if [[ -f "$REPO_ROOT/config/daemon_config.linux.json" ]]; then
  CONFIG_PATH="$REPO_ROOT/config/daemon_config.linux.json"
fi
START_IN_NEW_WINDOW=0
RUN_HOURS="0"
FINISHUP_MINUTES="60"
FINISHUP_FINAL_TRAINING_ROUNDS="1"
FINISHUP_TOPK="10"
FINISHUP_NOTE=""

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
    --start-in-new-window)
      START_IN_NEW_WINDOW=1
      shift
      ;;
    --run-hours)
      RUN_HOURS="$2"
      shift 2
      ;;
    --finishup-minutes)
      FINISHUP_MINUTES="$2"
      shift 2
      ;;
    --finishup-final-training-rounds)
      FINISHUP_FINAL_TRAINING_ROUNDS="$2"
      shift 2
      ;;
    --finishup-top-k)
      FINISHUP_TOPK="$2"
      shift 2
      ;;
    --finishup-note)
      FINISHUP_NOTE="$2"
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
data_raw = str(cfg.get("data_source_root", "")).strip()
data_src = resolve(data_raw, pathlib.Path(workspace)) if data_raw else ""
print(workspace)
print(data_src)
PY
)

WORKSPACE_ROOT="${CFG_OUT[0]}"
DATA_SOURCE_ROOT="${CFG_OUT[1]}"
EFFECTIVE_CONFIG_PATH="$CONFIG_PATH"

"$REPO_ROOT/scripts/link_data.sh" --workspace-root "$WORKSPACE_ROOT" --data-source-root "$DATA_SOURCE_ROOT"

if [[ "$("$PYTHON_BIN" - <<'PY' "$RUN_HOURS"
import sys
print("1" if float(sys.argv[1]) > 0 else "0")
PY
)" == "1" ]]; then
  "$REPO_ROOT/scripts/request_finishup.sh" \
    --workspace-root "$WORKSPACE_ROOT" \
    --run-hours "$RUN_HOURS" \
    --minutes-left "$FINISHUP_MINUTES" \
    --final-training-rounds "$FINISHUP_FINAL_TRAINING_ROUNDS" \
    --top-k "$FINISHUP_TOPK" \
    --note "$FINISHUP_NOTE"
  echo "Scheduled finish-up via startup: RunHours=$RUN_HOURS, FinishupMinutes=$FINISHUP_MINUTES"
fi

DAEMON_SCRIPT="$REPO_ROOT/scripts/start_llm_daemon.sh"
if [[ "$START_IN_NEW_WINDOW" -eq 1 ]]; then
  mkdir -p "$WORKSPACE_ROOT/.llm_loop/logs"
  nohup "$DAEMON_SCRIPT" --config-path "$EFFECTIVE_CONFIG_PATH" > "$WORKSPACE_ROOT/.llm_loop/logs/daemon.nohup.log" 2>&1 &
  echo "Daemon started in background (PID $!)."
else
  "$DAEMON_SCRIPT" --config-path "$EFFECTIVE_CONFIG_PATH"
fi
