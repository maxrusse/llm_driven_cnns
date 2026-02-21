#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT"
INTERVAL_SECONDS=60
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$INTERVAL_SECONDS" -lt 5 ]]; then
  INTERVAL_SECONDS=5
fi

STATUS_SCRIPT="$REPO_ROOT/scripts/status.sh"
if [[ ! -f "$STATUS_SCRIPT" ]]; then
  echo "Missing status script: $STATUS_SCRIPT" >&2
  exit 1
fi

while true; do
  clear || true
  echo "LLM Daemon Watch | $(date +"%Y-%m-%d %H:%M:%S")"
  echo "WorkspaceRoot: $WORKSPACE_ROOT"
  echo
  "$STATUS_SCRIPT" --workspace-root "$WORKSPACE_ROOT"
  [[ "$ONCE" -eq 1 ]] && break
  sleep "$INTERVAL_SECONDS"
done
