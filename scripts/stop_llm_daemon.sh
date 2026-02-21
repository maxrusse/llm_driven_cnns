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

LOOP_DIR="$WORKSPACE_ROOT/.llm_loop"
mkdir -p "$LOOP_DIR"
: > "$LOOP_DIR/STOP_CURRENT_RUN"
: > "$LOOP_DIR/STOP_DAEMON"
echo "Stop flags created."
