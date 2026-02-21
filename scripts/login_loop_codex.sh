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

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH." >&2
  exit 1
fi

CODEX_HOME_DIR="$WORKSPACE_ROOT/.llm_loop/codex_home"
mkdir -p "$CODEX_HOME_DIR"

echo "Using CODEX_HOME=$CODEX_HOME_DIR"
CODEX_HOME="$CODEX_HOME_DIR" codex login --device-auth
CODEX_HOME="$CODEX_HOME_DIR" codex login status
