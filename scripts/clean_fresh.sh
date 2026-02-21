#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT"
KEEP_DATA_LINK=0
KEEP_CODEX_LOGIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --keep-data-link)
      KEEP_DATA_LINK=1
      shift
      ;;
    --keep-codex-login)
      KEEP_CODEX_LOGIN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

LOOP_DIR="$WORKSPACE_ROOT/.llm_loop"
RUNS_DIR="$WORKSPACE_ROOT/runs"
DATA_LINK="$WORKSPACE_ROOT/data"
CODEX_HOME_DIR="$LOOP_DIR/codex_home"
CODEX_BACKUP_DIR="$WORKSPACE_ROOT/.codex_home_preserve_$$"
PRESERVED_CODEX_HOME=0

if [[ "$KEEP_CODEX_LOGIN" -eq 1 && -d "$CODEX_HOME_DIR" ]]; then
  mv "$CODEX_HOME_DIR" "$CODEX_BACKUP_DIR"
  PRESERVED_CODEX_HOME=1
fi

rm -rf "$LOOP_DIR"
mkdir -p "$LOOP_DIR/logs"

if [[ "$KEEP_CODEX_LOGIN" -eq 1 ]]; then
  if [[ "$PRESERVED_CODEX_HOME" -eq 1 && -d "$CODEX_BACKUP_DIR" ]]; then
    mv "$CODEX_BACKUP_DIR" "$CODEX_HOME_DIR"
  else
    mkdir -p "$CODEX_HOME_DIR"
  fi
fi
rm -rf "$CODEX_BACKUP_DIR"

mkdir -p "$RUNS_DIR"
find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

if [[ "$KEEP_DATA_LINK" -eq 0 ]]; then
  rm -rf "$DATA_LINK"
fi

if [[ "$KEEP_CODEX_LOGIN" -eq 1 && "$PRESERVED_CODEX_HOME" -eq 1 && -d "$CODEX_HOME_DIR" ]]; then
  echo "Fresh cleanup complete. Preserved Codex loop login at: $CODEX_HOME_DIR"
elif [[ "$KEEP_CODEX_LOGIN" -eq 1 && -d "$CODEX_HOME_DIR" ]]; then
  echo "Fresh cleanup complete. KeepCodexLogin was set; codex_home path exists at: $CODEX_HOME_DIR"
  echo "If login was not previously stored there, run ./scripts/login_loop_codex.sh once."
else
  echo "Fresh cleanup complete."
fi
