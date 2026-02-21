#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)"
VENV_PATH="$WORKSPACE_ROOT_DEFAULT/llm_driven_cnns_venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "No Python bootstrap executable found (python3/python)." >&2
  exit 1
fi

PY_BOOTSTRAP="python3"
if ! command -v "$PY_BOOTSTRAP" >/dev/null 2>&1; then
  PY_BOOTSTRAP="python"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating venv: $VENV_PATH"
  "$PY_BOOTSTRAP" -m venv "$VENV_PATH"
fi

PY_EXE="$VENV_PATH/bin/python"
if [[ ! -x "$PY_EXE" ]]; then
  echo "Python executable not found in venv: $PY_EXE" >&2
  exit 1
fi

echo "Installing Python requirements (wrapper-only)..."
"$PY_EXE" -m pip install --upgrade pip
"$PY_EXE" -m pip install -r "$REPO_ROOT/requirements_wrapper.txt"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH. Install Codex CLI before starting the daemon." >&2
  exit 1
fi

echo "Codex CLI detected."
echo "Tool install complete."
