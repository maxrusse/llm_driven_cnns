#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$REPO_ROOT"
DATA_SOURCE_ROOT="$(cd "$REPO_ROOT/../xray_fracture_benchmark/data" 2>/dev/null && pwd || true)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --data-source-root)
      DATA_SOURCE_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TARGET="$WORKSPACE_ROOT/data"
if [[ -e "$TARGET" ]]; then
  echo "Data link already exists: $TARGET"
  exit 0
fi

if [[ -z "$DATA_SOURCE_ROOT" || ! -e "$DATA_SOURCE_ROOT" ]]; then
  echo "Data source not found: $DATA_SOURCE_ROOT" >&2
  exit 1
fi

ln -s "$DATA_SOURCE_ROOT" "$TARGET"
echo "Created data symlink: $TARGET -> $DATA_SOURCE_ROOT"
