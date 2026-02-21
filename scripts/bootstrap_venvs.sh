#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LOOP_REPO_ROOT="$REPO_ROOT"
TRAIN_REPO_ROOT="$WORKSPACE_ROOT/xray_fracture_benchmark"
LOOP_VENV_PATH="$WORKSPACE_ROOT/llm_driven_cnns_venv"
TRAIN_VENV_PATH="$WORKSPACE_ROOT/xray_fracture_benchmark_venv"
SKIP_CUDA=0
SKIP_AUTOREPAIR_EXTRAS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      WORKSPACE_ROOT="$2"
      TRAIN_REPO_ROOT="$WORKSPACE_ROOT/xray_fracture_benchmark"
      LOOP_VENV_PATH="$WORKSPACE_ROOT/llm_driven_cnns_venv"
      TRAIN_VENV_PATH="$WORKSPACE_ROOT/xray_fracture_benchmark_venv"
      shift 2
      ;;
    --loop-repo-root)
      LOOP_REPO_ROOT="$2"
      shift 2
      ;;
    --train-repo-root)
      TRAIN_REPO_ROOT="$2"
      shift 2
      ;;
    --loop-venv-path)
      LOOP_VENV_PATH="$2"
      shift 2
      ;;
    --train-venv-path)
      TRAIN_VENV_PATH="$2"
      shift 2
      ;;
    --skip-cuda)
      SKIP_CUDA=1
      shift
      ;;
    --skip-auto-repair-extras)
      SKIP_AUTOREPAIR_EXTRAS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

LOOP_REQ="$LOOP_REPO_ROOT/requirements_wrapper.txt"
TRAIN_REQ="$TRAIN_REPO_ROOT/requirements.txt"
TRAIN_REQ_CUDA="$TRAIN_REPO_ROOT/requirements-cu128.txt"
DAEMON_CFG="$LOOP_REPO_ROOT/config/daemon_config.json"

[[ -f "$LOOP_REQ" ]] || { echo "Missing loop requirements file: $LOOP_REQ" >&2; exit 1; }
[[ -f "$TRAIN_REQ" ]] || { echo "Missing training requirements file: $TRAIN_REQ" >&2; exit 1; }

PY_BOOTSTRAP="python3"
if ! command -v "$PY_BOOTSTRAP" >/dev/null 2>&1; then
  PY_BOOTSTRAP="python"
fi
command -v "$PY_BOOTSTRAP" >/dev/null 2>&1 || { echo "No Python bootstrap executable found (python3/python)." >&2; exit 1; }

ensure_venv_python() {
  local venv_path="$1"
  if [[ ! -d "$venv_path" ]]; then
    echo "Creating venv: $venv_path"
    "$PY_BOOTSTRAP" -m venv "$venv_path"
  fi
  local py="$venv_path/bin/python"
  [[ -x "$py" ]] || { echo "Python executable not found in venv: $py" >&2; exit 1; }
  "$py" -m pip --version >/dev/null 2>&1 || "$py" -m ensurepip --upgrade >/dev/null 2>&1
  "$py" -m pip --version >/dev/null 2>&1 || { echo "pip unavailable in venv: $venv_path" >&2; exit 1; }
  echo "$py"
}

install_req_file() {
  local py="$1"
  local req="$2"
  local label="$3"
  echo "Installing requirements ($label): $req"
  "$py" -m pip install --upgrade pip
  "$py" -m pip install -r "$req"
}

LOOP_PY="$(ensure_venv_python "$LOOP_VENV_PATH")"
TRAIN_PY="$(ensure_venv_python "$TRAIN_VENV_PATH")"

install_req_file "$LOOP_PY" "$LOOP_REQ" "loop"
install_req_file "$TRAIN_PY" "$TRAIN_REQ" "training-base"

if [[ "$SKIP_CUDA" -eq 0 ]]; then
  if [[ -f "$TRAIN_REQ_CUDA" ]]; then
    install_req_file "$TRAIN_PY" "$TRAIN_REQ_CUDA" "training-cuda"
  else
    echo "WARNING: CUDA requirements file not found; skipping: $TRAIN_REQ_CUDA" >&2
  fi
else
  echo "SkipCuda set; not installing training CUDA wheel requirements."
fi

if [[ "$SKIP_AUTOREPAIR_EXTRAS" -eq 0 ]]; then
  if [[ -f "$DAEMON_CFG" ]]; then
    echo "Preinstalling auto-repair extras from: $DAEMON_CFG"
    mapfile -t extras < <(
      "$PY_BOOTSTRAP" - "$DAEMON_CFG" <<'PY'
import json
import pathlib
import sys

cfg = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pkgs = set()
for key in ("auto_repair_module_package_map", "auto_repair_module_alias_map", "auto_repair_model_package_map"):
    obj = cfg.get(key, {})
    if not isinstance(obj, dict):
        continue
    for v in obj.values():
        if isinstance(v, list):
            for item in v:
                s = str(item).strip()
                if s:
                    pkgs.add(s)
        else:
            s = str(v).strip()
            if s:
                pkgs.add(s)
for p in sorted(pkgs):
    print(p)
PY
    )
    if [[ "${#extras[@]}" -gt 0 ]]; then
      "$TRAIN_PY" -m pip install --disable-pip-version-check "${extras[@]}"
      echo "Installed auto-repair extras: ${extras[*]}"
    else
      echo "No auto-repair package extras configured."
    fi
  else
    echo "WARNING: daemon_config.json missing; skipping auto-repair extras preinstall." >&2
  fi
else
  echo "SkipAutoRepairExtras set; not preinstalling auto-repair packages."
fi

loop_count="$("$LOOP_PY" -m pip list --format=freeze | wc -l | tr -d ' ')"
train_count="$("$TRAIN_PY" -m pip list --format=freeze | wc -l | tr -d ' ')"
echo
echo "Bootstrap complete."
echo "Loop venv: $LOOP_VENV_PATH (packages: $loop_count)"
echo "Training venv: $TRAIN_VENV_PATH (packages: $train_count)"
