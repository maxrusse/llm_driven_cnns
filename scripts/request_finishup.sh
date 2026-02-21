#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "python3/python not found in PATH." >&2; exit 1; }

"$PYTHON_BIN" - "$REPO_ROOT" "$@" <<'PY'
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone

repo_root = pathlib.Path(sys.argv[1]).resolve()
argv = sys.argv[2:]

p = argparse.ArgumentParser(description="Create/show/cancel finish-up request.")
p.add_argument("--workspace-root", default=str(repo_root))
p.add_argument("--minutes-left", type=int, default=60)
p.add_argument("--final-training-rounds", type=int, default=1)
p.add_argument("--top-k", type=int, default=10)
p.add_argument("--note", default="")
p.add_argument("--activate-in-minutes", type=int, default=0)
p.add_argument("--activate-at-utc", default="")
p.add_argument("--run-hours", type=float, default=0.0)
p.add_argument("--force-report-now", action="store_true")
p.add_argument("--cancel", action="store_true")
p.add_argument("--show", action="store_true")
args = p.parse_args(argv)

workspace_root = pathlib.Path(args.workspace_root).resolve()
loop_dir = workspace_root / ".llm_loop"
loop_dir.mkdir(parents=True, exist_ok=True)
control_path = loop_dir / "FINISH_UP.json"

if args.cancel:
    if control_path.exists():
        control_path.unlink()
        print(f"Finish-up request removed: {control_path}")
    else:
        print("No finish-up request file found.")
    raise SystemExit(0)

if args.show:
    if control_path.exists():
        print(control_path.read_text(encoding="utf-8"))
    else:
        print(f"No finish-up request file found at {control_path}")
    raise SystemExit(0)

minutes = max(5, int(args.minutes_left))
rounds = max(0, int(args.final_training_rounds))
top_k = max(3, min(20, int(args.top_k)))
now = datetime.now(timezone.utc)
activate_at = now
activation_mode = "immediate"

activate_in_minutes = int(args.activate_in_minutes)
if float(args.run_hours) > 0:
    total_minutes = round(float(args.run_hours) * 60.0)
    activate_in_minutes = max(0, int(total_minutes) - minutes)

if str(args.activate_at_utc).strip():
    txt = str(args.activate_at_utc).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    activate_at = datetime.fromisoformat(txt)
    if activate_at.tzinfo is None:
        activate_at = activate_at.replace(tzinfo=timezone.utc)
    activate_at = activate_at.astimezone(timezone.utc)
    activation_mode = "scheduled_at_utc"
elif activate_in_minutes > 0:
    activate_at = now + timedelta(minutes=activate_in_minutes)
    activation_mode = "scheduled_in_minutes"

deadline = activate_at + timedelta(minutes=minutes)
status = "scheduled" if activate_at > now else "requested"

payload = {
    "enabled": True,
    "status": status,
    "requested_utc": now.isoformat(),
    "activate_at_utc": activate_at.isoformat(),
    "deadline_utc": deadline.isoformat(),
    "minutes_left": minutes,
    "total_minutes_window": minutes,
    "final_training_rounds_target": rounds,
    "report_top_k": top_k,
    "force_report_now": bool(args.force_report_now),
    "activation_mode": activation_mode,
    "note": str(args.note or ""),
}

control_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
print("Finish-up request written:")
print(f"  file: {control_path}")
print(f"  deadline_utc: {payload['deadline_utc']}")
print(f"  final_training_rounds_target: {payload['final_training_rounds_target']}")
print(f"  report_top_k: {payload['report_top_k']}")
print(f"  status: {payload['status']}")
print(f"  activate_at_utc: {payload['activate_at_utc']}")
if payload["force_report_now"]:
    print("  mode: report_now")
elif payload["status"] == "scheduled":
    print("  mode: scheduled_finishup_then_report")
else:
    print("  mode: final_training_then_report")
PY
