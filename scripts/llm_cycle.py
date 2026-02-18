from __future__ import annotations

import ast
import argparse
import re
import json
import os
import pathlib
import subprocess
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

FORBIDDEN_EXECUTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bnnunetv2_(plan_and_preprocess|train|predict|find_best_configuration|preprocess|evaluate)\b", "nnunetv2_cli"),
    (r"\bpython(?:\.exe)?\s+-m\s+nnunet(?:v2)?\b", "python_module_nnunet"),
    (r"\bnn-u-net\b", "nn-u-net_cli"),
)
ACTION_VALUES: tuple[str, ...] = ("run_command", "stop_current_run", "wait", "shutdown_daemon")
IDEA_CATEGORY_VALUES: tuple[str, ...] = (
    "augmentation",
    "preprocessing",
    "data_sampling",
    "loss",
    "model_arch",
    "optimization",
    "evaluation",
    "other",
)
DEFAULT_AUTO_REPAIR_MODULE_PACKAGE_MAP: dict[str, list[str]] = {
    "segmentation_models_pytorch": ["segmentation-models-pytorch", "timm"],
    "timm": ["timm"],
    "einops": ["einops"],
    "transformers": ["transformers"],
}
DEFAULT_AUTO_REPAIR_MODULE_ALIAS_MAP: dict[str, list[str]] = {
    "cv2": ["opencv-python"],
    "pil": ["pillow"],
    "yaml": ["pyyaml"],
    "sklearn": ["scikit-learn"],
}
DEFAULT_AUTO_REPAIR_MODEL_PACKAGE_MAP: dict[str, list[str]] = {
    "unet_resnet34": ["segmentation-models-pytorch", "timm"],
    "unet_resnet34_dual": ["segmentation-models-pytorch", "timm"],
    "unet_resnet34_dual_head": ["segmentation-models-pytorch", "timm"],
}
DEFAULT_AUTO_REPAIR_MODEL_FALLBACK_MAP: dict[str, str] = {
    "unet_resnet34": "simple_unet",
    "unet_resnet34_dual": "simple_unet_dual_head",
    "unet_resnet34_dual_head": "simple_unet_dual_head",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def read_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def tail_text(path: pathlib.Path, max_bytes: int = 24000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = size - max_bytes if size > max_bytes else 0
            f.seek(start)
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def read_text_head(path: pathlib.Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    txt = txt.strip()
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars]


def ensure_workpad_file(loop_dir: pathlib.Path) -> pathlib.Path:
    artifacts_dir = loop_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    workpad_path = artifacts_dir / "workpad.md"
    if workpad_path.exists():
        return workpad_path

    lines = [
        "# LLM Workpad",
        "",
        "## TODO",
        "",
        "## Notes",
        "",
        "## Data Exploration",
        "",
    ]
    workpad_path.write_text("\n".join(lines), encoding="utf-8")

    legacy_parts: list[str] = []
    legacy_files = [
        ("TODO", artifacts_dir / "todo.md"),
        ("Notes", artifacts_dir / "notes.md"),
        ("Data Exploration", artifacts_dir / "data_exploration.md"),
    ]
    for title, src in legacy_files:
        txt = read_text_head(src, max_chars=16000)
        if txt:
            legacy_parts.append(f"### Legacy Import: {title}\n{txt}\n")
    if legacy_parts:
        with workpad_path.open("a", encoding="utf-8") as f:
            f.write("\n## Legacy Imports\n\n")
            f.write("\n".join(legacy_parts))
    return workpad_path


def ensure_coordination_files(loop_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    artifacts_dir = loop_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mentor_notes_path = artifacts_dir / "mentor_notes.md"
    shared_todo_path = artifacts_dir / "shared_todo.md"
    if not mentor_notes_path.exists():
        mentor_notes_path.write_text("# Mentor Notes\n\n", encoding="utf-8")
    if not shared_todo_path.exists():
        shared_todo_path.write_text("# Shared TODO\n\n", encoding="utf-8")
    return mentor_notes_path, shared_todo_path


def count_unresolved_shared_todos(shared_todo_path: pathlib.Path) -> int:
    txt = read_text_head(shared_todo_path, max_chars=60000)
    if not txt:
        return 0
    return sum(1 for line in txt.splitlines() if line.strip().startswith("- [ ]"))


def append_entries_to_section(md_path: pathlib.Path, section_title: str, entries: list[str]) -> int:
    if not entries:
        return 0
    txt = md_path.read_text(encoding="utf-8", errors="ignore") if md_path.exists() else ""
    lines = txt.splitlines()
    heading = f"## {section_title}"
    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == heading.lower():
            start_idx = i
            break
    if start_idx < 0:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(heading)
        lines.append("")
        start_idx = len(lines) - 2
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if lines[j].strip().startswith("## "):
            end_idx = j
            break
    insert_at = end_idx
    if insert_at > 0 and lines[insert_at - 1].strip():
        lines.insert(insert_at, "")
        insert_at += 1
    for e in entries:
        lines.insert(insert_at, e)
        insert_at += 1
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(entries)


def resolve_shared_todo_ids(shared_todo_path: pathlib.Path, resolve_ids: list[str], cycle_index: int) -> int:
    if not resolve_ids or not shared_todo_path.exists():
        return 0
    ids = {str(x).strip() for x in resolve_ids if str(x).strip()}
    if not ids:
        return 0
    lines = shared_todo_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    resolved = 0
    for i, line in enumerate(lines):
        if "- [ ]" not in line:
            continue
        matched_id = None
        for rid in ids:
            if f"[{rid}]" in line:
                matched_id = rid
                break
        if not matched_id:
            continue
        new_line = line.replace("- [ ]", "- [x]", 1)
        if "resolved by worker" not in new_line:
            new_line = new_line + f" (resolved by worker C{cycle_index:04d})"
        lines[i] = new_line
        resolved += 1
    if resolved:
        shared_todo_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return resolved


def append_worker_housekeeping_artifacts(
    *,
    workpad_path: pathlib.Path,
    shared_todo_path: pathlib.Path,
    cycle_index: int,
    housekeeping: dict[str, Any],
) -> dict[str, Any]:
    hk = housekeeping if isinstance(housekeeping, dict) else {}
    now_utc = utc_now()
    todo_new = hk.get("todo_new", [])
    notes_update = str(hk.get("notes_update", "")).strip()
    data_update = str(hk.get("data_exploration_update", "")).strip()
    resolve_ids = hk.get("resolve_shared_todo_ids", [])

    if not isinstance(todo_new, list):
        todo_new = []
    if not isinstance(resolve_ids, list):
        resolve_ids = []

    todo_entries = []
    todo_seq = 1
    for item in todo_new[:4]:
        clean = " ".join(str(item).strip().split())
        if not clean:
            continue
        todo_entries.append(f"- [ ] [C{cycle_index:04d}-W{todo_seq:02d}] {clean[:220]} ({now_utc})")
        todo_seq += 1
    notes_entries = [f"- [{now_utc}] {notes_update[:500]}"] if notes_update else []
    data_entries = [f"- [{now_utc}] {data_update[:500]}"] if data_update else []

    todo_added = append_entries_to_section(workpad_path, "TODO", todo_entries)
    notes_added = append_entries_to_section(workpad_path, "Notes", notes_entries)
    data_added = append_entries_to_section(workpad_path, "Data Exploration", data_entries)
    resolved_shared = resolve_shared_todo_ids(shared_todo_path, [str(x) for x in resolve_ids[:6]], cycle_index)
    unresolved_after = count_unresolved_shared_todos(shared_todo_path)

    return {
        "todo_added": todo_added,
        "notes_added": notes_added,
        "data_exploration_added": data_added,
        "shared_todo_resolved": resolved_shared,
        "shared_todo_unresolved": unresolved_after,
    }


def append_mentor_coordination_artifacts(
    *,
    mentor_notes_path: pathlib.Path,
    shared_todo_path: pathlib.Path,
    cycle_index: int,
    mentor_recommendation: str,
    mentor_critique: str,
    mentor_questions: list[str],
    mentor_notes: str,
    todo_updates: list[str],
) -> tuple[int, int]:
    now_utc = utc_now()
    note_lines = [
        f"## Cycle {cycle_index:04d} | {now_utc}",
        f"- Recommendation: {mentor_recommendation or 'n/a'}",
        f"- Critique: {mentor_critique or 'n/a'}",
    ]
    if mentor_questions:
        note_lines.append("- Questions:")
        note_lines.extend([f"  - {q}" for q in mentor_questions[:3]])
    if mentor_notes:
        note_lines.append(f"- Notes: {mentor_notes}")
    note_lines.append("")
    with mentor_notes_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(note_lines))

    added = 0
    if todo_updates:
        with shared_todo_path.open("a", encoding="utf-8") as f:
            for idx, item in enumerate(todo_updates[:6], start=1):
                clean = str(item).strip()
                if not clean:
                    continue
                todo_id = f"C{cycle_index:04d}-M{idx:02d}"
                f.write(f"- [ ] [{todo_id}] mentor: {clean}\n")
                added += 1
            if added:
                f.write("\n")
    unresolved = count_unresolved_shared_todos(shared_todo_path)
    return added, unresolved


def run_cmd(args: list[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True, text=True)


def count_jsonl_rows(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def read_recent_jsonl(path: pathlib.Path, max_lines: int = 40) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = tail_text(path, max_bytes=120000).splitlines()
    out: list[dict[str, Any]] = []
    for raw in lines[-max_lines:]:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def infer_idea_category(*, rationale: str = "", command: str = "", run_label: str = "") -> str:
    text = f"{rationale} {command} {run_label}".lower()
    mapping = [
        ("augmentation", ["aug", "augmentation", "flip", "rotate", "colorjitter", "albumentation"]),
        ("preprocessing", ["preprocess", "normaliz", "resize", "crop", "clahe", "window", "histogram"]),
        ("data_sampling", ["sampl", "class weight", "oversampl", "undersampl", "max_train_batches", "batch"]),
        ("loss", ["loss", "dice", "focal", "bce", "presence_bce_weight", "weight_decay"]),
        ("model_arch", ["unet", "deeplab", "resnet", "architecture", "backbone", "encoder", "decoder"]),
        ("optimization", ["lr", "learning_rate", "scheduler", "cosine", "warmup", "epochs", "optimizer"]),
        ("evaluation", ["max_eval_batches", "validation", "eval", "threshold", "metric"]),
    ]
    for cat, keys in mapping:
        if any(k in text for k in keys):
            return cat
    return "other"


def detect_forbidden_command_usage(command: str) -> tuple[bool, str]:
    text = command or ""
    for pattern, label in FORBIDDEN_EXECUTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True, label
    return False, ""


def is_training_command_text(text: str) -> bool:
    t = (text or "").lower()
    hints = (
        "scripts\\train.py",
        "scripts/train.py",
        " train.py",
        "--output-dir",
        "trainer.fit(",
    )
    return any(h in t for h in hints)


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    proc = run_cmd(["cmd.exe", "/c", f"tasklist /FI \"PID eq {pid}\""])
    if proc.returncode != 0:
        return False
    out = (proc.stdout or "").lower()
    return str(pid) in out and "no tasks are running" not in out


def kill_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    proc = run_cmd(["cmd.exe", "/c", f"taskkill /PID {pid} /T /F"])
    return proc.returncode == 0


def build_decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "rationale",
            "command",
            "run_label",
            "idea_category",
            "monitor_seconds",
            "stop_patterns",
            "housekeeping",
        ],
        "properties": {
            "action": {"type": "string", "enum": list(ACTION_VALUES)},
            "rationale": {"type": "string"},
            "command": {"type": "string"},
            "run_label": {"type": "string"},
            "idea_category": {
                "type": "string",
                "enum": list(IDEA_CATEGORY_VALUES),
            },
            "monitor_seconds": {"type": "integer", "minimum": 15, "maximum": 7200},
            "stop_patterns": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string"},
            },
            "housekeeping": {
                "type": "object",
                "additionalProperties": False,
                "required": ["todo_new", "notes_update", "data_exploration_update", "resolve_shared_todo_ids"],
                "properties": {
                    "todo_new": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                    "notes_update": {"type": "string"},
                    "data_exploration_update": {"type": "string"},
                    "resolve_shared_todo_ids": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {"type": "string"},
                    },
                },
            },
        },
    }


def build_output_schema() -> dict[str, Any]:
    return build_decision_schema()


def build_mentor_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["recommendation", "critique", "questions", "mentor_notes", "todo_updates", "suggested_decision"],
        "properties": {
            "recommendation": {"type": "string", "enum": ["continue", "challenge"]},
            "critique": {"type": "string"},
            "questions": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "mentor_notes": {"type": "string"},
            "todo_updates": {
                "type": "array",
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "suggested_decision": {
                "anyOf": [
                    {"type": "null"},
                    build_decision_schema(),
                ]
            },
        },
    }


def coerce_decision(
    raw_decision: Any,
    *,
    default_monitor_seconds: int,
    max_monitor_seconds: int,
) -> dict[str, Any]:
    raw = raw_decision if isinstance(raw_decision, dict) else {}
    action = str(raw.get("action", "wait")).strip().lower()
    if action not in ACTION_VALUES:
        action = "wait"
    command = str(raw.get("command", "")).strip()
    run_label = str(raw.get("run_label", "run")).strip() or "run"
    rationale = str(raw.get("rationale", "")).strip()
    idea_category = str(raw.get("idea_category", "")).strip().lower()
    if idea_category not in IDEA_CATEGORY_VALUES:
        idea_category = infer_idea_category(rationale=rationale, command=command, run_label=run_label)
    if idea_category not in IDEA_CATEGORY_VALUES:
        idea_category = "other"
    monitor_seconds_raw = raw.get("monitor_seconds", default_monitor_seconds)
    try:
        monitor_seconds = int(monitor_seconds_raw)
    except Exception:
        monitor_seconds = int(default_monitor_seconds)
    monitor_seconds = max(15, min(max_monitor_seconds, monitor_seconds))
    stop_patterns = raw.get("stop_patterns", [])
    if not isinstance(stop_patterns, list):
        stop_patterns = []
    stop_patterns_out = [str(x) for x in stop_patterns][:8]
    hk_raw = raw.get("housekeeping", {})
    if not isinstance(hk_raw, dict):
        hk_raw = {}
    todo_new_raw = hk_raw.get("todo_new", [])
    if not isinstance(todo_new_raw, list):
        todo_new_raw = []
    todo_new = []
    for item in todo_new_raw[:4]:
        clean = " ".join(str(item).strip().split())
        if clean:
            todo_new.append(clean[:220])
    notes_update = " ".join(str(hk_raw.get("notes_update", "")).strip().split())[:500]
    data_exploration_update = " ".join(str(hk_raw.get("data_exploration_update", "")).strip().split())[:500]
    resolve_ids_raw = hk_raw.get("resolve_shared_todo_ids", [])
    if not isinstance(resolve_ids_raw, list):
        resolve_ids_raw = []
    resolve_ids = []
    for rid in resolve_ids_raw[:6]:
        token = str(rid).strip()
        if token:
            resolve_ids.append(token[:64])
    return {
        "action": action,
        "command": command,
        "run_label": run_label,
        "rationale": rationale,
        "idea_category": idea_category,
        "monitor_seconds": monitor_seconds,
        "stop_patterns": stop_patterns_out,
        "housekeeping": {
            "todo_new": todo_new,
            "notes_update": notes_update,
            "data_exploration_update": data_exploration_update,
            "resolve_shared_todo_ids": resolve_ids,
        },
    }


def decision_quality_issues(decision: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    action = str(decision.get("action", "")).strip().lower()
    rationale = str(decision.get("rationale", "")).strip()
    run_label = str(decision.get("run_label", "")).strip()
    command = str(decision.get("command", "")).strip()
    hk = decision.get("housekeeping", {})
    if not isinstance(hk, dict):
        hk = {}

    if len(rationale) < 20:
        issues.append("rationale_too_short")

    # For routine worker cycles, require at least one concrete housekeeping update.
    if action in {"run_command", "wait"}:
        todo_new = hk.get("todo_new", [])
        resolve_ids = hk.get("resolve_shared_todo_ids", [])
        notes_update = str(hk.get("notes_update", "")).strip()
        data_update = str(hk.get("data_exploration_update", "")).strip()
        if not isinstance(todo_new, list):
            todo_new = []
        if not isinstance(resolve_ids, list):
            resolve_ids = []
        if not todo_new and not resolve_ids and not notes_update and not data_update:
            issues.append("housekeeping_empty")

    if action == "run_command":
        if not command:
            issues.append("run_command_empty")
        if run_label in {"", "run"}:
            issues.append("run_label_generic")
        cmd_l = command.lower()
        has_repo_context = ("set-location" in cmd_l) or bool(re.search(r"\bcd\s+", cmd_l))
        if not has_repo_context:
            issues.append("run_command_missing_repo_context")

    return issues


def apply_decision_quality_gate(decision: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    issues = decision_quality_issues(decision)
    if not issues:
        return decision, [], False
    blocked = any(
        key in issues
        for key in (
            "rationale_too_short",
            "housekeeping_empty",
            "run_command_empty",
            "run_label_generic",
            "run_command_missing_repo_context",
        )
    )
    if not blocked:
        return decision, issues, False

    blocked_decision = dict(decision)
    prior_rationale = str(decision.get("rationale", "")).strip()
    blocked_decision["action"] = "wait"
    blocked_decision["command"] = ""
    blocked_decision["run_label"] = "quality_gate_wait"
    blocked_decision["rationale"] = (
        ((prior_rationale + " | ") if prior_rationale else "")
        + "quality gate blocked decision: "
        + ", ".join(issues[:6])
    )
    return blocked_decision, issues, True


def build_prompt(
    *,
    run_id: str,
    context: dict[str, Any],
    default_monitor_seconds: int,
    max_monitor_seconds: int,
    rechallenge_on_done: bool,
) -> str:
    rules = [
        "LLM is in the driver seat. No automatic picking outside your decision.",
        "Do not tune on test split.",
        "If current run looks unhealthy or blocked, prefer stop_current_run.",
        "You may choose wait when there is not enough evidence for a new action.",
        "Use shutdown_daemon only when run should end now.",
        "When helpful, you may delegate to subagents conceptually; still output one action now.",
        "Use mission/contract text as agenda and execute it step-by-step.",
        "At the start of every cycle, re-check runtime status: active_run, last_completed_run, and recent events before deciding.",
        "Re-check mission goals each cycle against mission_text; if goals and action diverge, correct course now.",
        "Re-read .llm_loop/artifacts/storyline.md and .llm_loop/artifacts/workpad.md when uncertain or stuck, then decide using already-completed work as evidence.",
        "Use mentor_context when present: if recent mentor feedback challenged your direction, address it explicitly before repeating similar commands.",
        "If mentor_context.search_requirement_unmet is true, run a web-search-supported cycle before another training-heavy command.",
        "Role separation: worker owns analysis/exploration/training execution; mentor owns critique only.",
        "Check .llm_loop/artifacts/shared_todo.md each cycle; prioritize unresolved mentor items when they are high-impact and evidence-seeking.",
        "Use .llm_loop/artifacts/workpad.md as worker-owned notes; do not repurpose mentor notes as execution logs.",
        "Maintain one structured workspace file at .llm_loop/artifacts/workpad.md.",
        "Inside workpad.md, keep sections for TODO, Notes, and Data Exploration updated with concise UTC-stamped entries.",
        "Every cycle, provide housekeeping updates in the required `housekeeping` object, even if some fields are empty.",
        "Resolve shared mentor TODOs when done by listing IDs in housekeeping.resolve_shared_todo_ids.",
        "If there is no active run and mission goals are clear, prefer run_command over wait.",
        "Operate like a data scientist, not only a hyperparameter tuner.",
        "Maintain idea diversity across categories: preprocessing, augmentation, data_sampling, loss, model_arch, optimization, evaluation.",
        "If recent runs repeat one category or metrics stagnate, choose a different category next and state why.",
        "Always set `idea_category` for your chosen action.",
        "Use a lightweight discovery flow: baseline -> inspect data -> quick online search for relevant strong approaches -> targeted experiments.",
        "Use fast-dev only for initial orientation (about 1-2 cycles) to verify pipeline and learn data behavior.",
        "After initial orientation, shift to stronger experiments and avoid lingering in tiny-budget fast-dev loops.",
        "When unresolved, do regular online research passes and adapt generic strong patterns to this task.",
        "Keep a dual objective: improve segmentation overlap and push fracture-presence classification metrics toward domain-competitive (SOTA-like) ranges.",
        "Track and discuss classification behavior explicitly (presence precision/recall and calibration), not only segmentation dice.",
        "Use online references to anchor what strong classification performance looks like in this domain, then adapt pragmatically.",
        "Avoid train-only loops: insert regular non-training cycles for deeper data exploration and literature synthesis.",
        "Treat exploration_cadence_context cadence flags as guidance, not hard gates.",
        "When research_priority is high, strongly prefer a web-supported evidence cycle before additional tuning.",
        "When non_training_priority is high, strongly prefer a non-training cycle unless there is a clear execution-critical reason not to.",
        "Treat quick data-audit scripts as first pass only; continue deeper data exploration throughout the run.",
        "In data exploration, include split/leakage checks, label quality checks, resolution/view heterogeneity, and positive-case strata analysis.",
        "Translate data findings into concrete hypotheses and experiments; do not stay in threshold/LR tuning only.",
        "Architecture probes are optional (1-2 quick probes when uncertainty is high), not mandatory before data-centric work.",
        "Shift early into data-centric exploration: include preprocessing and augmentation or data_sampling ideas before many optimizer micro-tweaks.",
        "Fast-dev settings are for scouting only; promote promising recipes to stronger budgets quickly (more epochs/batches and broader eval) when signal is flat.",
        "When breakout_priority is high, prioritize structural experiments over micro-tuning: supported backbones/models, head/decoder changes, and training-budget increases.",
        "Avoid local tuning traps by pivoting away from repeated threshold/LR-only tweaks when evidence quality is weak.",
        "When unresolved_shared_todo_count is high, address high-impact open TODOs promptly and explain deferrals explicitly in housekeeping notes.",
        "Keep tone practical and concise in rationale; avoid over-planning.",
        "Hard constraint: do not execute nnU-Net/nnUNet/nnUNetv2 pipelines here (reserved for separate manual comparison).",
        "When using internet research, extract generic strategy patterns and evidence quality signals; do not copy a turnkey pipeline verbatim.",
    ]
    if rechallenge_on_done:
        rules.append(
            "If a prior run completed and no run is active, prefer a rechallenge run_command that changes one meaningful factor."
        )

    prompt = {
        "role": "Autonomous CNN experiment driver",
        "mission": "Keep control of CNN experiments. Decide what to run, when to stop, and when to wait.",
        "dataset_domain": "X-ray fracture segmentation (medical imaging).",
        "run_id": run_id,
        "rules": rules,
        "mission_contract": {
            "worker_mission_file": context.get("mission_file"),
            "worker_mission_text": context.get("mission_text"),
        },
        "action_contract": {
            "run_command": "Provide a concrete powershell command in `command` and a short `run_label`.",
            "stop_current_run": "Stop active process tracked by wrapper.",
            "wait": "No process action this cycle.",
            "shutdown_daemon": "Create stop flag and end daemon loop.",
        },
        "artifacts_contract": {
            "storyline": ".llm_loop/artifacts/storyline.md",
            "workpad": ".llm_loop/artifacts/workpad.md",
            "shared_todo": ".llm_loop/artifacts/shared_todo.md",
            "housekeeping_output": {
                "todo_new": "0-4 concise TODO bullets to append under workpad TODO",
                "notes_update": "one concise note for workpad Notes (or empty string)",
                "data_exploration_update": "one concise observation/hypothesis for Data Exploration (or empty string)",
                "resolve_shared_todo_ids": "0-6 shared TODO IDs to mark resolved (e.g. C0002-M01)",
            },
        },
        "defaults": {
            "default_monitor_seconds": default_monitor_seconds,
            "max_monitor_seconds": max_monitor_seconds,
            "rechallenge_on_done": rechallenge_on_done,
        },
        "hard_constraints": {
            "forbidden_approaches": [
                "nnU-Net",
                "nnUNet",
                "nnUNetv2",
            ],
            "forbidden_execution_only": True,
        },
        "runtime_context": context,
    }
    return json.dumps(prompt, ensure_ascii=True)


def build_mentor_prompt(
    *,
    run_id: str,
    context: dict[str, Any],
    primary_decision: dict[str, Any],
    require_web_search: bool,
) -> str:
    review_context = {
        "time_utc": context.get("time_utc"),
        "active_run": context.get("active_run"),
        "last_completed_run": context.get("last_completed_run"),
        "exploration_context": context.get("exploration_context"),
        "performance_context": context.get("performance_context"),
        "breakout_context": context.get("breakout_context"),
        "exploration_cadence_context": context.get("exploration_cadence_context"),
        "mentor_context": context.get("mentor_context"),
        "coordination_context": context.get("coordination_context"),
        "mentor_mission_file": context.get("mentor_mission_file", context.get("mission_file")),
        "mentor_mission_text": context.get("mentor_mission_text", context.get("mission_text")),
        "workpad_text": context.get("workpad_text", ""),
        "mentor_notes_tail": context.get("mentor_notes_tail", ""),
        "shared_todo_tail": context.get("shared_todo_tail", ""),
        "storyline_tail": context.get("storyline_tail", ""),
        "recent_events_tail": context.get("recent_events_tail", ""),
    }
    rules = [
        "Act as a critical but practical mentor reviewing the primary decision.",
        "Role separation is strict: you are advisory and strategic only; do not take execution ownership.",
        "Identify when the plan is too narrow, stuck, or missing evidence.",
        "Use adaptive guidance, not rigid cadence quotas; judge by evidence quality and risk.",
        "Ask concise critical questions only when they can change the next action.",
        "Your output must be actionable for the worker and wrapper-managed notes/todo artifacts.",
        "If the decision is sound, return recommendation=continue and suggested_decision=null.",
        "If the decision is weak, return recommendation=challenge and provide a full suggested_decision.",
        "When uncertainty remains after repeated non-improving cycles, propose at least one concrete model_arch alternative with expected tradeoff.",
        "Do not propose forbidden approaches: nnU-Net / nnUNet / nnUNetv2.",
    ]
    if require_web_search:
        rules.append("Before final recommendation, run at least one web search and use it to validate or challenge the plan.")

    prompt = {
        "role": "Critical mentor for autonomous CNN experimentation loop",
        "run_id": run_id,
        "rules": rules,
        "mission_contract": {
            "mentor_mission_file": review_context.get("mentor_mission_file"),
            "mentor_mission_text": review_context.get("mentor_mission_text"),
        },
        "review_contract": {
            "recommendation": "continue or challenge",
            "critique": "short practical critique; include what is risky or missing",
            "questions": "up to 3 critical questions",
            "mentor_notes": "short note for mentor_notes.md (advisory only)",
            "todo_updates": "0-6 concrete TODO items for shared_todo.md",
            "suggested_decision": "null when continue; full decision object when challenge",
        },
        "primary_decision": primary_decision,
        "runtime_context": review_context,
    }
    return json.dumps(prompt, ensure_ascii=True)


def should_run_mentor(
    *,
    mentor_enabled: bool,
    cycle_index: int,
    mentor_every_n_cycles: int,
    mentor_force_when_stuck: bool,
    context: dict[str, Any],
) -> tuple[bool, list[str]]:
    if not mentor_enabled:
        return False, ["mentor_disabled"]
    reasons: list[str] = []
    n = max(1, int(mentor_every_n_cycles))
    if cycle_index % n == 0:
        reasons.append(f"cadence_every_{n}")
    if mentor_force_when_stuck:
        exploration_cadence = context.get("exploration_cadence_context")
        if isinstance(exploration_cadence, dict):
            research_priority = safe_float(exploration_cadence.get("research_priority"))
            non_training_priority = safe_float(exploration_cadence.get("non_training_priority"))
            todo_pressure = safe_float(exploration_cadence.get("todo_pressure"))
            if research_priority is not None and research_priority >= 0.75:
                reasons.append("research_priority_high")
            elif bool(exploration_cadence.get("research_pass_due", False)):
                reasons.append("research_pass_due")
            if non_training_priority is not None and non_training_priority >= 0.75:
                reasons.append("non_training_priority_high")
            elif bool(exploration_cadence.get("non_training_cycle_due", False)):
                reasons.append("non_training_cycle_due")
            if todo_pressure is not None and todo_pressure >= 0.70:
                reasons.append("todo_pressure_high")
        breakout_context = context.get("breakout_context")
        if isinstance(breakout_context, dict):
            breakout_priority = safe_float(breakout_context.get("breakout_priority"))
            if breakout_priority is not None and breakout_priority >= 0.75:
                reasons.append("breakout_priority_high")
            elif bool(breakout_context.get("breakout_needed", False)):
                reasons.append("breakout_needed")
        mentor_context = context.get("mentor_context")
        if isinstance(mentor_context, dict) and int(mentor_context.get("challenge_streak", 0) or 0) >= 2:
            reasons.append("mentor_challenge_streak")
    return bool(reasons), (reasons if reasons else ["not_due"])


def call_codex(
    *,
    codex_exe: str,
    workspace_root: pathlib.Path,
    thread_id_file: pathlib.Path,
    model: str,
    reasoning_effort: str,
    web_search_mode: str,
    network_access_enabled: bool,
    skip_git_repo_check: bool,
    codex_home: pathlib.Path,
    prompt: str,
    output_schema: dict[str, Any],
) -> dict[str, Any]:
    io_dir = workspace_root / ".llm_loop" / "logs" / "_codex_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    schema_path = io_dir / f"schema_{uuid.uuid4().hex}.json"
    schema_path.write_text(json.dumps(output_schema, ensure_ascii=True), encoding="utf-8")
    try:
        cmd = [
            codex_exe,
            "exec",
            "--experimental-json",
            "--model",
            model,
            "--sandbox",
            "workspace-write",
            "--cd",
            str(workspace_root),
            "--output-schema",
            str(schema_path),
        ]
        if skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        cmd.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
        cmd.extend(["--config", f'web_search="{web_search_mode}"'])
        cmd.extend(
            [
                "--config",
                "sandbox_workspace_write.network_access="
                + ("true" if network_access_enabled else "false"),
            ]
        )
        thread_id = ""
        if thread_id_file.exists():
            thread_id = thread_id_file.read_text(encoding="utf-8").strip()
        if thread_id:
            cmd.extend(["resume", thread_id])

        proc = subprocess.run(
            cmd,
            input=prompt,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            env={**os.environ, "CODEX_HOME": str(codex_home)},
        )

        parsed_result: dict[str, Any] | None = None
        seen_thread_id = ""
        turn_failed = ""
        event_errors: list[str] = []
        event_counts: Counter[str] = Counter()
        item_type_counts: Counter[str] = Counter()
        tool_signals: set[str] = set()
        used_web_search = False
        parsed_event_lines = 0

        logs_dir = workspace_root / ".llm_loop" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        trace_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trace_path = logs_dir / f"{trace_stamp}_codex_events.jsonl"
        trace_path.write_text(proc.stdout or "", encoding="utf-8")

        for raw_line in (proc.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            parsed_event_lines += 1
            evt_type = str(evt.get("type", ""))
            if evt_type:
                event_counts[evt_type] += 1
            evt_dump = json.dumps(evt, ensure_ascii=True).lower()
            if any(tok in evt_dump for tok in ("web_search", "search_query", "image_query", "internet")):
                used_web_search = True
            for mcp_name in re.findall(r"mcp__[a-z0-9_:.\\-]+", evt_dump):
                tool_signals.add(mcp_name)
            for field in ("tool_name", "name", "server_name", "connector_name"):
                v = evt.get(field)
                if isinstance(v, str) and v.strip():
                    tool_signals.add(v.strip())

            if evt_type == "thread.started":
                seen_thread_id = str(evt.get("thread_id", "")).strip()
            elif evt_type == "item.completed":
                item = evt.get("item")
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).strip()
                    if item_type:
                        item_type_counts[item_type] += 1
                    item_dump = json.dumps(item, ensure_ascii=True).lower()
                    if any(tok in item_dump for tok in ("web_search", "search_query", "image_query", "internet")):
                        used_web_search = True
                    for field in ("tool_name", "name", "server_name", "connector_name"):
                        v = item.get(field)
                        if isinstance(v, str) and v.strip():
                            tool_signals.add(v.strip())
                    if item.get("type") == "agent_message":
                        msg = item.get("text")
                        if isinstance(msg, str):
                            try:
                                obj = json.loads(msg)
                            except Exception:
                                continue
                            if isinstance(obj, dict):
                                parsed_result = obj
            elif evt_type == "turn.failed":
                err = evt.get("error")
                if isinstance(err, dict):
                    turn_failed = str(err.get("message", "")).strip()
            elif evt_type == "error":
                msg = str(evt.get("message", "")).strip()
                if msg:
                    event_errors.append(msg)

        codex_telemetry = {
            "trace_file": str(trace_path),
            "event_counts": dict(event_counts),
            "item_type_counts": dict(item_type_counts),
            "parsed_event_lines": parsed_event_lines,
            "used_web_search": used_web_search,
            "tool_signals": sorted(tool_signals)[:24],
        }

        if seen_thread_id:
            thread_id_file.parent.mkdir(parents=True, exist_ok=True)
            thread_id_file.write_text(seen_thread_id, encoding="utf-8")
        if parsed_result is not None:
            parsed_result["__codex_telemetry"] = codex_telemetry
            return parsed_result
        details = []
        if turn_failed:
            details.append("turn_failed=" + turn_failed)
        if event_errors:
            details.append("errors=" + " | ".join(event_errors[-3:]))
        if proc.stderr.strip():
            details.append("stderr=" + proc.stderr.strip())
        if proc.returncode != 0:
            details.append("exit_code=" + str(proc.returncode))
        fail_log = workspace_root / ".llm_loop" / "logs" / "codex_last_failure.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_payload = {
            "ts_utc": utc_now(),
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "turn_failed": turn_failed,
            "event_errors": event_errors,
            "codex_telemetry": codex_telemetry,
        }
        fail_log.write_text(json.dumps(fail_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        if not details:
            details.append("codex result missing or unparsable")
        raise RuntimeError("codex runner failed: " + "; ".join(details))
    finally:
        schema_path.unlink(missing_ok=True)


def collect_context(
    *,
    workspace_root: pathlib.Path,
    state: dict[str, Any],
    data_source_root: str,
    worker_mission_path: pathlib.Path,
    mentor_mission_path: pathlib.Path,
) -> dict[str, Any]:
    loop_dir = workspace_root / ".llm_loop"
    logs_dir = loop_dir / "logs"
    artifacts_dir = loop_dir / "artifacts"
    events_path = logs_dir / "events.jsonl"
    summaries_path = logs_dir / "cycle_summaries.jsonl"
    model_selection_marker = artifacts_dir / "MODEL_SELECTION_DONE.md"
    workpad_path = ensure_workpad_file(loop_dir)
    mentor_notes_path, shared_todo_path = ensure_coordination_files(loop_dir)
    storyline_path = artifacts_dir / "storyline.md"
    unresolved_shared_todo_count = count_unresolved_shared_todos(shared_todo_path)
    latest_logs = []
    for p in sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:6]:
        latest_logs.append(
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "tail": tail_text(p, max_bytes=10000)[-4000:],
            }
        )

    active = state.get("active_run") if isinstance(state.get("active_run"), dict) else None
    active_live = False
    if active and isinstance(active.get("pid"), int):
        active_live = is_pid_running(int(active["pid"]))
    if active and not active_live:
        active = None

    recent_events = read_recent_jsonl(events_path, max_lines=24)
    recent_categories: list[str] = []
    recent_training_flags: list[bool] = []
    recent_web_flags: list[bool] = []
    recent_mentor_recommendations: list[str] = []
    mentor_search_requirement_unmet = False
    last_mentor_critique = ""
    last_mentor_questions: list[str] = []
    for e in recent_events:
        cat = str(e.get("idea_category", "")).strip().lower()
        if not cat:
            cat = infer_idea_category(
                rationale=str(e.get("rationale", "")),
                command=str(e.get("command_preview", "")),
                run_label=str(e.get("run_label", "")),
            )
        recent_categories.append(cat)
        cmd_preview = str(e.get("command_preview", ""))
        run_label_hint = str(e.get("run_label", ""))
        recent_training_flags.append(is_training_command_text(cmd_preview) or is_training_command_text(run_label_hint))
        codex_info = e.get("codex")
        web_used = bool(codex_info.get("used_web_search", False)) if isinstance(codex_info, dict) else False
        recent_web_flags.append(web_used)
        mentor_info = codex_info.get("mentor") if isinstance(codex_info, dict) else None
        if isinstance(mentor_info, dict) and bool(mentor_info.get("run", False)):
            rec = str(mentor_info.get("recommendation", "")).strip().lower()
            if rec:
                recent_mentor_recommendations.append(rec)
            if bool(mentor_info.get("search_requirement_met", True)) is False:
                mentor_search_requirement_unmet = True
            critique = str(mentor_info.get("critique", "")).strip()
            if critique:
                last_mentor_critique = critique
            q = mentor_info.get("questions", [])
            if isinstance(q, list):
                questions = [str(x).strip() for x in q if str(x).strip()][:3]
                if questions:
                    last_mentor_questions = questions
    distinct_recent_categories = len({c for c in recent_categories if c})
    repeated_category_streak = 0
    if recent_categories:
        last_cat = recent_categories[-1]
        for c in reversed(recent_categories):
            if c == last_cat:
                repeated_category_streak += 1
            else:
                break

    recent_summaries = read_recent_jsonl(summaries_path, max_lines=20)
    long_summaries = read_recent_jsonl(summaries_path, max_lines=260)
    non_improving_streak = 0
    for s in reversed(recent_summaries):
        d = safe_float(s.get("delta_best_val_dice_pos"))
        if d is None:
            continue
        if d > 0:
            break
        non_improving_streak += 1
    diversify_hint = repeated_category_streak >= 3 or non_improving_streak >= 3

    cycles_with_summaries = len(long_summaries)
    best_val_dice_overall: float | None = None
    best_cycle_overall: int | None = None
    best_run_label_overall = ""
    latest_cycle: int | None = None
    latest_best_val_dice: float | None = None
    for idx, s in enumerate(long_summaries, start=1):
        cyc = int(safe_float(s.get("cycle")) or idx)
        latest_cycle = cyc
        latest_best_val_dice = safe_float(s.get("best_val_dice_pos"))
        v = latest_best_val_dice
        if v is None:
            continue
        if best_val_dice_overall is None or v > best_val_dice_overall:
            best_val_dice_overall = v
            best_cycle_overall = cyc
            best_run_label_overall = str(s.get("run_label", ""))

    cycles_since_best: int | None = None
    if best_cycle_overall is not None and latest_cycle is not None:
        cycles_since_best = max(0, latest_cycle - best_cycle_overall)

    recent_window = recent_categories[-8:] if recent_categories else []
    micro_tune_categories = {"evaluation", "optimization", "loss", "data_sampling"}
    structural_categories = {"model_arch", "preprocessing", "augmentation"}
    micro_tune_count = sum(1 for c in recent_window if c in micro_tune_categories)
    structural_count = sum(1 for c in recent_window if c in structural_categories)
    micro_tuning_drift = len(recent_window) >= 6 and micro_tune_count >= 6 and structural_count <= 2

    model_mentions: Counter[str] = Counter()
    for e in recent_events[-14:]:
        txt = " ".join(
            [
                str(e.get("run_label", "")),
                str(e.get("command_preview", "")),
                str(e.get("rationale", "")),
            ]
        ).lower()
        for model_key in ("simple_unet", "deeplabv3_resnet50", "unet_resnet34"):
            if model_key in txt:
                model_mentions[model_key] += 1
    dominant_model = ""
    dominant_model_ratio = 0.0
    if model_mentions:
        dominant_model, dom_count = model_mentions.most_common(1)[0]
        dominant_model_ratio = float(dom_count) / float(sum(model_mentions.values()))

    recent_event_window = recent_events[-8:] if recent_events else []
    recent_web_search_count = 0
    for e in recent_event_window:
        c = e.get("codex")
        if isinstance(c, dict) and bool(c.get("used_web_search", False)):
            recent_web_search_count += 1
    online_research_needed = cycles_with_summaries >= 3 and recent_web_search_count == 0
    orientation_phase_limit = 2
    orientation_phase_over = cycles_with_summaries > orientation_phase_limit

    recent_training_window = recent_training_flags[-8:] if recent_training_flags else []
    consecutive_training_runs = 0
    for tf in reversed(recent_training_window):
        if tf:
            consecutive_training_runs += 1
        else:
            break
    cycles_since_last_web_search = 0
    seen_web = False
    for wf in reversed(recent_web_flags):
        if wf:
            seen_web = True
            break
        cycles_since_last_web_search += 1
    if not seen_web:
        cycles_since_last_web_search = len(recent_web_flags)
    unresolved_todo_pressure = clamp01(float(unresolved_shared_todo_count) / 8.0)
    research_priority = clamp01(
        0.14 * float(consecutive_training_runs)
        + 0.12 * float(cycles_since_last_web_search)
        + 0.08 * float(non_improving_streak)
        + (0.10 if mentor_search_requirement_unmet else 0.0)
        + 0.06 * unresolved_todo_pressure
    )
    non_training_priority = clamp01(
        0.16 * float(consecutive_training_runs)
        + 0.08 * float(non_improving_streak)
        + 0.12 * (1.0 if micro_tuning_drift else 0.0)
        + 0.06 * unresolved_todo_pressure
    )
    research_pass_due = bool(research_priority >= 0.66)
    non_training_cycle_due = bool(non_training_priority >= 0.70)
    mentor_challenge_streak = 0
    for rec in reversed(recent_mentor_recommendations):
        if rec == "challenge":
            mentor_challenge_streak += 1
        else:
            break

    breakout_priority = clamp01(
        (0.28 if cycles_with_summaries >= 8 else 0.0)
        + 0.06 * float(non_improving_streak)
        + (0.20 if micro_tuning_drift else 0.0)
        + (0.18 if dominant_model_ratio >= 0.70 else 0.0)
        + 0.05 * float(mentor_challenge_streak)
    )
    breakout_needed = bool(breakout_priority >= 0.70)

    return {
        "time_utc": utc_now(),
        "workspace_root": str(workspace_root),
        "data_source_root": data_source_root,
        "active_run": active,
        "active_run_live": active_live,
        "last_completed_run": state.get("last_completed_run"),
        "recent_events_tail": tail_text(events_path, max_bytes=14000)[-5000:],
        "recent_logs": latest_logs,
        "mission_file": str(worker_mission_path),
        "mission_text": read_text_head(worker_mission_path, max_chars=12000),
        "mentor_mission_file": str(mentor_mission_path),
        "mentor_mission_text": read_text_head(mentor_mission_path, max_chars=12000),
        "workpad_file": str(workpad_path),
        "workpad_text": read_text_head(workpad_path, max_chars=20000),
        "mentor_notes_file": str(mentor_notes_path),
        "mentor_notes_tail": read_text_head(mentor_notes_path, max_chars=12000),
        "shared_todo_file": str(shared_todo_path),
        "shared_todo_tail": read_text_head(shared_todo_path, max_chars=12000),
        "storyline_file": str(storyline_path),
        "storyline_tail": read_text_head(storyline_path, max_chars=9000),
        "architecture_probe": {
            "optional": True,
            "marker_file": str(model_selection_marker),
            "marker_exists": model_selection_marker.exists(),
            "candidates_hint": [
                "simple_unet",
                "deeplabv3_resnet50",
                "unet_resnet34",
            ],
        },
        "exploration_context": {
            "recent_idea_categories": recent_categories[-8:],
            "distinct_recent_categories": distinct_recent_categories,
            "repeated_category_streak": repeated_category_streak,
            "non_improving_streak": non_improving_streak,
            "diversify_hint": diversify_hint,
            "target_categories": [
                "preprocessing",
                "augmentation",
                "data_sampling",
                "loss",
                "model_arch",
                "optimization",
                "evaluation",
            ],
        },
        "performance_context": {
            "cycles_with_summaries": cycles_with_summaries,
            "latest_cycle": latest_cycle,
            "latest_best_val_dice_pos": latest_best_val_dice,
            "best_val_dice_pos_overall": best_val_dice_overall,
            "best_cycle_overall": best_cycle_overall,
            "best_run_label_overall": best_run_label_overall,
            "cycles_since_best": cycles_since_best,
            "orientation_phase_limit_cycles": orientation_phase_limit,
            "orientation_phase_over": orientation_phase_over,
            "recent_web_search_count_8": recent_web_search_count,
            "online_research_needed": online_research_needed,
        },
        "breakout_context": {
            "breakout_needed": breakout_needed,
            "breakout_priority": breakout_priority,
            "non_improving_streak": non_improving_streak,
            "micro_tuning_drift": micro_tuning_drift,
            "recent_micro_tune_count": micro_tune_count,
            "recent_structural_count": structural_count,
            "dominant_model_hint": dominant_model,
            "dominant_model_ratio": dominant_model_ratio,
            "suggested_structural_moves": [
                "larger supported backbone or architecture variant",
                "explicit head/decoder modification",
                "training budget increase (epochs/max_train_batches/max_eval_batches)",
                "data-centric preprocessing/augmentation redesign",
            ],
        },
        "exploration_cadence_context": {
            "recent_training_runs_8": int(sum(1 for x in recent_training_window if x)),
            "consecutive_training_runs": consecutive_training_runs,
            "cycles_since_last_web_search": cycles_since_last_web_search,
            "research_pass_due": research_pass_due,
            "non_training_cycle_due": non_training_cycle_due,
            "research_priority": research_priority,
            "non_training_priority": non_training_priority,
            "todo_pressure": unresolved_todo_pressure,
            "recommended_non_training_actions": [
                "deeper dataset quality/split analysis and update workpad Data Exploration section",
                "online literature/pattern scan summarized in workpad Notes with explicit takeaways",
                "error analysis on recent outputs to identify dominant failure mode",
            ],
        },
        "mentor_context": {
            "recent_recommendations": recent_mentor_recommendations[-6:],
            "challenge_streak": mentor_challenge_streak,
            "search_requirement_unmet": mentor_search_requirement_unmet,
            "last_critique": last_mentor_critique,
            "last_questions": last_mentor_questions,
            "unresolved_shared_todo_count": unresolved_shared_todo_count,
        },
        "coordination_context": {
            "worker_owned_workpad": str(workpad_path),
            "mentor_owned_notes": str(mentor_notes_path),
            "shared_todo": str(shared_todo_path),
            "unresolved_shared_todo_count": unresolved_shared_todo_count,
        },
    }


def run_new_command(
    *,
    workspace_root: pathlib.Path,
    command: str,
    run_label: str,
    monitor_seconds: int,
    cycle_poll_seconds: int,
    stop_flags: list[pathlib.Path],
    stop_patterns: list[str],
    state: dict[str, Any],
    state_file: pathlib.Path,
    events_path: pathlib.Path,
) -> dict[str, Any]:
    logs_dir = workspace_root / ".llm_loop" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in run_label.lower()).strip("-")
    if not safe_label:
        safe_label = "run"
    run_tag = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_label}"
    stdout_path = logs_dir / f"{run_tag}.stdout.log"
    stderr_path = logs_dir / f"{run_tag}.stderr.log"
    out_f = stdout_path.open("w", encoding="utf-8", buffering=1)
    err_f = stderr_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        ["pwsh.exe", "-NoProfile", "-Command", command],
        cwd=str(workspace_root),
        stdout=out_f,
        stderr=err_f,
        text=True,
    )
    state["active_run"] = {
        "pid": proc.pid,
        "run_label": run_label,
        "command": command,
        "started_utc": utc_now(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    write_json(state_file, state)

    started = time.time()
    killed_reason = ""
    stop_patterns_lc = [p.lower() for p in stop_patterns if p.strip()]
    while True:
        rc = proc.poll()
        if rc is not None:
            out_f.close()
            err_f.close()
            state["active_run"] = None
            write_json(state_file, state)
            return {
                "status": "completed",
                "exit_code": rc,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }

        for flag in stop_flags:
            if flag.exists():
                killed_reason = f"stop_flag:{flag.name}"
                break
        if killed_reason:
            break

        if stop_patterns_lc:
            stdout_tail = tail_text(stdout_path, max_bytes=12000).lower()
            stderr_tail = tail_text(stderr_path, max_bytes=12000).lower()
            for patt in stop_patterns_lc:
                if patt and (patt in stdout_tail or patt in stderr_tail):
                    killed_reason = f"pattern:{patt}"
                    break
        if killed_reason:
            break

        if (time.time() - started) >= monitor_seconds:
            break
        time.sleep(max(1, cycle_poll_seconds))

    if killed_reason:
        try:
            proc.kill()
        except Exception:
            pass
        out_f.close()
        err_f.close()
        state["active_run"] = None
        write_json(state_file, state)
        append_jsonl(
            events_path,
            {
                "ts_utc": utc_now(),
                "event": "run_killed",
                "pid": proc.pid,
                "reason": killed_reason,
                "run_label": run_label,
            },
        )
        return {
            "status": "killed",
            "reason": killed_reason,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    return {
        "status": "monitor_window_elapsed",
        "pid": proc.pid,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def normalize_package_map(raw: Any, defaults: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {str(k).strip().lower(): [str(x).strip() for x in v if str(x).strip()] for k, v in defaults.items()}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        k = str(key).strip().lower()
        if not k:
            continue
        pkgs: list[str] = []
        if isinstance(val, list):
            pkgs = [str(x).strip() for x in val if str(x).strip()]
        elif isinstance(val, str):
            pkgs = [str(x).strip() for x in val.split() if str(x).strip()]
        if pkgs:
            out[k] = pkgs
    return out


def normalize_fallback_map(raw: Any, defaults: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {str(k).strip().lower(): str(v).strip() for k, v in defaults.items() if str(k).strip() and str(v).strip()}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        k = str(key).strip().lower()
        v = str(val).strip()
        if k and v:
            out[k] = v
    return out


def normalize_clone_map(raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        k = str(key).strip().lower()
        if not k:
            continue
        repo = ""
        ref = ""
        subdir = ""
        editable = True
        if isinstance(val, str):
            repo = val.strip()
        elif isinstance(val, dict):
            repo = str(val.get("repo", "")).strip()
            ref = str(val.get("ref", "")).strip()
            subdir = str(val.get("subdir", "")).strip()
            editable = bool(val.get("editable", True))
        if not repo:
            continue
        out[k] = {"repo": repo, "ref": ref, "subdir": subdir, "editable": editable}
    return out


def extract_python_exe_from_command(command: str) -> str | None:
    if not command:
        return None
    for m in re.finditer(r"['\"]([A-Za-z]:[\\/][^'\"]*python(?:\.exe)?)['\"]", command, flags=re.IGNORECASE):
        candidate = m.group(1).strip()
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    return None


def extract_missing_module(stderr_text: str) -> str | None:
    if not stderr_text:
        return None
    m = re.search(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", stderr_text)
    if not m:
        return None
    return str(m.group(1)).strip()


def extract_unsupported_model_name(stderr_text: str) -> str | None:
    if not stderr_text:
        return None
    m = re.search(r"Unsupported model\.name:\s*([A-Za-z0-9_\-\.]+)", stderr_text)
    if not m:
        return None
    return str(m.group(1)).strip()


def infer_direct_module_packages(missing_module: str, alias_map: dict[str, list[str]]) -> list[str]:
    if not missing_module:
        return []
    raw = missing_module.strip()
    if not raw:
        return []
    root = raw.split(".")[0].strip()
    key = root.lower()
    if key in alias_map:
        return [str(x).strip() for x in alias_map[key] if str(x).strip()]
    candidates: list[str] = []
    for c in [raw, root, root.replace("_", "-"), raw.replace(".", "-")]:
        c = str(c).strip()
        if c and c not in candidates:
            candidates.append(c)
    return candidates


def apply_model_name_fallback(command: str, unsupported_model: str, fallback_map: dict[str, str]) -> str | None:
    if not command or not unsupported_model:
        return None
    key = unsupported_model.strip().lower()
    repl = fallback_map.get(key)
    if not repl:
        return None
    patched = command.replace(unsupported_model, repl)
    if patched != command:
        return patched
    patched_ci = re.sub(rf"\b{re.escape(unsupported_model)}\b", repl, command, flags=re.IGNORECASE)
    if patched_ci != command:
        return patched_ci
    return None


def extract_command_workdir(command: str) -> pathlib.Path | None:
    if not command:
        return None
    m = re.search(r"(?:Set-Location|cd)\s+['\"]([^'\"]+)['\"]", command, flags=re.IGNORECASE)
    if not m:
        return None
    txt = str(m.group(1)).strip()
    if not txt:
        return None
    try:
        p = pathlib.Path(txt)
        if p.exists():
            return p.resolve()
        return p
    except Exception:
        return None


def extract_config_path_token(command: str) -> tuple[str, str] | None:
    if not command:
        return None
    m = re.search(r"--config\s+(?:['\"]([^'\"]+)['\"]|(\S+))", command, flags=re.IGNORECASE)
    if not m:
        return None
    raw = str(m.group(1) or m.group(2) or "").strip()
    if not raw:
        return None
    return raw, raw


def patch_model_name_in_yaml_config(
    *,
    config_path: pathlib.Path,
    unsupported_model: str,
    replacement_model: str,
    workspace_root: pathlib.Path,
    run_label: str,
) -> pathlib.Path | None:
    if not config_path.exists():
        return None
    txt = config_path.read_text(encoding="utf-8", errors="ignore")
    if not txt.strip():
        return None
    pattern = re.compile(
        rf"(?mi)^(\s*name\s*:\s*)(['\"]?){re.escape(unsupported_model)}(['\"]?)\s*$"
    )
    patched_txt, changed = pattern.subn(rf"\1\2{replacement_model}\3", txt, count=1)
    if changed <= 0:
        return None

    out_dir = workspace_root / ".llm_loop" / "autofix_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in run_label.lower()).strip("-") or "run"
    safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in replacement_model.lower()).strip("-") or "model"
    out_name = f"{ts}_{safe_label}_{config_path.stem}_autofix_{safe_model}{config_path.suffix or '.yaml'}"
    out_path = out_dir / out_name
    out_path.write_text(patched_txt, encoding="utf-8")
    return out_path


def apply_model_name_fallback_to_config_command(
    *,
    workspace_root: pathlib.Path,
    command: str,
    run_label: str,
    unsupported_model: str,
    fallback_map: dict[str, str],
) -> str | None:
    key = unsupported_model.strip().lower()
    replacement_model = str(fallback_map.get(key, "")).strip()
    if not replacement_model:
        return None
    token = extract_config_path_token(command)
    if not token:
        return None
    config_arg_raw, config_arg_replace = token
    command_workdir = extract_command_workdir(command) or workspace_root
    cfg_path = pathlib.Path(config_arg_raw)
    if not cfg_path.is_absolute():
        cfg_path = (command_workdir / cfg_path).resolve()
    patched_cfg = patch_model_name_in_yaml_config(
        config_path=cfg_path,
        unsupported_model=unsupported_model,
        replacement_model=replacement_model,
        workspace_root=workspace_root,
        run_label=run_label,
    )
    if not patched_cfg:
        return None
    patched_command = command.replace(config_arg_replace, str(patched_cfg), 1)
    if patched_command == command:
        return None
    return patched_command


def clone_source_repo(
    *,
    workspace_root: pathlib.Path,
    source_key: str,
    clone_spec: dict[str, Any],
    run_label: str,
) -> dict[str, Any]:
    logs_dir = workspace_root / ".llm_loop" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in run_label.lower()).strip("-") or "run"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clone_log = logs_dir / f"{ts}_{safe_label}_autofix_clone.log"

    repo = str(clone_spec.get("repo", "")).strip()
    ref = str(clone_spec.get("ref", "")).strip()
    subdir = str(clone_spec.get("subdir", "")).strip()
    if not repo:
        return {
            "attempted": False,
            "success": False,
            "error": "missing_repo",
        }

    safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in source_key.lower()).strip("-") or "source"
    sources_root = workspace_root / ".llm_loop" / "model_sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    source_dir = sources_root / safe_key
    install_dir = source_dir / subdir if subdir else source_dir

    if source_dir.exists() and not (source_dir / ".git").exists():
        clone_log.write_text(
            "\n".join(
                [
                    f"repo={repo}",
                    f"ref={ref}",
                    f"source_dir={source_dir}",
                    "error=source_dir_exists_without_git_repo",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "success": False,
            "repo": repo,
            "ref": ref,
            "source_dir": str(source_dir),
            "install_dir": str(install_dir),
            "log_file": str(clone_log),
            "error": "source_dir_exists_without_git_repo",
        }

    commands: list[list[str]] = []
    if (source_dir / ".git").exists():
        commands.append(["git", "-C", str(source_dir), "fetch", "--all", "--prune"])
        if ref:
            commands.append(["git", "-C", str(source_dir), "checkout", ref])
            commands.append(["git", "-C", str(source_dir), "pull", "--ff-only", "origin", ref])
    else:
        clone_cmd = ["git", "clone", "--depth", "1"]
        if ref:
            clone_cmd.extend(["--branch", ref])
        clone_cmd.extend([repo, str(source_dir)])
        commands.append(clone_cmd)

    started = utc_now()
    logs: list[str] = [f"started_utc={started}", f"repo={repo}", f"ref={ref}", f"source_dir={source_dir}"]
    for cmd in commands:
        logs.append("")
        logs.append("command=" + " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=900,
            )
            logs.append("[stdout]")
            logs.append(proc.stdout or "")
            logs.append("[stderr]")
            logs.append(proc.stderr or "")
            logs.append(f"returncode={proc.returncode}")
            if proc.returncode != 0:
                clone_log.write_text("\n".join(logs), encoding="utf-8")
                return {
                    "attempted": True,
                    "success": False,
                    "repo": repo,
                    "ref": ref,
                    "source_dir": str(source_dir),
                    "install_dir": str(install_dir),
                    "log_file": str(clone_log),
                    "exit_code": int(proc.returncode),
                }
        except Exception as exc:
            logs.append(f"exception={exc}")
            clone_log.write_text("\n".join(logs), encoding="utf-8")
            return {
                "attempted": True,
                "success": False,
                "repo": repo,
                "ref": ref,
                "source_dir": str(source_dir),
                "install_dir": str(install_dir),
                "log_file": str(clone_log),
                "error": str(exc),
                "exit_code": -1,
            }

    if not install_dir.exists():
        logs.append(f"error=install_dir_missing:{install_dir}")
        clone_log.write_text("\n".join(logs), encoding="utf-8")
        return {
            "attempted": True,
            "success": False,
            "repo": repo,
            "ref": ref,
            "source_dir": str(source_dir),
            "install_dir": str(install_dir),
            "log_file": str(clone_log),
            "error": "install_dir_missing",
        }
    logs.append(f"ended_utc={utc_now()}")
    clone_log.write_text("\n".join(logs), encoding="utf-8")
    return {
        "attempted": True,
        "success": True,
        "repo": repo,
        "ref": ref,
        "source_dir": str(source_dir),
        "install_dir": str(install_dir),
        "log_file": str(clone_log),
        "ended_utc": utc_now(),
    }


def install_packages_with_python(
    *,
    workspace_root: pathlib.Path,
    python_exe: str,
    packages: list[str],
    run_label: str,
) -> dict[str, Any]:
    logs_dir = workspace_root / ".llm_loop" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in run_label.lower()).strip("-") or "run"
    install_tag = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_label}_autofix_pip"
    install_log = logs_dir / f"{install_tag}.log"

    cmd = [python_exe, "-m", "pip", "install", "--disable-pip-version-check", *packages]
    started = utc_now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=900,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        install_log.write_text(
            "\n".join(
                [
                    f"started_utc={started}",
                    f"python_exe={python_exe}",
                    "command=" + " ".join(cmd),
                    "",
                    "[stdout]",
                    stdout,
                    "",
                    "[stderr]",
                    stderr,
                ]
            ),
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "success": proc.returncode == 0,
            "exit_code": int(proc.returncode),
            "python_exe": python_exe,
            "packages": packages,
            "log_file": str(install_log),
            "started_utc": started,
            "ended_utc": utc_now(),
        }
    except Exception as exc:
        install_log.write_text(
            "\n".join(
                [
                    f"started_utc={started}",
                    f"python_exe={python_exe}",
                    "command=" + " ".join(cmd),
                    "",
                    f"exception={exc}",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "success": False,
            "exit_code": -1,
            "python_exe": python_exe,
            "packages": packages,
            "log_file": str(install_log),
            "started_utc": started,
            "ended_utc": utc_now(),
            "error": str(exc),
        }


def install_source_with_python(
    *,
    workspace_root: pathlib.Path,
    python_exe: str,
    source_dir: pathlib.Path,
    editable: bool,
    run_label: str,
) -> dict[str, Any]:
    packages: list[str] = []
    if editable:
        packages.extend(["-e", str(source_dir)])
    else:
        packages.append(str(source_dir))
    return install_packages_with_python(
        workspace_root=workspace_root,
        python_exe=python_exe,
        packages=packages,
        run_label=run_label,
    )


def attempt_auto_repair(
    *,
    workspace_root: pathlib.Path,
    command: str,
    run_label: str,
    stderr_log: pathlib.Path,
    fallback_python_exe: str,
    module_package_map: dict[str, list[str]],
    module_alias_map: dict[str, list[str]],
    module_clone_map: dict[str, dict[str, Any]],
    model_package_map: dict[str, list[str]],
    model_clone_map: dict[str, dict[str, Any]],
    model_fallback_map: dict[str, str],
    allow_direct_module_install: bool,
) -> dict[str, Any] | None:
    stderr_tail = tail_text(stderr_log, max_bytes=24000)
    if not stderr_tail:
        return None

    missing_module = extract_missing_module(stderr_tail)
    unsupported_model = extract_unsupported_model_name(stderr_tail)
    packages: list[str] = []
    reason = ""
    clone_spec: dict[str, Any] | None = None
    model_patched_command: str | None = None
    model_fallback_applied = False

    if missing_module:
        key = missing_module.strip().lower()
        reason = f"missing_module:{missing_module}"
        packages = module_package_map.get(key, [])
        if not packages and allow_direct_module_install:
            packages = infer_direct_module_packages(missing_module, module_alias_map)
        clone_spec = module_clone_map.get(key)
    elif unsupported_model:
        key = unsupported_model.strip().lower()
        reason = f"unsupported_model:{unsupported_model}"
        model_patched_command = apply_model_name_fallback(command, unsupported_model, model_fallback_map)
        if not model_patched_command:
            model_patched_command = apply_model_name_fallback_to_config_command(
                workspace_root=workspace_root,
                command=command,
                run_label=run_label,
                unsupported_model=unsupported_model,
                fallback_map=model_fallback_map,
            )
        model_fallback_applied = bool(model_patched_command and model_patched_command != command)
        if not model_fallback_applied:
            packages = model_package_map.get(key, [])
            clone_spec = model_clone_map.get(key)
    else:
        return None

    python_exe = extract_python_exe_from_command(command) or (fallback_python_exe.strip() if fallback_python_exe else "")
    python_exe = python_exe.strip()
    if python_exe and not pathlib.Path(python_exe).exists():
        python_exe = ""

    out: dict[str, Any] = {
        "reason": reason or "unknown",
        "missing_module": missing_module or "",
        "unsupported_model": unsupported_model or "",
        "packages": packages,
        "python_exe": python_exe,
        "pip": {"attempted": False, "success": False},
        "clone": {"attempted": False, "success": False},
        "pip_from_clone": {"attempted": False, "success": False},
        "model_fallback_applied": model_fallback_applied,
    }
    if model_fallback_applied and model_patched_command:
        out["patched_command"] = model_patched_command
        out["skip_install_reason"] = "model_fallback_available"
        return out

    if python_exe and packages:
        out["pip"] = install_packages_with_python(
            workspace_root=workspace_root,
            python_exe=python_exe,
            packages=packages,
            run_label=run_label,
        )

    clone_needed = bool(clone_spec) and bool(python_exe) and not bool((out.get("pip") or {}).get("success", False))
    if clone_needed and isinstance(clone_spec, dict):
        clone = clone_source_repo(
            workspace_root=workspace_root,
            source_key=missing_module or unsupported_model or "source",
            clone_spec=clone_spec,
            run_label=run_label,
        )
        out["clone"] = clone
        if bool(clone.get("success", False)):
            install_dir_txt = str(clone.get("install_dir", "")).strip()
            install_dir = pathlib.Path(install_dir_txt) if install_dir_txt else None
            if install_dir and install_dir.exists():
                out["pip_from_clone"] = install_source_with_python(
                    workspace_root=workspace_root,
                    python_exe=python_exe,
                    source_dir=install_dir,
                    editable=bool(clone_spec.get("editable", True)),
                    run_label=run_label,
                )

    attempted_any = (
        bool((out.get("pip") or {}).get("attempted", False))
        or bool((out.get("clone") or {}).get("attempted", False))
        or bool((out.get("pip_from_clone") or {}).get("attempted", False))
        or bool(out.get("model_fallback_applied", False))
    )
    if not attempted_any:
        return None
    return out


def safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def format_num(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def extract_stdout_log(entry: Any) -> pathlib.Path | None:
    if not isinstance(entry, dict):
        return None
    direct = entry.get("stdout_log")
    if isinstance(direct, str) and direct.strip():
        p = pathlib.Path(direct.strip())
        if p.exists():
            return p
    outcome = entry.get("outcome")
    if isinstance(outcome, dict):
        out = outcome.get("stdout_log")
        if isinstance(out, str) and out.strip():
            p = pathlib.Path(out.strip())
            if p.exists():
                return p
    return None


def extract_epoch_rows(stdout_log: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in tail_text(stdout_log, max_bytes=400000).splitlines():
        line = raw.strip()
        idx = line.find("{'epoch':")
        if idx < 0:
            continue
        line = line[idx:]
        end_idx = line.rfind("}")
        if end_idx > 0:
            line = line[: end_idx + 1]
        try:
            obj = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def best_and_latest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not rows:
        return None, None
    latest = max(rows, key=lambda r: int(safe_float(r.get("epoch")) or 0))
    best = max(rows, key=lambda r: safe_float(r.get("val_dice_pos")) or -1.0)
    return best, latest


def metric_short(row: dict[str, Any] | None) -> str:
    if not row:
        return "n/a"
    return (
        "epoch="
        + str(int(safe_float(row.get("epoch")) or 0))
        + ", val_dice_pos="
        + format_num(safe_float(row.get("val_dice_pos")))
        + ", precision_presence="
        + format_num(safe_float(row.get("val_precision_presence")))
        + ", recall_presence="
        + format_num(safe_float(row.get("val_recall_presence")))
        + ", n="
        + str(int(safe_float(row.get("val_num_samples")) or 0))
    )


def build_quality_evidence(
    *,
    cycle_result: dict[str, Any],
    previous_last_completed: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    run_outcome = cycle_result.get("run_outcome") if isinstance(cycle_result.get("run_outcome"), dict) else {}
    current_log = None
    if isinstance(run_outcome, dict):
        out = run_outcome.get("stdout_log")
        if isinstance(out, str) and out.strip():
            p = pathlib.Path(out.strip())
            if p.exists():
                current_log = p
    if current_log is None:
        current_log = extract_stdout_log(state.get("last_completed_run"))
    prev_log = extract_stdout_log(previous_last_completed)

    current_rows = extract_epoch_rows(current_log) if current_log else []
    prev_rows = extract_epoch_rows(prev_log) if prev_log else []
    current_best, current_latest = best_and_latest(current_rows)
    prev_best, _ = best_and_latest(prev_rows)

    current_best_dice = safe_float(current_best.get("val_dice_pos")) if current_best else None
    prev_best_dice = safe_float(prev_best.get("val_dice_pos")) if prev_best else None
    delta = None
    if current_best_dice is not None and prev_best_dice is not None:
        delta = current_best_dice - prev_best_dice

    return {
        "run_outcome": run_outcome,
        "current_log": str(current_log) if current_log else "",
        "previous_log": str(prev_log) if prev_log else "",
        "current_rows_count": len(current_rows),
        "previous_rows_count": len(prev_rows),
        "current_best": current_best,
        "current_latest": current_latest,
        "previous_best": prev_best,
        "current_best_dice": current_best_dice,
        "previous_best_dice": prev_best_dice,
        "delta_best_dice": delta,
    }


def write_cycle_summary(
    *,
    workspace_root: pathlib.Path,
    cycle_index: int,
    cycle_result: dict[str, Any],
    run_label: str,
    monitor_seconds: int,
    previous_last_completed: Any,
    state: dict[str, Any],
    codex_telemetry: dict[str, Any],
) -> pathlib.Path:
    now_utc = utc_now()
    evidence = build_quality_evidence(
        cycle_result=cycle_result,
        previous_last_completed=previous_last_completed,
        state=state,
    )
    run_outcome = evidence["run_outcome"] if isinstance(evidence["run_outcome"], dict) else {}
    current_best = evidence["current_best"] if isinstance(evidence["current_best"], dict) else None
    current_latest = evidence["current_latest"] if isinstance(evidence["current_latest"], dict) else None
    current_best_dice = safe_float(evidence.get("current_best_dice"))
    prev_best_dice = safe_float(evidence.get("previous_best_dice"))
    delta = safe_float(evidence.get("delta_best_dice"))
    delta_txt = f"{delta:+.3f}" if delta is not None else "n/a"

    run_status = "n/a"
    if isinstance(run_outcome, dict):
        run_status = str(run_outcome.get("status", "n/a"))
        if run_status == "completed":
            run_status += ", exit_code=" + str(run_outcome.get("exit_code", "n/a"))
        if run_status == "killed":
            run_status += ", reason=" + str(run_outcome.get("reason", "n/a"))
    active_run = state.get("active_run")
    active_txt = "none"
    if isinstance(active_run, dict):
        active_txt = "pid=" + str(active_run.get("pid", "n/a")) + ", label=" + str(active_run.get("run_label", "n/a"))

    rationale = str(cycle_result.get("rationale", "")).strip()
    if len(rationale) > 280:
        rationale = rationale[:277] + "..."

    event_counts = codex_telemetry.get("event_counts", {}) if isinstance(codex_telemetry, dict) else {}
    parsed_events = int(codex_telemetry.get("parsed_event_lines", 0)) if isinstance(codex_telemetry, dict) else 0
    used_web_search = bool(codex_telemetry.get("used_web_search", False)) if isinstance(codex_telemetry, dict) else False
    tool_signals = codex_telemetry.get("tool_signals", []) if isinstance(codex_telemetry, dict) else []
    if not isinstance(tool_signals, list):
        tool_signals = []
    tool_short = ", ".join(str(x) for x in tool_signals[:4]) if tool_signals else "none"
    trace_file = str(codex_telemetry.get("trace_file", "n/a")) if isinstance(codex_telemetry, dict) else "n/a"

    summaries_log = workspace_root / ".llm_loop" / "logs" / "cycle_summaries.jsonl"
    append_jsonl(
        summaries_log,
        {
            "ts_utc": now_utc,
            "cycle": cycle_index,
            "action": cycle_result.get("action"),
            "result": cycle_result.get("result"),
            "run_label": run_label,
            "best_val_dice_pos": current_best_dice,
            "prev_best_val_dice_pos": prev_best_dice,
            "delta_best_val_dice_pos": delta,
            "rows_parsed": int(evidence.get("current_rows_count", 0)),
            "used_web_search": used_web_search,
            "codex_event_counts": event_counts,
            "tool_signals": tool_signals[:12],
        },
    )
    return summaries_log


def write_storyline(
    *,
    workspace_root: pathlib.Path,
    cycle_index: int,
    cycle_result: dict[str, Any],
    run_label: str,
    monitor_seconds: int,
    previous_last_completed: Any,
    state: dict[str, Any],
    codex_telemetry: dict[str, Any],
) -> pathlib.Path:
    now_utc = utc_now()
    evidence = build_quality_evidence(
        cycle_result=cycle_result,
        previous_last_completed=previous_last_completed,
        state=state,
    )
    current_best = evidence["current_best"] if isinstance(evidence["current_best"], dict) else None
    current_latest = evidence["current_latest"] if isinstance(evidence["current_latest"], dict) else None
    delta = safe_float(evidence.get("delta_best_dice"))
    run_outcome = evidence["run_outcome"] if isinstance(evidence["run_outcome"], dict) else {}

    rationale = str(cycle_result.get("rationale", "")).strip()
    if len(rationale) > 300:
        rationale = rationale[:297] + "..."
    command_preview = str(cycle_result.get("command_preview", "")).strip()
    if len(command_preview) > 220:
        command_preview = command_preview[:217] + "..."

    parsed_events = int(codex_telemetry.get("parsed_event_lines", 0)) if isinstance(codex_telemetry, dict) else 0
    used_web_search = bool(codex_telemetry.get("used_web_search", False)) if isinstance(codex_telemetry, dict) else False
    tool_signals = codex_telemetry.get("tool_signals", []) if isinstance(codex_telemetry, dict) else []
    if not isinstance(tool_signals, list):
        tool_signals = []
    tool_short = ", ".join(str(x) for x in tool_signals[:5]) if tool_signals else "none"
    mentor_info = {}
    worker_hk_info = {}
    quality_gate_info = {}
    codex_info = cycle_result.get("codex")
    if isinstance(codex_info, dict) and isinstance(codex_info.get("mentor"), dict):
        mentor_info = codex_info.get("mentor") or {}
    if isinstance(codex_info, dict) and isinstance(codex_info.get("worker_housekeeping"), dict):
        worker_hk_info = codex_info.get("worker_housekeeping") or {}
    if isinstance(cycle_result.get("quality_gate"), dict):
        quality_gate_info = cycle_result.get("quality_gate") or {}

    outcome_txt = "n/a"
    if isinstance(run_outcome, dict):
        outcome_txt = str(run_outcome.get("status", "n/a"))
        if "exit_code" in run_outcome:
            outcome_txt += f" (exit={run_outcome.get('exit_code')})"
    delta_txt = f"{delta:+.3f}" if delta is not None else "n/a"
    next_txt = "continue rechallenge loop" if str(cycle_result.get("result", "")) == "run_command" else "await next signal"
    mentor_txt = "mentor=skipped"
    if mentor_info:
        mentor_txt = (
            "mentor="
            + ("run" if bool(mentor_info.get("run", False)) else "not_due")
            + ", rec="
            + (str(mentor_info.get("recommendation", "")).strip() or "n/a")
            + ", applied="
            + str(bool(mentor_info.get("applied", False))).lower()
            + ", search_ok="
            + str(bool(mentor_info.get("search_requirement_met", True))).lower()
            + ", todo_added="
            + str(int(mentor_info.get("shared_todo_added", 0) or 0))
            + ", todo_open="
            + str(int(mentor_info.get("shared_todo_unresolved", 0) or 0))
        )
    worker_hk_txt = "todo_added=0, notes=0, data=0, resolved=0, shared_open=0"
    if worker_hk_info:
        worker_hk_txt = (
            "todo_added="
            + str(int(worker_hk_info.get("todo_added", 0) or 0))
            + ", notes="
            + str(int(worker_hk_info.get("notes_added", 0) or 0))
            + ", data="
            + str(int(worker_hk_info.get("data_exploration_added", 0) or 0))
            + ", resolved="
            + str(int(worker_hk_info.get("shared_todo_resolved", 0) or 0))
            + ", shared_open="
            + str(int(worker_hk_info.get("shared_todo_unresolved", 0) or 0))
        )
    quality_gate_txt = "source=primary, blocked=false, issues=none"
    if quality_gate_info:
        issues = quality_gate_info.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        issue_txt = ", ".join(str(x) for x in issues[:4]) if issues else "none"
        quality_gate_txt = (
            "source="
            + str(quality_gate_info.get("source", "primary"))
            + ", blocked="
            + str(bool(quality_gate_info.get("blocked", False))).lower()
            + ", issues="
            + issue_txt
        )

    lines = [
        f"## Cycle {cycle_index:04d} | {now_utc}",
        f"1. Situation: active_run={str(state.get('active_run') is not None).lower()}, last_result={cycle_result.get('result', 'n/a')}",
        f"2. Decision: action={cycle_result.get('action', 'n/a')}, category={cycle_result.get('idea_category', 'n/a')}, run_label={run_label}, why={rationale or 'n/a'}",
        f"3. Quality gate: {quality_gate_txt}",
        f"4. Execution: outcome={outcome_txt}, monitor_seconds={monitor_seconds}, command={command_preview or 'n/a'}",
        f"5. Data evidence: rows_parsed={evidence.get('current_rows_count', 0)}, best={metric_short(current_best)}, latest={metric_short(current_latest)}, delta_vs_prev={delta_txt}",
        f"6. LLM evidence: parsed_events={parsed_events}, used_web_search={str(used_web_search).lower()}, tools={tool_short}",
        f"7. Mentor: {mentor_txt}",
        f"8. Worker housekeeping: {worker_hk_txt}",
        f"9. Next checkpoint: {next_txt}",
        "",
    ]

    artifacts_dir = workspace_root / ".llm_loop" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    storyline_file = artifacts_dir / "storyline.md"
    if not storyline_file.exists():
        storyline_file.write_text("# LLM Work Storyline\n\n", encoding="utf-8")
    with storyline_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return storyline_file


def main() -> None:
    p = argparse.ArgumentParser(description="Single LLM control cycle for llm_driven_cnns.")
    p.add_argument("--workspace-root", required=True)
    p.add_argument("--config-path", required=True)
    p.add_argument("--codex-exe", required=True)
    p.add_argument("--codex-home", required=True)
    p.add_argument("--thread-id-file", required=True)
    p.add_argument("--state-file", required=True)
    p.add_argument("--events-file", required=True)
    p.add_argument("--stop-daemon-flag", required=True)
    p.add_argument("--stop-current-run-flag", required=True)
    args = p.parse_args()

    workspace_root = pathlib.Path(args.workspace_root).resolve()
    config_path = pathlib.Path(args.config_path).resolve()
    codex_exe = str(args.codex_exe).strip()
    codex_home = pathlib.Path(args.codex_home).resolve()
    if codex_exe.lower().endswith(".ps1"):
        maybe_cmd = str(pathlib.Path(codex_exe).with_suffix(".cmd"))
        if pathlib.Path(maybe_cmd).exists():
            codex_exe = maybe_cmd
    if codex_exe and not codex_exe.lower().endswith(".cmd"):
        maybe_cmd = pathlib.Path(codex_exe + ".cmd")
        if maybe_cmd.exists():
            codex_exe = str(maybe_cmd)
    thread_id_file = pathlib.Path(args.thread_id_file).resolve()
    state_file = pathlib.Path(args.state_file).resolve()
    events_file = pathlib.Path(args.events_file).resolve()
    stop_daemon_flag = pathlib.Path(args.stop_daemon_flag).resolve()
    stop_current_run_flag = pathlib.Path(args.stop_current_run_flag).resolve()

    cfg = read_json(config_path, {})
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid config JSON")

    run_id = str(cfg.get("run_id", "llm_driven_cnns_day01"))
    model = str(cfg.get("model", "gpt-5.3-codex"))
    reasoning_effort = str(cfg.get("reasoning_effort", "high"))
    web_search_mode = str(cfg.get("web_search_mode", "live"))
    network_access_enabled = bool(cfg.get("network_access_enabled", True))
    skip_git_repo_check = bool(cfg.get("skip_git_repo_check", True))
    cycle_poll_seconds = int(cfg.get("cycle_poll_seconds", 5))
    default_monitor_seconds = int(cfg.get("default_monitor_seconds", 300))
    max_monitor_seconds = int(cfg.get("max_monitor_seconds", 1800))
    rechallenge_on_done = bool(cfg.get("rechallenge_on_done", True))
    data_source_root = str(cfg.get("data_source_root", ""))
    mission_file = str(cfg.get("mission_file", "AGENTS.md")).strip() or "AGENTS.md"
    worker_mission_file = str(cfg.get("worker_mission_file", mission_file)).strip() or mission_file
    mentor_mission_file = str(cfg.get("mentor_mission_file", mission_file)).strip() or mission_file
    force_bootstrap_if_idle = bool(cfg.get("force_bootstrap_if_idle", False))
    bootstrap_command = str(cfg.get("bootstrap_command", "")).strip()
    bootstrap_label = str(cfg.get("bootstrap_label", "bootstrap")).strip() or "bootstrap"
    mentor_enabled = bool(cfg.get("mentor_enabled", False))
    mentor_model = str(cfg.get("mentor_model", model)).strip() or model
    mentor_reasoning_effort = str(cfg.get("mentor_reasoning_effort", reasoning_effort)).strip() or reasoning_effort
    mentor_web_search_mode = str(cfg.get("mentor_web_search_mode", web_search_mode)).strip() or web_search_mode
    mentor_network_access_enabled = bool(cfg.get("mentor_network_access_enabled", network_access_enabled))
    mentor_force_when_stuck = bool(cfg.get("mentor_force_when_stuck", True))
    mentor_apply_suggestions = bool(cfg.get("mentor_apply_suggestions", True))
    mentor_require_web_search = bool(cfg.get("mentor_require_web_search", True))
    auto_repair_enabled = bool(cfg.get("auto_repair_enabled", True))
    auto_repair_retry_on_success = bool(cfg.get("auto_repair_retry_on_success", True))
    auto_repair_allow_direct_module_install = bool(cfg.get("auto_repair_allow_direct_module_install", True))
    auto_repair_python_exe = str(cfg.get("auto_repair_python_exe", "")).strip()
    auto_repair_module_package_map = normalize_package_map(
        cfg.get("auto_repair_module_package_map"),
        DEFAULT_AUTO_REPAIR_MODULE_PACKAGE_MAP,
    )
    auto_repair_module_alias_map = normalize_package_map(
        cfg.get("auto_repair_module_alias_map"),
        DEFAULT_AUTO_REPAIR_MODULE_ALIAS_MAP,
    )
    auto_repair_module_clone_map = normalize_clone_map(
        cfg.get("auto_repair_module_clone_map"),
    )
    auto_repair_model_package_map = normalize_package_map(
        cfg.get("auto_repair_model_package_map"),
        DEFAULT_AUTO_REPAIR_MODEL_PACKAGE_MAP,
    )
    auto_repair_model_clone_map = normalize_clone_map(
        cfg.get("auto_repair_model_clone_map"),
    )
    auto_repair_model_fallback_map = normalize_fallback_map(
        cfg.get("auto_repair_model_fallback_map"),
        DEFAULT_AUTO_REPAIR_MODEL_FALLBACK_MAP,
    )
    try:
        mentor_every_n_cycles = int(cfg.get("mentor_every_n_cycles", 2))
    except Exception:
        mentor_every_n_cycles = 2
    mentor_every_n_cycles = max(1, mentor_every_n_cycles)
    worker_mission_path = pathlib.Path(worker_mission_file)
    if not worker_mission_path.is_absolute():
        worker_mission_path = workspace_root / worker_mission_path
    mentor_mission_path = pathlib.Path(mentor_mission_file)
    if not mentor_mission_path.is_absolute():
        mentor_mission_path = workspace_root / mentor_mission_path
    if not mentor_mission_path.exists():
        mentor_mission_path = worker_mission_path
    codex_home.mkdir(parents=True, exist_ok=True)
    loop_dir = workspace_root / ".llm_loop"
    mentor_notes_path, shared_todo_path = ensure_coordination_files(loop_dir)

    state = read_json(state_file, {"active_run": None})
    if not isinstance(state, dict):
        state = {"active_run": None}
    active = state.get("active_run")
    if isinstance(active, dict) and isinstance(active.get("pid"), int):
        if not is_pid_running(int(active["pid"])):
            state["active_run"] = None

    context = collect_context(
        workspace_root=workspace_root,
        state=state,
        data_source_root=data_source_root,
        worker_mission_path=worker_mission_path,
        mentor_mission_path=mentor_mission_path,
    )
    cycle_index = count_jsonl_rows(events_file) + 1
    mentor_due, mentor_reasons = should_run_mentor(
        mentor_enabled=mentor_enabled,
        cycle_index=cycle_index,
        mentor_every_n_cycles=mentor_every_n_cycles,
        mentor_force_when_stuck=mentor_force_when_stuck,
        context=context,
    )
    context["bootstrap_command"] = bootstrap_command
    context["bootstrap_label"] = bootstrap_label
    context["force_bootstrap_if_idle"] = force_bootstrap_if_idle
    prompt = build_prompt(
        run_id=run_id,
        context=context,
        default_monitor_seconds=default_monitor_seconds,
        max_monitor_seconds=max_monitor_seconds,
        rechallenge_on_done=rechallenge_on_done,
    )

    decision_raw = call_codex(
        codex_exe=codex_exe,
        workspace_root=workspace_root,
        thread_id_file=thread_id_file,
        model=model,
        reasoning_effort=reasoning_effort,
        web_search_mode=web_search_mode,
        network_access_enabled=network_access_enabled,
        skip_git_repo_check=skip_git_repo_check,
        codex_home=codex_home,
        prompt=prompt,
        output_schema=build_output_schema(),
    )
    codex_telemetry = decision_raw.pop("__codex_telemetry", {}) if isinstance(decision_raw, dict) else {}
    primary_decision = coerce_decision(
        decision_raw,
        default_monitor_seconds=default_monitor_seconds,
        max_monitor_seconds=max_monitor_seconds,
    )
    primary_quality_issues: list[str] = []
    primary_quality_blocked = False
    primary_decision, primary_quality_issues, primary_quality_blocked = apply_decision_quality_gate(primary_decision)

    mentor_thread_id_file = thread_id_file.with_name("codex_mentor_thread_id.txt")
    mentor_telemetry: dict[str, Any] = {}
    mentor_error = ""
    mentor_recommendation = ""
    mentor_critique = ""
    mentor_questions: list[str] = []
    mentor_notes_txt = ""
    mentor_todo_updates: list[str] = []
    mentor_search_requirement_met = True
    mentor_applied = False
    effective_decision = dict(primary_decision)
    effective_quality_source = "primary"
    effective_quality_issues = list(primary_quality_issues)
    effective_quality_blocked = bool(primary_quality_blocked)

    if mentor_due:
        try:
            mentor_prompt = build_mentor_prompt(
                run_id=run_id,
                context=context,
                primary_decision=primary_decision,
                require_web_search=mentor_require_web_search,
            )
            mentor_out = call_codex(
                codex_exe=codex_exe,
                workspace_root=workspace_root,
                thread_id_file=mentor_thread_id_file,
                model=mentor_model,
                reasoning_effort=mentor_reasoning_effort,
                web_search_mode=mentor_web_search_mode,
                network_access_enabled=mentor_network_access_enabled,
                skip_git_repo_check=skip_git_repo_check,
                codex_home=codex_home,
                prompt=mentor_prompt,
                output_schema=build_mentor_output_schema(),
            )
            mentor_telemetry = mentor_out.pop("__codex_telemetry", {}) if isinstance(mentor_out, dict) else {}
            mentor_review = mentor_out if isinstance(mentor_out, dict) else {}
            mentor_recommendation = str(mentor_review.get("recommendation", "")).strip().lower()
            mentor_critique = str(mentor_review.get("critique", "")).strip()
            q_raw = mentor_review.get("questions", [])
            if isinstance(q_raw, list):
                mentor_questions = [str(x).strip() for x in q_raw if str(x).strip()][:3]
            mentor_notes_txt = str(mentor_review.get("mentor_notes", "")).strip()
            todo_raw = mentor_review.get("todo_updates", [])
            if isinstance(todo_raw, list):
                mentor_todo_updates = [str(x).strip() for x in todo_raw if str(x).strip()][:6]
            if mentor_require_web_search and mentor_network_access_enabled:
                mentor_search_requirement_met = bool(mentor_telemetry.get("used_web_search", False))
            if mentor_apply_suggestions and mentor_search_requirement_met and mentor_recommendation == "challenge":
                suggested_raw = mentor_review.get("suggested_decision")
                if isinstance(suggested_raw, dict):
                    suggested = coerce_decision(
                        suggested_raw,
                        default_monitor_seconds=default_monitor_seconds,
                        max_monitor_seconds=max_monitor_seconds,
                    )
                    suggested, suggested_quality_issues, suggested_quality_blocked = apply_decision_quality_gate(suggested)
                    # Protect execution from malformed mentor suggestions.
                    if not (suggested["action"] == "run_command" and not suggested["command"]):
                        effective_decision = suggested
                        mentor_applied = True
                        effective_quality_source = "mentor_suggested"
                        effective_quality_issues = list(suggested_quality_issues)
                        effective_quality_blocked = bool(suggested_quality_blocked)
                    else:
                        mentor_error = "mentor challenge ignored: suggested run_command without command"
                else:
                    mentor_error = "mentor challenge ignored: suggested_decision missing"
        except Exception as exc:
            mentor_error = str(exc)
    if mentor_due and mentor_require_web_search and mentor_network_access_enabled and not mentor_search_requirement_met:
        if mentor_error:
            mentor_error += " | "
        mentor_error += "mentor search requirement not met"

    mentor_todo_added = 0
    unresolved_shared_todo_count = count_unresolved_shared_todos(shared_todo_path)
    if mentor_due:
        if mentor_error and not mentor_notes_txt:
            mentor_notes_txt = "mentor_error: " + mentor_error
        mentor_todo_added, unresolved_shared_todo_count = append_mentor_coordination_artifacts(
            mentor_notes_path=mentor_notes_path,
            shared_todo_path=shared_todo_path,
            cycle_index=cycle_index,
            mentor_recommendation=mentor_recommendation or "n/a",
            mentor_critique=mentor_critique,
            mentor_questions=mentor_questions,
            mentor_notes=mentor_notes_txt,
            todo_updates=mentor_todo_updates,
        )

    action = str(effective_decision.get("action", "wait"))
    command = str(effective_decision.get("command", "")).strip()
    run_label = str(effective_decision.get("run_label", "run")).strip() or "run"
    idea_category = str(effective_decision.get("idea_category", "")).strip().lower()
    rationale = str(effective_decision.get("rationale", "")).strip()
    monitor_seconds = int(effective_decision.get("monitor_seconds", default_monitor_seconds))
    stop_patterns = effective_decision.get("stop_patterns", [])
    if not isinstance(stop_patterns, list):
        stop_patterns = []
    stop_patterns = [str(x) for x in stop_patterns]
    worker_housekeeping = effective_decision.get("housekeeping", {})
    if not isinstance(worker_housekeeping, dict):
        worker_housekeeping = {}
    worker_housekeeping_result = append_worker_housekeeping_artifacts(
        workpad_path=loop_dir / "artifacts" / "workpad.md",
        shared_todo_path=shared_todo_path,
        cycle_index=cycle_index,
        housekeeping=worker_housekeeping,
    )
    unresolved_shared_todo_count = int(worker_housekeeping_result.get("shared_todo_unresolved", unresolved_shared_todo_count))

    previous_last_completed = state.get("last_completed_run")
    cycle_result: dict[str, Any] = {
        "ts_utc": utc_now(),
        "action": action,
        "idea_category": idea_category,
        "rationale": rationale,
        "decision_source": "mentor" if mentor_applied else "primary",
        "quality_gate": {
            "source": effective_quality_source,
            "blocked": bool(effective_quality_blocked),
            "issues": effective_quality_issues[:8],
        },
        "codex": {
            "used_web_search": bool(codex_telemetry.get("used_web_search", False)),
            "parsed_event_lines": int(codex_telemetry.get("parsed_event_lines", 0)) if isinstance(codex_telemetry, dict) else 0,
            "trace_file": str(codex_telemetry.get("trace_file", "")) if isinstance(codex_telemetry, dict) else "",
            "tool_signals": codex_telemetry.get("tool_signals", []) if isinstance(codex_telemetry, dict) else [],
            "mentor": {
                "enabled": mentor_enabled,
                "run": mentor_due,
                "reasons": mentor_reasons,
                "recommendation": mentor_recommendation,
                "applied": mentor_applied,
                "critique": mentor_critique,
                "questions": mentor_questions,
                "require_web_search": mentor_require_web_search,
                "search_requirement_met": mentor_search_requirement_met,
                "used_web_search": bool(mentor_telemetry.get("used_web_search", False)) if isinstance(mentor_telemetry, dict) else False,
                "parsed_event_lines": int(mentor_telemetry.get("parsed_event_lines", 0)) if isinstance(mentor_telemetry, dict) else 0,
                "trace_file": str(mentor_telemetry.get("trace_file", "")) if isinstance(mentor_telemetry, dict) else "",
                "tool_signals": mentor_telemetry.get("tool_signals", []) if isinstance(mentor_telemetry, dict) else [],
                "mentor_notes_file": str(mentor_notes_path),
                "shared_todo_file": str(shared_todo_path),
                "shared_todo_added": mentor_todo_added,
                "shared_todo_unresolved": unresolved_shared_todo_count,
                "error": mentor_error,
            },
            "worker_housekeeping": {
                "todo_added": int(worker_housekeeping_result.get("todo_added", 0)),
                "notes_added": int(worker_housekeeping_result.get("notes_added", 0)),
                "data_exploration_added": int(worker_housekeeping_result.get("data_exploration_added", 0)),
                "shared_todo_resolved": int(worker_housekeeping_result.get("shared_todo_resolved", 0)),
                "shared_todo_unresolved": int(worker_housekeeping_result.get("shared_todo_unresolved", unresolved_shared_todo_count)),
            },
        },
    }

    active = state.get("active_run")
    bootstrap_attempts = int(state.get("bootstrap_attempts", 0))
    bootstrap_successful = bool(state.get("bootstrap_successful", False))
    no_active_live = not (
        isinstance(active, dict)
        and isinstance(active.get("pid"), int)
        and is_pid_running(int(active["pid"]))
    )
    if (
        force_bootstrap_if_idle
        and action == "wait"
        and no_active_live
        and not bootstrap_successful
        and bootstrap_attempts < 3
        and bootstrap_command
    ):
        action = "run_command"
        command = bootstrap_command
        run_label = bootstrap_label
        monitor_seconds = max(monitor_seconds, default_monitor_seconds)
        cycle_result["action"] = action
        cycle_result["rationale"] = (
            (rationale + " | ") if rationale else ""
        ) + "forced bootstrap on initial idle cycle"
        cycle_result["forced_bootstrap"] = True
        state["bootstrap_attempted"] = True
        state["bootstrap_attempts"] = bootstrap_attempts + 1
        cycle_result["idea_category"] = infer_idea_category(
            rationale=cycle_result.get("rationale", ""),
            command=command,
            run_label=run_label,
        )

    if command:
        cycle_result["command_preview"] = command[:260]
    final_record_run_label = run_label
    final_record_command = command

    if action == "shutdown_daemon":
        stop_daemon_flag.parent.mkdir(parents=True, exist_ok=True)
        stop_daemon_flag.touch()
        cycle_result["result"] = "daemon_stop_requested"
    elif action == "stop_current_run":
        if isinstance(active, dict) and isinstance(active.get("pid"), int) and is_pid_running(int(active["pid"])):
            ok = kill_pid(int(active["pid"]))
            if ok:
                state["last_completed_run"] = {
                    "ended_utc": utc_now(),
                    "status": "stopped_by_action",
                    "run_label": active.get("run_label"),
                    "command": active.get("command"),
                    "stdout_log": active.get("stdout_log"),
                    "stderr_log": active.get("stderr_log"),
                }
            state["active_run"] = None
            cycle_result["result"] = "stop_current_run"
            cycle_result["stopped"] = bool(ok)
        else:
            state["active_run"] = None
            cycle_result["result"] = "no_active_run"
    elif action == "run_command":
        if not command:
            cycle_result["result"] = "invalid_empty_command"
        else:
            blocked, blocked_token = detect_forbidden_command_usage(command)
            if blocked:
                cycle_result["result"] = "blocked_forbidden_approach"
                cycle_result["blocked_token"] = blocked_token
                cycle_result["rationale"] = (
                    (rationale + " | ") if rationale else ""
                ) + f"blocked forbidden approach token: {blocked_token}"
            elif isinstance(active, dict) and isinstance(active.get("pid"), int) and is_pid_running(int(active["pid"])):
                cycle_result["result"] = "active_run_exists"
                cycle_result["active_pid"] = int(active["pid"])
            else:
                initial_outcome = run_new_command(
                    workspace_root=workspace_root,
                    command=command,
                    run_label=run_label,
                    monitor_seconds=monitor_seconds,
                    cycle_poll_seconds=cycle_poll_seconds,
                    stop_flags=[stop_daemon_flag, stop_current_run_flag],
                    stop_patterns=stop_patterns,
                    state=state,
                    state_file=state_file,
                    events_path=events_file,
                )
                run_outcome = dict(initial_outcome)
                retry_command = ""
                retry_run_label = ""
                status_initial = str(initial_outcome.get("status", ""))
                exit_initial = initial_outcome.get("exit_code")
                if auto_repair_enabled and status_initial == "completed" and exit_initial not in (None, 0):
                    stderr_log_txt = str(initial_outcome.get("stderr_log", "")).strip()
                    stderr_log_path = pathlib.Path(stderr_log_txt) if stderr_log_txt else None
                    if stderr_log_path and stderr_log_path.exists():
                        repair = attempt_auto_repair(
                            workspace_root=workspace_root,
                            command=command,
                            run_label=run_label,
                            stderr_log=stderr_log_path,
                            fallback_python_exe=auto_repair_python_exe,
                            module_package_map=auto_repair_module_package_map,
                            module_alias_map=auto_repair_module_alias_map,
                            module_clone_map=auto_repair_module_clone_map,
                            model_package_map=auto_repair_model_package_map,
                            model_clone_map=auto_repair_model_clone_map,
                            model_fallback_map=auto_repair_model_fallback_map,
                            allow_direct_module_install=auto_repair_allow_direct_module_install,
                        )
                        if isinstance(repair, dict):
                            cycle_result["auto_repair"] = repair
                            if bool(repair.get("model_fallback_applied", False)) and isinstance(repair.get("patched_command"), str):
                                retry_command = str(repair.get("patched_command", "")).strip()
                                retry_run_label = (run_label + "_retry_model_fallback").strip()
                            elif (
                                bool((repair.get("pip") or {}).get("attempted", False))
                                and bool((repair.get("pip") or {}).get("success", False))
                            ) or (
                                bool((repair.get("pip_from_clone") or {}).get("attempted", False))
                                and bool((repair.get("pip_from_clone") or {}).get("success", False))
                            ):
                                retry_command = command
                                retry_run_label = (run_label + "_retry_after_install").strip()
                if auto_repair_retry_on_success and retry_command and retry_run_label:
                    retry_outcome = run_new_command(
                        workspace_root=workspace_root,
                        command=retry_command,
                        run_label=retry_run_label,
                        monitor_seconds=monitor_seconds,
                        cycle_poll_seconds=cycle_poll_seconds,
                        stop_flags=[stop_daemon_flag, stop_current_run_flag],
                        stop_patterns=stop_patterns,
                        state=state,
                        state_file=state_file,
                        events_path=events_file,
                    )
                    cycle_result["run_outcome_initial"] = initial_outcome
                    cycle_result["run_outcome_retry"] = retry_outcome
                    run_outcome = dict(retry_outcome)
                    final_record_run_label = retry_run_label
                    final_record_command = retry_command
                cycle_result["result"] = "run_command"
                cycle_result["run_outcome"] = run_outcome
                status = str(run_outcome.get("status", ""))
                exit_code = run_outcome.get("exit_code")
                if status in {"completed", "killed"}:
                    state["last_completed_run"] = {
                        "ended_utc": utc_now(),
                        "status": status,
                        "run_label": final_record_run_label,
                        "command": final_record_command,
                        "outcome": run_outcome,
                    }
                    if final_record_run_label == bootstrap_label:
                        state["bootstrap_successful"] = bool(status == "completed" and exit_code == 0)
    else:
        cycle_result["result"] = "wait"

    try:
        summary_file = write_cycle_summary(
            workspace_root=workspace_root,
            cycle_index=cycle_index,
            cycle_result=cycle_result,
            run_label=final_record_run_label,
            monitor_seconds=monitor_seconds,
            previous_last_completed=previous_last_completed,
            state=state,
            codex_telemetry=codex_telemetry if isinstance(codex_telemetry, dict) else {},
        )
        cycle_result["summary_log"] = str(summary_file)
    except Exception as exc:
        cycle_result["summary_error"] = str(exc)
    try:
        storyline_file = write_storyline(
            workspace_root=workspace_root,
            cycle_index=cycle_index,
            cycle_result=cycle_result,
            run_label=final_record_run_label,
            monitor_seconds=monitor_seconds,
            previous_last_completed=previous_last_completed,
            state=state,
            codex_telemetry=codex_telemetry if isinstance(codex_telemetry, dict) else {},
        )
        cycle_result["storyline_file"] = str(storyline_file)
    except Exception as exc:
        cycle_result["storyline_error"] = str(exc)
    write_json(state_file, state)
    append_jsonl(events_file, cycle_result)
    print(json.dumps(cycle_result, ensure_ascii=True))


if __name__ == "__main__":
    main()
