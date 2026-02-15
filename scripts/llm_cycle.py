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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def build_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "rationale", "command", "run_label", "idea_category", "monitor_seconds", "stop_patterns"],
        "properties": {
            "action": {"type": "string", "enum": ["run_command", "stop_current_run", "wait", "shutdown_daemon"]},
            "rationale": {"type": "string"},
            "command": {"type": "string"},
            "run_label": {"type": "string"},
            "idea_category": {
                "type": "string",
                "enum": [
                    "augmentation",
                    "preprocessing",
                    "data_sampling",
                    "loss",
                    "model_arch",
                    "optimization",
                    "evaluation",
                    "other",
                ],
            },
            "monitor_seconds": {"type": "integer", "minimum": 15, "maximum": 7200},
            "stop_patterns": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string"},
            },
        },
    }


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
        "Maintain one structured workspace file at .llm_loop/artifacts/workpad.md.",
        "Inside workpad.md, keep sections for TODO, Notes, and Data Exploration updated with concise UTC-stamped entries.",
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
        "If exploration_cadence_context.research_pass_due is true, do web search in this decision cycle before choosing the next command.",
        "If exploration_cadence_context.non_training_cycle_due is true, choose a non-training run_command this cycle (no scripts/train.py).",
        "Treat quick data-audit scripts as first pass only; continue deeper data exploration throughout the run.",
        "In data exploration, include split/leakage checks, label quality checks, resolution/view heterogeneity, and positive-case strata analysis.",
        "Translate data findings into concrete hypotheses and experiments; do not stay in threshold/LR tuning only.",
        "Architecture probes are optional (1-2 quick probes when uncertainty is high), not mandatory before data-centric work.",
        "Shift early into data-centric exploration: include preprocessing and augmentation or data_sampling ideas before many optimizer micro-tweaks.",
        "Fast-dev settings are for scouting only; promote promising recipes to stronger budgets quickly (more epochs/batches and broader eval) when signal is flat.",
        "When breakout_context.breakout_needed is true, prioritize structural experiments over micro-tuning: larger supported backbones/models, head/decoder changes, and training-budget increases.",
        "Avoid local tuning traps: when breakout_context.breakout_needed is true, do not spend more than two consecutive cycles on threshold-only or LR-only tweaks.",
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
        "action_contract": {
            "run_command": "Provide a concrete powershell command in `command` and a short `run_label`.",
            "stop_current_run": "Stop active process tracked by wrapper.",
            "wait": "No process action this cycle.",
            "shutdown_daemon": "Create stop flag and end daemon loop.",
        },
        "artifacts_contract": {
            "storyline": ".llm_loop/artifacts/storyline.md",
            "workpad": ".llm_loop/artifacts/workpad.md",
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
    mission_path: pathlib.Path,
) -> dict[str, Any]:
    loop_dir = workspace_root / ".llm_loop"
    logs_dir = loop_dir / "logs"
    artifacts_dir = loop_dir / "artifacts"
    events_path = logs_dir / "events.jsonl"
    summaries_path = logs_dir / "cycle_summaries.jsonl"
    model_selection_marker = artifacts_dir / "MODEL_SELECTION_DONE.md"
    workpad_path = ensure_workpad_file(loop_dir)
    storyline_path = artifacts_dir / "storyline.md"
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
    research_pass_due = (consecutive_training_runs >= 3 and cycles_since_last_web_search >= 2) or cycles_since_last_web_search >= 4
    non_training_cycle_due = consecutive_training_runs >= 4

    breakout_needed = False
    if cycles_with_summaries >= 8:
        if non_improving_streak >= 5:
            breakout_needed = True
        if micro_tuning_drift:
            breakout_needed = True
        if dominant_model_ratio >= 0.70 and non_improving_streak >= 3:
            breakout_needed = True

    return {
        "time_utc": utc_now(),
        "workspace_root": str(workspace_root),
        "data_source_root": data_source_root,
        "active_run": active,
        "active_run_live": active_live,
        "last_completed_run": state.get("last_completed_run"),
        "recent_events_tail": tail_text(events_path, max_bytes=14000)[-5000:],
        "recent_logs": latest_logs,
        "mission_file": str(mission_path),
        "mission_text": read_text_head(mission_path, max_chars=12000),
        "workpad_file": str(workpad_path),
        "workpad_text": read_text_head(workpad_path, max_chars=20000),
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
            "recommended_non_training_actions": [
                "deeper dataset quality/split analysis and update workpad Data Exploration section",
                "online literature/pattern scan summarized in workpad Notes with explicit takeaways",
                "error analysis on recent outputs to identify dominant failure mode",
            ],
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

    started = time.time()
    killed_reason = ""
    stop_patterns_lc = [p.lower() for p in stop_patterns if p.strip()]
    while True:
        rc = proc.poll()
        if rc is not None:
            out_f.close()
            err_f.close()
            state["active_run"] = None
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

    outcome_txt = "n/a"
    if isinstance(run_outcome, dict):
        outcome_txt = str(run_outcome.get("status", "n/a"))
        if "exit_code" in run_outcome:
            outcome_txt += f" (exit={run_outcome.get('exit_code')})"
    delta_txt = f"{delta:+.3f}" if delta is not None else "n/a"
    next_txt = "continue rechallenge loop" if str(cycle_result.get("result", "")) == "run_command" else "await next signal"

    lines = [
        f"## Cycle {cycle_index:04d} | {now_utc}",
        f"1. Situation: active_run={str(state.get('active_run') is not None).lower()}, last_result={cycle_result.get('result', 'n/a')}",
        f"2. Decision: action={cycle_result.get('action', 'n/a')}, category={cycle_result.get('idea_category', 'n/a')}, run_label={run_label}, why={rationale or 'n/a'}",
        f"3. Execution: outcome={outcome_txt}, monitor_seconds={monitor_seconds}, command={command_preview or 'n/a'}",
        f"4. Data evidence: rows_parsed={evidence.get('current_rows_count', 0)}, best={metric_short(current_best)}, latest={metric_short(current_latest)}, delta_vs_prev={delta_txt}",
        f"5. LLM evidence: parsed_events={parsed_events}, used_web_search={str(used_web_search).lower()}, tools={tool_short}",
        f"6. Next checkpoint: {next_txt}",
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
    force_bootstrap_if_idle = bool(cfg.get("force_bootstrap_if_idle", False))
    bootstrap_command = str(cfg.get("bootstrap_command", "")).strip()
    bootstrap_label = str(cfg.get("bootstrap_label", "bootstrap")).strip() or "bootstrap"
    mission_path = pathlib.Path(mission_file)
    if not mission_path.is_absolute():
        mission_path = workspace_root / mission_path
    codex_home.mkdir(parents=True, exist_ok=True)

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
        mission_path=mission_path,
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

    decision = call_codex(
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
    codex_telemetry = decision.pop("__codex_telemetry", {}) if isinstance(decision, dict) else {}

    action = str(decision.get("action", "wait"))
    command = str(decision.get("command", "")).strip()
    run_label = str(decision.get("run_label", "run")).strip() or "run"
    idea_category = str(decision.get("idea_category", "")).strip().lower()
    if not idea_category:
        idea_category = infer_idea_category(rationale=str(decision.get("rationale", "")), command=command, run_label=run_label)
    monitor_seconds = int(decision.get("monitor_seconds", default_monitor_seconds))
    monitor_seconds = max(15, min(max_monitor_seconds, monitor_seconds))
    stop_patterns = decision.get("stop_patterns", [])
    if not isinstance(stop_patterns, list):
        stop_patterns = []
    stop_patterns = [str(x) for x in stop_patterns]
    rationale = str(decision.get("rationale", "")).strip()

    previous_last_completed = state.get("last_completed_run")
    cycle_result: dict[str, Any] = {
        "ts_utc": utc_now(),
        "action": action,
        "idea_category": idea_category,
        "rationale": rationale,
        "codex": {
            "used_web_search": bool(codex_telemetry.get("used_web_search", False)),
            "parsed_event_lines": int(codex_telemetry.get("parsed_event_lines", 0)) if isinstance(codex_telemetry, dict) else 0,
            "trace_file": str(codex_telemetry.get("trace_file", "")) if isinstance(codex_telemetry, dict) else "",
            "tool_signals": codex_telemetry.get("tool_signals", []) if isinstance(codex_telemetry, dict) else [],
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
                run_outcome = run_new_command(
                    workspace_root=workspace_root,
                    command=command,
                    run_label=run_label,
                    monitor_seconds=monitor_seconds,
                    cycle_poll_seconds=cycle_poll_seconds,
                    stop_flags=[stop_daemon_flag, stop_current_run_flag],
                    stop_patterns=stop_patterns,
                    state=state,
                    events_path=events_file,
                )
                cycle_result["result"] = "run_command"
                cycle_result["run_outcome"] = run_outcome
                status = str(run_outcome.get("status", ""))
                exit_code = run_outcome.get("exit_code")
                if status in {"completed", "killed"}:
                    state["last_completed_run"] = {
                        "ended_utc": utc_now(),
                        "status": status,
                        "run_label": run_label,
                        "command": command,
                        "outcome": run_outcome,
                    }
                    if run_label == bootstrap_label:
                        state["bootstrap_successful"] = bool(status == "completed" and exit_code == 0)
    else:
        cycle_result["result"] = "wait"

    cycle_index = count_jsonl_rows(events_file) + 1
    try:
        summary_file = write_cycle_summary(
            workspace_root=workspace_root,
            cycle_index=cycle_index,
            cycle_result=cycle_result,
            run_label=run_label,
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
            run_label=run_label,
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
