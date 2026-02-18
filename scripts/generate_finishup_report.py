from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Any


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def read_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_metrics_rows(metrics_path: pathlib.Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    txt = metrics_path.read_text(encoding="utf-8", errors="ignore")
    if not txt.strip():
        return []

    rows: list[dict[str, Any]] = []
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            rows = [r for r in obj if isinstance(r, dict)]
        elif isinstance(obj, dict):
            history = obj.get("history")
            if isinstance(history, list):
                rows = [r for r in history if isinstance(r, dict)]
            else:
                rows = [obj]
        if rows:
            return rows
    except Exception:
        pass

    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(line)
                break
            except Exception:
                continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def extract_model_name_from_yaml(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_model = False
    model_indent = 0
    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if re.match(r"^\s*model\s*:\s*$", line):
            in_model = True
            model_indent = indent
            continue
        if in_model:
            if indent <= model_indent:
                in_model = False
                continue
            m = re.match(r"^\s*name\s*:\s*['\"]?([A-Za-z0-9_\-\.]+)['\"]?\s*$", line)
            if m:
                return str(m.group(1)).strip()
    return ""


def collect_run_record(run_dir: pathlib.Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    rows = parse_metrics_rows(metrics_path)
    if not rows:
        return None

    model_name = extract_model_name_from_yaml(run_dir / "resolved_config.yaml")
    if not model_name:
        model_name = extract_model_name_from_yaml(run_dir / "config.yaml")
    if not model_name:
        model_name = "unknown"

    best_dice = None
    best_ap = None
    best_auc = None
    best_precision = None
    best_recall = None
    best_dice_epoch = None
    latest_epoch = None
    latest_row: dict[str, Any] | None = None

    for row in rows:
        epoch = safe_int(row.get("epoch"))
        if epoch is not None and (latest_epoch is None or epoch >= latest_epoch):
            latest_epoch = epoch
            latest_row = row
        dice = safe_float(row.get("val_dice_pos"))
        if dice is not None and (best_dice is None or dice > best_dice):
            best_dice = dice
            best_dice_epoch = epoch
            best_ap = safe_float(row.get("val_average_precision_presence"))
            best_auc = safe_float(row.get("val_roc_auc_presence"))
            best_precision = safe_float(row.get("val_precision_presence"))
            best_recall = safe_float(row.get("val_recall_presence"))

    latest_dice = safe_float((latest_row or {}).get("val_dice_pos"))
    latest_ap = safe_float((latest_row or {}).get("val_average_precision_presence"))
    latest_auc = safe_float((latest_row or {}).get("val_roc_auc_presence"))
    latest_precision = safe_float((latest_row or {}).get("val_precision_presence"))
    latest_recall = safe_float((latest_row or {}).get("val_recall_presence"))

    status = "ok"
    if not (run_dir / "best_model.pt").exists():
        status = "no_best_checkpoint"

    return {
        "run_label": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "rows_parsed": len(rows),
        "status": status,
        "best": {
            "epoch": best_dice_epoch,
            "val_dice_pos": best_dice,
            "val_average_precision_presence": best_ap,
            "val_roc_auc_presence": best_auc,
            "val_precision_presence": best_precision,
            "val_recall_presence": best_recall,
        },
        "latest": {
            "epoch": latest_epoch,
            "val_dice_pos": latest_dice,
            "val_average_precision_presence": latest_ap,
            "val_roc_auc_presence": latest_auc,
            "val_precision_presence": latest_precision,
            "val_recall_presence": latest_recall,
        },
    }


def rank_score(entry: dict[str, Any]) -> float:
    best = entry.get("best", {}) if isinstance(entry.get("best"), dict) else {}
    dice = safe_float(best.get("val_dice_pos")) or 0.0
    ap = safe_float(best.get("val_average_precision_presence")) or 0.0
    auc = safe_float(best.get("val_roc_auc_presence")) or 0.0
    return 1000.0 * dice + 100.0 * ap + 10.0 * auc


def infer_repo_root(workspace_root: pathlib.Path, explicit_repo_root: str) -> pathlib.Path:
    if explicit_repo_root.strip():
        return pathlib.Path(explicit_repo_root).resolve()
    cfg = read_json(workspace_root / "config" / "daemon_config.json", {})
    if isinstance(cfg, dict):
        data_root = str(cfg.get("data_source_root", "")).strip()
        if data_root:
            p = pathlib.Path(data_root)
            if p.exists():
                return p.resolve().parent
    sibling = workspace_root.parent / "xray_fracture_benchmark"
    if sibling.exists():
        return sibling.resolve()
    return workspace_root.resolve()


def collect_cycle_summary_rows(workspace_root: pathlib.Path) -> list[dict[str, Any]]:
    path = workspace_root / ".llm_loop" / "logs" / "cycle_summaries.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def collect_failure_rows(workspace_root: pathlib.Path) -> list[dict[str, Any]]:
    path = workspace_root / ".llm_loop" / "logs" / "events.jsonl"
    if not path.exists():
        return []
    failures: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        if not isinstance(evt, dict):
            continue
        out = evt.get("run_outcome")
        if not isinstance(out, dict):
            continue
        ec = out.get("exit_code")
        if not isinstance(ec, int) or ec == 0:
            continue
        reason = ""
        stderr_log = str(out.get("stderr_log", "")).strip()
        if stderr_log:
            p = pathlib.Path(stderr_log)
            if p.exists():
                tail = p.read_text(encoding="utf-8", errors="ignore")[-8000:]
                for pat in [
                    r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
                    r"Unsupported model\.name:\s*([A-Za-z0-9_\-\.]+)",
                    r"Object of type .* not JSON serializable",
                    r"metrics missing or empty",
                    r"RuntimeError:[^\n]*",
                    r"ValueError:[^\n]*",
                ]:
                    m = re.search(pat, tail, flags=re.IGNORECASE)
                    if m:
                        reason = str(m.group(1) if m.groups() else m.group(0)).strip()
                        break
        failures.append(
            {
                "ts_utc": str(evt.get("ts_utc", "")),
                "run_label": str(evt.get("run_label", "")),
                "exit_code": ec,
                "reason": reason or "non_zero_exit",
            }
        )
    return failures


def render_leaderboard_md(top_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Final Leaderboard",
        "",
        "| Rank | Run Label | Model | Best Dice+ | Best AP | Best AUC | Best Epoch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(top_rows, start=1):
        best = row.get("best", {}) if isinstance(row.get("best"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("run_label", "")),
                    str(row.get("model_name", "")),
                    fmt_num(safe_float(best.get("val_dice_pos"))),
                    fmt_num(safe_float(best.get("val_average_precision_presence"))),
                    fmt_num(safe_float(best.get("val_roc_auc_presence"))),
                    str(best.get("epoch", "n/a")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final finish-up reports and leaderboard.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--condensed-story-out", default="")
    parser.add_argument("--paper-report-out", default="")
    parser.add_argument("--leaderboard-json-out", default="")
    parser.add_argument("--leaderboard-md-out", default="")
    args = parser.parse_args()

    workspace_root = pathlib.Path(args.workspace_root).resolve()
    repo_root = infer_repo_root(workspace_root, str(args.repo_root or ""))
    top_k = max(3, min(20, int(args.top_k)))

    artifacts_dir = workspace_root / ".llm_loop" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    condensed_story_out = pathlib.Path(args.condensed_story_out).resolve() if str(args.condensed_story_out).strip() else (artifacts_dir / "final_condensed_story.md")
    paper_report_out = pathlib.Path(args.paper_report_out).resolve() if str(args.paper_report_out).strip() else (artifacts_dir / "final_paper_report.md")
    leaderboard_json_out = pathlib.Path(args.leaderboard_json_out).resolve() if str(args.leaderboard_json_out).strip() else (artifacts_dir / f"final_leaderboard_top{top_k}.json")
    leaderboard_md_out = pathlib.Path(args.leaderboard_md_out).resolve() if str(args.leaderboard_md_out).strip() else (artifacts_dir / f"final_leaderboard_top{top_k}.md")

    runs_dir = repo_root / "runs"
    run_rows: list[dict[str, Any]] = []
    if runs_dir.exists():
        for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
            rec = collect_run_record(run_dir)
            if rec:
                run_rows.append(rec)
    ranked = sorted(run_rows, key=rank_score, reverse=True)
    top_rows = ranked[:top_k]

    cycle_rows = collect_cycle_summary_rows(workspace_root)
    failures = collect_failure_rows(workspace_root)

    now = datetime.now(timezone.utc).isoformat()
    best_row = top_rows[0] if top_rows else {}
    best_best = best_row.get("best", {}) if isinstance(best_row.get("best"), dict) else {}

    leaderboard_payload = {
        "generated_utc": now,
        "workspace_root": str(workspace_root),
        "repo_root": str(repo_root),
        "top_k": top_k,
        "total_ranked_runs": len(ranked),
        "rows": top_rows,
    }
    leaderboard_json_out.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_json_out.write_text(json.dumps(leaderboard_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    leaderboard_md_out.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_md_out.write_text(render_leaderboard_md(top_rows), encoding="utf-8")

    non_null_deltas = [
        r
        for r in cycle_rows
        if isinstance(r, dict) and safe_float(r.get("delta_best_val_dice_pos")) is not None and safe_float(r.get("delta_best_val_dice_pos")) > 0
    ]
    non_null_deltas = sorted(non_null_deltas, key=lambda x: safe_float(x.get("delta_best_val_dice_pos")) or 0.0, reverse=True)

    condensed_lines = [
        "# Final Condensed Story",
        "",
        f"- Generated (UTC): {now}",
        f"- Workspace: {workspace_root}",
        f"- Training repo: {repo_root}",
        f"- Total scored runs: {len(ranked)}",
        f"- Top-K: {top_k}",
        "",
        "## Best Final Candidate",
        f"- Run: {best_row.get('run_label', 'n/a')}",
        f"- Model: {best_row.get('model_name', 'n/a')}",
        f"- Best val_dice_pos: {fmt_num(safe_float(best_best.get('val_dice_pos')))}",
        f"- Best val_average_precision_presence: {fmt_num(safe_float(best_best.get('val_average_precision_presence')))}",
        f"- Best val_roc_auc_presence: {fmt_num(safe_float(best_best.get('val_roc_auc_presence')))}",
        "",
        "## What Worked",
    ]
    if top_rows:
        model_counts: dict[str, int] = {}
        for row in top_rows:
            key = str(row.get("model_name", "unknown"))
            model_counts[key] = model_counts.get(key, 0) + 1
        dominant = sorted(model_counts.items(), key=lambda kv: kv[1], reverse=True)
        for model_name, cnt in dominant[:3]:
            condensed_lines.append(f"- Strong contributor: `{model_name}` appears {cnt} time(s) in top-{top_k}.")
    else:
        condensed_lines.append("- No valid scored runs were found.")
    condensed_lines.extend(
        [
            "",
            "## What Did Not Work",
        ]
    )
    if failures:
        for f in failures[:8]:
            condensed_lines.append(
                f"- {f.get('run_label', 'n/a')} failed (exit={f.get('exit_code', 'n/a')}): {f.get('reason', 'n/a')}"
            )
    else:
        condensed_lines.append("- No non-zero exit failures recorded in events.")

    condensed_lines.extend(
        [
            "",
            "## Key Decision Milestones",
        ]
    )
    if non_null_deltas:
        for row in non_null_deltas[:8]:
            condensed_lines.append(
                f"- C{safe_int(row.get('cycle')) or 'n/a'} `{row.get('run_label', 'n/a')}` improved dice by {fmt_num(safe_float(row.get('delta_best_val_dice_pos')))}."
            )
    else:
        condensed_lines.append("- No positive delta cycles were parsed from cycle summaries.")

    condensed_lines.extend(
        [
            "",
            f"## Leaderboard",
            f"- JSON: `{leaderboard_json_out}`",
            f"- Markdown: `{leaderboard_md_out}`",
            "",
        ]
    )
    condensed_story_out.parent.mkdir(parents=True, exist_ok=True)
    condensed_story_out.write_text("\n".join(condensed_lines), encoding="utf-8")

    paper_lines = [
        "# Paper-Style Final Report",
        "",
        "## Abstract",
        (
            "This report summarizes an autonomous model search over fracture segmentation runs, "
            "with joint tracking of segmentation (`val_dice_pos`) and fracture-presence behavior "
            "(AP/AUC/precision/recall). The final recommendation is selected from a ranked run leaderboard."
        ),
        "",
        "## Experimental Setting",
        f"- Workspace root: `{workspace_root}`",
        f"- Training repository: `{repo_root}`",
        f"- Number of scored runs: {len(ranked)}",
        f"- Selection metric priority: `val_dice_pos` with AP/AUC tie-break support",
        "",
        "## Search Journey",
        f"- Total loop cycles captured: {len(cycle_rows)}",
        f"- Non-zero exit failures captured: {len(failures)}",
        "- Search phases observed: baseline/bootstrapping, structural variants, calibration/reconciliation, and final ranking.",
        "",
        "## Best Model Recommendation",
        f"- Run label: `{best_row.get('run_label', 'n/a')}`",
        f"- Model family: `{best_row.get('model_name', 'n/a')}`",
        f"- Best epoch: `{best_best.get('epoch', 'n/a')}`",
        f"- Best val_dice_pos: `{fmt_num(safe_float(best_best.get('val_dice_pos')))}`",
        f"- Best val_average_precision_presence: `{fmt_num(safe_float(best_best.get('val_average_precision_presence')))}`",
        f"- Best val_roc_auc_presence: `{fmt_num(safe_float(best_best.get('val_roc_auc_presence')))}`",
        "",
        "## Failure Analysis",
    ]
    if failures:
        for f in failures[:10]:
            paper_lines.append(
                f"- `{f.get('run_label', 'n/a')}`: exit `{f.get('exit_code', 'n/a')}`, reason `{f.get('reason', 'n/a')}`."
            )
    else:
        paper_lines.append("- No non-zero failure events captured.")

    paper_lines.extend(
        [
            "",
            "## Top-K Leaderboard",
            "",
            "| Rank | Run Label | Model | Best Dice+ | Best AP | Best AUC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for idx, row in enumerate(top_rows, start=1):
        best = row.get("best", {}) if isinstance(row.get("best"), dict) else {}
        paper_lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("run_label", "")),
                    str(row.get("model_name", "")),
                    fmt_num(safe_float(best.get("val_dice_pos"))),
                    fmt_num(safe_float(best.get("val_average_precision_presence"))),
                    fmt_num(safe_float(best.get("val_roc_auc_presence"))),
                ]
            )
            + " |"
        )
    paper_lines.extend(
        [
            "",
            "## Artifacts",
            f"- Condensed story: `{condensed_story_out}`",
            f"- Leaderboard JSON: `{leaderboard_json_out}`",
            f"- Leaderboard Markdown: `{leaderboard_md_out}`",
            "",
        ]
    )
    paper_report_out.parent.mkdir(parents=True, exist_ok=True)
    paper_report_out.write_text("\n".join(paper_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "generated_utc": now,
                "workspace_root": str(workspace_root),
                "repo_root": str(repo_root),
                "top_k": top_k,
                "total_ranked_runs": len(ranked),
                "best_run_label": best_row.get("run_label", ""),
                "best_model_name": best_row.get("model_name", ""),
                "outputs": {
                    "condensed_story_md": str(condensed_story_out),
                    "paper_report_md": str(paper_report_out),
                    "leaderboard_json": str(leaderboard_json_out),
                    "leaderboard_md": str(leaderboard_md_out),
                },
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
