# llm_driven_cnns

Cleanroom wrapper for Codex-driven CNN experimentation.
This loop is used in a competitive challenge setting, with focus on high-quality validation gains and reproducibility.

## Quick Start
```powershell
cd C:\Users\Max\code\llm_driven_cnns
.\scripts\install_tools.ps1
.\scripts\login_loop_codex.ps1
.\scripts\startup.ps1
```

## Stop
```powershell
.\scripts\stop_llm_daemon.ps1
```

Wait until status shows stopped:
```powershell
.\scripts\status.ps1
```

## Monitoring
One-shot status:
```powershell
.\scripts\status.ps1
```

Continuous watch with a fresh screen each refresh:
```powershell
.\scripts\watch_status.ps1
```

Examples:
```powershell
.\scripts\watch_status.ps1 -IntervalSeconds 30
.\scripts\watch_status.ps1 -Once
```

Behavior note (important):
- Heartbeat (`.llm_loop/logs/daemon_heartbeat.json`) is refreshed after a cycle returns.
- A `run_command` cycle can monitor a training process for a long window (`monitor_seconds`) before returning.
- During that window, heartbeat timestamp may look stale even though GPU training is still active.

How to confirm active training:
```powershell
# active run metadata (pid/log paths)
Get-Content .\.llm_loop\state.json

# latest training logs changing over time = run is alive
Get-ChildItem .\.llm_loop\logs\*xray*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 6 Name,Length,LastWriteTime
```

Where timing comes from:
- `daemon_poll_seconds` in `config/daemon_config.json` controls daemon loop sleep between cycles.
- `monitor_seconds` is chosen per cycle decision (clamped by `default_monitor_seconds` and `max_monitor_seconds` in config).

Cycle traceability:
- Each cycle now has a `cycle_trace_id` written into events, summaries, and storyline entries for easier debugging.

## Role Contracts
Loop mission is now role-separated:
- Worker mission: `WORKER_MISSION.md`
- Mentor mission: `MENTOR_MISSION.md`
- Display names (ops/UI): `AI-Builder` (worker) and `AI-Mentor` (mentor).
- Shared strategy dimensions (both roles): `augmentation`, `preprocessing`, `data_sampling`, `loss`, `model_arch`, `optimization`, `evaluation`.

Current interaction profile:
- `interaction_mode=standard` (in `config/daemon_config.json`)
- Builder and Mentor both participate in autonomous learning (execution owner + strategic challenger).
- Interaction is tuned for low overhead: concise challenges, sparse TODOs, and execution-first follow-through.

Cadence policy:
- Adaptive and evidence-driven, not fixed cycle quotas.
- Loop uses priority signals (`research_priority`, `non_training_priority`, `breakout_priority`, TODO pressure).
- Signals guide decisions; they do not hard-force a specific action.

Configured in `config/daemon_config.json`:
- `worker_mission_file`
- `mentor_mission_file`
- `interaction_mode` (`standard` or `builder_health_helper`)

Fallback:
- `mission_file` remains supported and is used when role-specific files are not set.

Interaction model (SOTA-inspired):
- **OpenAI handoffs pattern**: keep a clear owner per step, then hand off explicitly when review/critique is needed.
  - https://openai.github.io/openai-agents-js/guides/handoffs/
- **Anthropic agent workflow guidance**: keep loops simple, structured, and inspectable before adding complexity.
  - https://www.anthropic.com/engineering/building-effective-agents
- **OpenClaw multi-agent architecture**: separate agent responsibilities and shared memory/state for robust long runs.
  - https://docs.openclaw.ai/home/architecture

## Worker Housekeeping (Per Cycle)
Worker decision output includes required housekeeping fields. Wrapper writes these into artifacts:
- `todo_new` -> `.llm_loop/artifacts/shared_todo.md` (single shared TODO queue, capped to 0-2)
- `notes_update` -> `.llm_loop/artifacts/workpad.md` (`## Notes`)
- `data_exploration_update` -> `.llm_loop/artifacts/workpad.md` (`## Data Exploration`)
- `resolve_shared_todo_ids` -> marks matching IDs in `.llm_loop/artifacts/shared_todo.md` as resolved
- mentor TODO writes are constrained: challenge-oriented and capped to 0-1 per cycle

Context ingestion behavior:
- Worker/mentor prompts read latest tails of `workpad.md` and `mentor_notes.md` (not file heads).
- `shared_todo.md` is ingested as a compact focus summary (oldest unresolved + newest unresolved, deduped) to reduce context bloat while keeping long-lived priorities visible.
- `storyline.md` is treated as backup context and is included only when failure/stall signals are high.
- Recent `.log` tails are injected only when the latest completed run failed.
- Unresolved shared TODO count is computed from the full `shared_todo.md` file.
- New TODO writes are backlog-aware and duplicate-aware (caps tighten when open backlog grows).
- Runtime prompt context is intentionally compact (high-level signals over long heuristic payloads).

## Lightweight Quality Gate
Before execution, wrapper applies a simple decision-quality checklist.
If violated, the cycle is auto-downgraded to `wait` and reasons are logged.

Current gate checks:
- rationale must be non-trivial (not very short)
- housekeeping quality is tracked, but empty housekeeping no longer blocks execution
- `run_command` must include non-empty command, non-generic run label, and explicit repo context (`Set-Location`/`cd`)

Quality gate telemetry appears in cycle events under `quality_gate` and storyline entries.

## Auto Repair (Install + Retry)
When a run fails, the wrapper can do one automatic recovery pass:
- Detect known `ModuleNotFoundError` cases from stderr.
- Install mapped packages into the configured training venv (`auto_repair_python_exe`).
- If a missing module is not mapped, try a direct package guess (module name / alias mapping).
- If configured, clone a model source repo and install it into the same training venv.
- Retry the same command once.
- If stderr shows `Unsupported model.name` for known aliases (for example `unet_resnet34`), apply a configured fallback model name and retry once.
- If a model fallback exists, fallback is preferred over blind package install attempts.
- For `--config`-driven runs, fallback writes a patched YAML copy into `.llm_loop/autofix_configs/` and retries with that file.

Config keys in `config/daemon_config.json`:
- `auto_repair_enabled`
- `auto_repair_retry_on_success`
- `auto_repair_allow_direct_module_install`
- `auto_repair_python_exe`
- `auto_repair_module_package_map`
- `auto_repair_module_alias_map`
- `auto_repair_module_clone_map`
- `auto_repair_model_package_map`
- `auto_repair_model_clone_map`
- `auto_repair_model_fallback_map`

Clone map format (optional):
```json
{
  "auto_repair_module_clone_map": {
    "some_missing_module": {
      "repo": "https://github.com/org/repo.git",
      "ref": "main",
      "subdir": "",
      "editable": true
    }
  }
}
```
`repo` can also be set as a direct string value.

Auto-repair evidence is written into cycle events under `auto_repair` and retry outcomes under `run_outcome_retry`.

## Finish-Up Mode (Last Hour + Final Report)
Trigger finish-up when you want one last focused round and then final reporting:
```powershell
.\scripts\request_finishup.ps1 -MinutesLeft 60 -FinalTrainingRounds 1 -TopK 10 -Note "final hour"
```

Schedule finish-up in advance (example: run 8h total, last 1h is finish-up):
```powershell
.\scripts\request_finishup.ps1 -RunHours 8 -MinutesLeft 60 -FinalTrainingRounds 1 -TopK 10 -Note "overnight schedule"
```

Alternative schedule flags:
- `-ActivateInMinutes <N>`
- `-ActivateAtUtc "2026-02-18T22:00:00Z"`

Or set it directly at startup:
```powershell
.\scripts\startup.ps1 -StartInNewWindow -RunHours 8 -FinishupMinutes 60 -FinishupFinalTrainingRounds 1 -FinishupTopK 10 -FinishupNote "overnight run"
```

What this does:
- Writes `.llm_loop/FINISH_UP.json`.
- Worker + mentor prompts become deadline-aware.
- If feasible: one final high-value training/fine-tune round.
- Then: report generation with condensed story, paper-style report, and top-k leaderboard.

Status / control:
```powershell
.\scripts\status.ps1
.\scripts\request_finishup.ps1 -Show
.\scripts\request_finishup.ps1 -Cancel
```

Force report-only mode (skip final training):
```powershell
.\scripts\request_finishup.ps1 -ForceReportNow -TopK 10 -Note "report now"
```

Manual report generation command:
```powershell
python .\scripts\generate_finishup_report.py --workspace-root C:\Users\Max\code\llm_driven_cnns --top-k 10
```

Output artifacts:
- `.llm_loop/artifacts/final_condensed_story.md`
- `.llm_loop/artifacts/final_paper_report.md`
- `.llm_loop/artifacts/final_leaderboard_top10.json`
- `.llm_loop/artifacts/final_leaderboard_top10.md`

## Fresh Reset
```powershell
.\scripts\clean_fresh.ps1
```

Keep loop login + data link:
```powershell
.\scripts\clean_fresh.ps1 -KeepCodexLogin -KeepDataLink
```

## Best Behavior
Clean stop + fresh restart (preserve login + data link):
```powershell
.\scripts\stop_llm_daemon.ps1
.\scripts\clean_fresh.ps1 -KeepCodexLogin -KeepDataLink
.\scripts\login_loop_codex.ps1
.\scripts\startup.ps1
.\scripts\status.ps1
```

Notes:
- `-KeepCodexLogin` is an option of `clean_fresh.ps1`, not `login_loop_codex.ps1`.
- If startup reports loop login missing, run `.\scripts\login_loop_codex.ps1` once and start again.

## Mentor Review Loop
- A second Codex reviewer can challenge the primary loop decision before execution.
- Role split:
  - Worker Codex: analysis/exploration/training execution and command decisions.
  - Mentor Codex: critique/validation/questions and strategic guidance only.
- Shared coordination artifacts:
  - `.llm_loop/artifacts/workpad.md` (worker-owned)
  - `.llm_loop/artifacts/mentor_notes.md` (mentor-owned, wrapper-written from mentor output)
  - `.llm_loop/artifacts/shared_todo.md` (single shared queue; worker + mentor append via wrapper)
- Config keys in `config/daemon_config.json`:
  - `interaction_mode`
  - `mentor_enabled`
  - `mentor_every_n_cycles`
  - `mentor_health_check_every_n_cycles` (used by `builder_health_helper` mode)
  - `mentor_force_when_stuck`
  - `mentor_challenge_streak_min_idle_cycles` (prevents stuck-trigger mentor forcing while worker is already executing)
  - `mentor_apply_suggestions`
  - `enable_stuck_wait_recovery` (defaults to `false`; prevents forced relaunches during long wait-only windows)
  - `stale_orphan_idle_cycles`, `stuck_wait_recovery_cycles`
  - `worker_display_name`, `mentor_display_name`
  - `mentor_require_web_search`
  - `mentor_model`, `mentor_reasoning_effort`, `mentor_web_search_mode`
- Mentor telemetry and critique are written into each cycle event under `codex.mentor`.
- Mentor is expected to propose concrete model-architecture alternatives when stagnation persists, but not on every cycle by default.

Standard dual-agent behavior (current default):
- Mentor cadence defaults to every 4 cycles (plus stuck-condition checks).
- Mentor can challenge with a concrete replacement decision, but challenge loops are discouraged unless there is new evidence.
- Worker responds by action and evaluation, not extended back-and-forth discussion.
- Mentor TODO writes are capped to at most one item on challenge, zero on continue.
- Mentor review is lean-scripted: explicit trajectory verdict (`on_track`/`off_track`) and at most one critical question.
- Prompt rule budget is hard-capped to stay lean: worker <=12, mentor <=8 (current standard profile: 9 + 6 = 15).
- Storyline is lean: one main decision line per cycle for quick scanability.
