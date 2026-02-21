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

## Quick Start (Linux)
```bash
cd /path/to/llm_driven_cnns
bash ./scripts/install_tools.sh
bash ./scripts/login_loop_codex.sh
bash ./scripts/startup.sh --config-path config/daemon_config.linux.json
```

## Script Reference
This section is the single source of truth for `scripts/` usage.

Conventions:
- PowerShell examples assume Windows from repo root.
- Bash examples assume Linux/macOS from repo root.
- Defaults are shown exactly as implemented in the scripts.

### `install_tools` (loop-control venv only)
Purpose:
- Creates/uses the loop venv.
- Installs `requirements_wrapper.txt`.
- Verifies `codex` CLI is available.

PowerShell:
```powershell
.\scripts\install_tools.ps1 [-VenvPath <path>]
```
Options:
- `-VenvPath` (default: `C:\Users\Max\code\llm_driven_cnns_venv`)

Bash:
```bash
bash ./scripts/install_tools.sh [--venv-path <path>]
```
Options:
- `--venv-path` (default: `<workspace_root>/llm_driven_cnns_venv`, where `<workspace_root>` is parent of repo)

### `bootstrap_venvs` (loop + training venvs)
Purpose:
- Creates/uses both venvs:
  - loop: `llm_driven_cnns_venv`
  - training: `xray_fracture_benchmark_venv`
- Installs loop requirements, training requirements, optional CUDA requirements, and optional auto-repair extras.

PowerShell:
```powershell
.\scripts\bootstrap_venvs.ps1 `
  [-WorkspaceRoot <path>] `
  [-LoopRepoRoot <path>] `
  [-TrainRepoRoot <path>] `
  [-LoopVenvPath <path>] `
  [-TrainVenvPath <path>] `
  [-SkipCuda] `
  [-SkipAutoRepairExtras]
```
Defaults:
- `-WorkspaceRoot`: `C:\Users\Max\code`
- `-LoopRepoRoot`: repo root (auto)
- `-TrainRepoRoot`: `<WorkspaceRoot>\xray_fracture_benchmark`
- `-LoopVenvPath`: `<WorkspaceRoot>\llm_driven_cnns_venv`
- `-TrainVenvPath`: `<WorkspaceRoot>\xray_fracture_benchmark_venv`

Bash:
```bash
bash ./scripts/bootstrap_venvs.sh \
  [--workspace-root <path>] \
  [--loop-repo-root <path>] \
  [--train-repo-root <path>] \
  [--loop-venv-path <path>] \
  [--train-venv-path <path>] \
  [--skip-cuda] \
  [--skip-auto-repair-extras]
```
Defaults:
- `--workspace-root`: parent of repo
- `--loop-repo-root`: repo root
- `--train-repo-root`: `<workspace_root>/xray_fracture_benchmark`
- `--loop-venv-path`: `<workspace_root>/llm_driven_cnns_venv`
- `--train-venv-path`: `<workspace_root>/xray_fracture_benchmark_venv`

### `login_loop_codex` (isolated loop login)
Purpose:
- Performs Codex device auth with `CODEX_HOME` set to `.llm_loop/codex_home`.

PowerShell:
```powershell
.\scripts\login_loop_codex.ps1 [-WorkspaceRoot <path>]
```
Options:
- `-WorkspaceRoot` (default: `C:\Users\Max\code\llm_driven_cnns`)

Bash:
```bash
bash ./scripts/login_loop_codex.sh [--workspace-root <path>]
```
Options:
- `--workspace-root` (default: repo root)

### `startup` (orchestrated start helper)
Purpose:
- Links data.
- Optionally schedules finish-up window.
- Starts daemon foreground or background/new window.

PowerShell:
```powershell
.\scripts\startup.ps1 `
  [-ConfigPath <path>] `
  [-StartInNewWindow] `
  [-RunHours <hours>] `
  [-FinishupMinutes <int>] `
  [-FinishupFinalTrainingRounds <int>] `
  [-FinishupTopK <int>] `
  [-FinishupNote <text>]
```
Defaults:
- `-ConfigPath`: `config/daemon_config.json`
- `-RunHours`: `0` (disabled)
- `-FinishupMinutes`: `60`
- `-FinishupFinalTrainingRounds`: `1`
- `-FinishupTopK`: `10`
- `-FinishupNote`: empty

Bash:
```bash
bash ./scripts/startup.sh \
  [--config-path <path>] \
  [--start-in-new-window] \
  [--run-hours <hours>] \
  [--finishup-minutes <int>] \
  [--finishup-final-training-rounds <int>] \
  [--finishup-top-k <int>] \
  [--finishup-note <text>]
```
Defaults:
- `--config-path`: `config/daemon_config.linux.json` if present, else `config/daemon_config.json`
- `--run-hours`: `0`
- `--finishup-minutes`: `60`
- `--finishup-final-training-rounds`: `1`
- `--finishup-top-k`: `10`
- `--finishup-note`: empty
- `--start-in-new-window`: starts daemon with `nohup` in background

### `start_llm_daemon` (main daemon loop launcher)
Purpose:
- Initializes loop directories/files.
- Verifies loop login.
- Repeatedly invokes `scripts/llm_cycle.py`.
- Writes heartbeat and degraded/stopped status.

PowerShell:
```powershell
.\scripts\start_llm_daemon.ps1 [-ConfigPath <path>]
```
Options:
- `-ConfigPath` (default: `config/daemon_config.json`)

Bash:
```bash
bash ./scripts/start_llm_daemon.sh [--config-path <path>]
```
Options:
- `--config-path` (default: `config/daemon_config.linux.json` if present, else `config/daemon_config.json`)

### `status` (one-shot health dump)
Purpose:
- Prints heartbeat, state, finish-up request, storyline tail, workpad tail, recent events.
- Includes stale heartbeat warning logic.

PowerShell:
```powershell
.\scripts\status.ps1 [-WorkspaceRoot <path>]
```
Options:
- `-WorkspaceRoot` (default: `C:\Users\Max\code\llm_driven_cnns`)

Bash:
```bash
bash ./scripts/status.sh [--workspace-root <path>]
```
Options:
- `--workspace-root` (default: repo root)

### `watch_status` (refreshing monitor)
Purpose:
- Clears screen and calls status repeatedly.

PowerShell:
```powershell
.\scripts\watch_status.ps1 [-WorkspaceRoot <path>] [-IntervalSeconds <int>] [-Once]
```
Defaults:
- `-WorkspaceRoot`: `C:\Users\Max\code\llm_driven_cnns`
- `-IntervalSeconds`: `60` (minimum effective: `5`)

Bash:
```bash
bash ./scripts/watch_status.sh [--workspace-root <path>] [--interval-seconds <int>] [--once]
```
Defaults:
- `--workspace-root`: repo root
- `--interval-seconds`: `60` (minimum effective: `5`)

### `stop_llm_daemon` (graceful stop flags)
Purpose:
- Creates `.llm_loop/STOP_CURRENT_RUN` and `.llm_loop/STOP_DAEMON`.

PowerShell:
```powershell
.\scripts\stop_llm_daemon.ps1 [-WorkspaceRoot <path>]
```
Options:
- `-WorkspaceRoot` (default: `C:\Users\Max\code\llm_driven_cnns`)

Bash:
```bash
bash ./scripts/stop_llm_daemon.sh [--workspace-root <path>]
```
Options:
- `--workspace-root` (default: repo root)

### `request_finishup` (finish-up control file manager)
Purpose:
- Creates, schedules, shows, or cancels `.llm_loop/FINISH_UP.json`.

PowerShell:
```powershell
.\scripts\request_finishup.ps1 `
  [-WorkspaceRoot <path>] `
  [-MinutesLeft <int>] `
  [-FinalTrainingRounds <int>] `
  [-TopK <int>] `
  [-Note <text>] `
  [-ActivateInMinutes <int>] `
  [-ActivateAtUtc <iso8601>] `
  [-RunHours <float>] `
  [-ForceReportNow] `
  [-Cancel] `
  [-Show]
```
Defaults and bounds:
- `-WorkspaceRoot`: `C:\Users\Max\code\llm_driven_cnns`
- `-MinutesLeft`: `60` (minimum effective: `5`)
- `-FinalTrainingRounds`: `1` (minimum effective: `0`)
- `-TopK`: `10` (clamped to `3..20`)
- `-RunHours`: `0` (disabled)

Bash:
```bash
bash ./scripts/request_finishup.sh \
  [--workspace-root <path>] \
  [--minutes-left <int>] \
  [--final-training-rounds <int>] \
  [--top-k <int>] \
  [--note <text>] \
  [--activate-in-minutes <int>] \
  [--activate-at-utc <iso8601>] \
  [--run-hours <float>] \
  [--force-report-now] \
  [--cancel] \
  [--show]
```
Defaults and bounds:
- `--workspace-root`: repo root
- `--minutes-left`: `60` (minimum effective: `5`)
- `--final-training-rounds`: `1` (minimum effective: `0`)
- `--top-k`: `10` (clamped to `3..20`)
- `--run-hours`: `0` (disabled)

### `generate_finishup_report.py` (manual report generator)
Purpose:
- Builds condensed story, paper-style report, and leaderboard artifacts from run history.

Usage:
```bash
python ./scripts/generate_finishup_report.py \
  --workspace-root <path> \
  [--repo-root <path>] \
  [--top-k <int>] \
  [--condensed-story-out <path>] \
  [--paper-report-out <path>] \
  [--leaderboard-json-out <path>] \
  [--leaderboard-md-out <path>]
```
Defaults:
- `--top-k`: `10`
- Optional output paths default to standard `.llm_loop/artifacts/` locations when omitted.

### `link_data` (repo `data` link helper)
Purpose:
- Ensures `./data` points to training data root.

PowerShell:
```powershell
.\scripts\link_data.ps1 [-WorkspaceRoot <path>] [-DataSourceRoot <path>]
```
Defaults:
- `-WorkspaceRoot`: `C:\Users\Max\code\llm_driven_cnns`
- `-DataSourceRoot`: `C:\Users\Max\code\xray_fracture_benchmark\data`
- Creates a Windows junction at `<WorkspaceRoot>\data`.

Bash:
```bash
bash ./scripts/link_data.sh [--workspace-root <path>] [--data-source-root <path>]
```
Defaults:
- `--workspace-root`: repo root
- `--data-source-root`: `../xray_fracture_benchmark/data` (if present)
- Creates a symlink at `<workspace_root>/data`.

### `clean_fresh` (state reset between runs)
Purpose:
- Clears `.llm_loop` and `runs/`.
- Optionally preserves loop login and/or data link.

PowerShell:
```powershell
.\scripts\clean_fresh.ps1 [-WorkspaceRoot <path>] [-KeepDataLink] [-KeepCodexLogin]
```
Defaults:
- `-WorkspaceRoot`: `C:\Users\Max\code\llm_driven_cnns`

Bash:
```bash
bash ./scripts/clean_fresh.sh [--workspace-root <path>] [--keep-data-link] [--keep-codex-login]
```
Defaults:
- `--workspace-root`: repo root

### `llm_cycle.py` (internal cycle engine)
Purpose:
- Implements the decision/execution/evaluation cycle and event logging.

Usage:
- Not intended for direct manual invocation in normal operations.
- Called by `start_llm_daemon.ps1`/`start_llm_daemon.sh` with required internal arguments.

## Bootstrap Both Venvs
Create and initialize:
- loop/control venv (`llm_driven_cnns_venv`)
- training/base venv (`xray_fracture_benchmark_venv`)

```powershell
cd C:\Users\Max\code\llm_driven_cnns
.\scripts\bootstrap_venvs.ps1
```

What it installs:
- loop venv: `requirements_wrapper.txt`
- training venv: `xray_fracture_benchmark\requirements.txt`
- training CUDA stack: `xray_fracture_benchmark\requirements-cu128.txt` (unless `-SkipCuda`)
- training auto-repair extras from `config\daemon_config.json` (unless `-SkipAutoRepairExtras`)

Useful flags:
```powershell
# CPU-only / no CUDA wheel install
.\scripts\bootstrap_venvs.ps1 -SkipCuda

# Keep base install small; skip proactive auto-repair extras
.\scripts\bootstrap_venvs.ps1 -SkipAutoRepairExtras
```

Cluster note (Conda base + project venvs):
```powershell
# after activating your conda base env
.\scripts\bootstrap_venvs.ps1 -WorkspaceRoot C:\path\to\workspace
```

Linux:
```bash
cd /path/to/llm_driven_cnns
bash ./scripts/bootstrap_venvs.sh --workspace-root /path/to/workspace
```

## Rebuild Requirements
Wrapper/control venv rebuild:
```powershell
cd C:\Users\Max\code\llm_driven_cnns
python -m pip install -r .\requirements.txt
```

Notes:
- `requirements.txt` is wrapper-only and resolves to `requirements_wrapper.txt`.
- Training dependencies stay in `xray_fracture_benchmark` (`requirements.txt`, optional `requirements-cu128.txt`).
- `scripts\install_tools.ps1` installs only `requirements_wrapper.txt` to keep the loop-control venv lean.
- `scripts/install_tools.sh` does the same on Linux.

## Codex Exec Mode
- No SDK mode switch is required for Linux migration; this project uses Codex CLI.
- The loop uses JSON event output mode (`codex exec --json`) with `--output-schema` from `scripts/llm_cycle.py`.
- Config template for Linux is provided at `config/daemon_config.linux.json`.
- `run_shell` in daemon config controls command execution shell (`auto`, `bash`, `sh`, `pwsh`, `powershell`).

## Stop
```powershell
.\scripts\stop_llm_daemon.ps1
```
```bash
bash ./scripts/stop_llm_daemon.sh
```

Wait until status shows stopped:
```powershell
.\scripts\status.ps1
```
```bash
bash ./scripts/status.sh
```

## Monitoring
One-shot status:
```powershell
.\scripts\status.ps1
```
```bash
bash ./scripts/status.sh
```

Continuous watch with a fresh screen each refresh:
```powershell
.\scripts\watch_status.ps1
```
```bash
bash ./scripts/watch_status.sh --interval-seconds 30
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
