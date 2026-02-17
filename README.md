# llm_driven_cnns

Cleanroom wrapper for Codex-driven CNN experimentation.

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

## Role Contracts
Loop mission is now role-separated:
- Worker mission: `WORKER_MISSION.md`
- Mentor mission: `MENTOR_MISSION.md`

Configured in `config/daemon_config.json`:
- `worker_mission_file`
- `mentor_mission_file`

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
- `todo_new` -> `.llm_loop/artifacts/workpad.md` (`## TODO`)
- `notes_update` -> `.llm_loop/artifacts/workpad.md` (`## Notes`)
- `data_exploration_update` -> `.llm_loop/artifacts/workpad.md` (`## Data Exploration`)
- `resolve_shared_todo_ids` -> marks matching IDs in `.llm_loop/artifacts/shared_todo.md` as resolved

## Lightweight Quality Gate
Before execution, wrapper applies a simple decision-quality checklist.
If violated, the cycle is auto-downgraded to `wait` and reasons are logged.

Current gate checks:
- rationale should be non-trivial (not very short)
- housekeeping cannot be fully empty on routine worker cycles (`run_command`/`wait`)
- `run_command` must include non-empty command, non-generic run label, and explicit repo context (`Set-Location`/`cd`)

Quality gate telemetry appears in cycle events under `quality_gate` and storyline entries.

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
  - `.llm_loop/artifacts/shared_todo.md` (shared queue; mentor can append via wrapper)
- Config keys in `config/daemon_config.json`:
  - `mentor_enabled`
  - `mentor_every_n_cycles`
  - `mentor_force_when_stuck`
  - `mentor_apply_suggestions`
  - `mentor_require_web_search`
  - `mentor_model`, `mentor_reasoning_effort`, `mentor_web_search_mode`
- Mentor telemetry and critique are written into each cycle event under `codex.mentor`.
