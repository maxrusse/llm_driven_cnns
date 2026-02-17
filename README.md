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
