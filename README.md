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
