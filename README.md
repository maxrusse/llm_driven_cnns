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

## Fresh Reset
```powershell
.\scripts\clean_fresh.ps1
```

Keep loop login + data link:
```powershell
.\scripts\clean_fresh.ps1 -KeepCodexLogin -KeepDataLink
```
