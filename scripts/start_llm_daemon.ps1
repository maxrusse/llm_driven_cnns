param(
    [string]$ConfigPath = "config/daemon_config.json"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$cfgAbs = if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $repoRoot $ConfigPath }
if (-not (Test-Path $cfgAbs)) { throw "Config not found: $cfgAbs" }

$cfg = Get-Content -Path $cfgAbs -Raw | ConvertFrom-Json
$workspaceRoot = if ([string]::IsNullOrWhiteSpace([string]$cfg.workspace_root)) { $repoRoot } else { [string]$cfg.workspace_root }
$pythonExe = [string]$cfg.python_exe
if ([string]::IsNullOrWhiteSpace($pythonExe) -or -not (Test-Path $pythonExe)) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $py) { throw "Python not found and config.python_exe is invalid." }
    $pythonExe = $py.Source
}

$loopDir = Join-Path $workspaceRoot ".llm_loop"
$logsDir = Join-Path $loopDir "logs"
$artifactsDir = Join-Path $loopDir "artifacts"
foreach ($d in @($loopDir, $logsDir, $artifactsDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$stateFile = Join-Path $loopDir "state.json"
$eventsFile = Join-Path $logsDir "events.jsonl"
$heartbeatFile = Join-Path $logsDir "daemon_heartbeat.json"
$codexHome = Join-Path $loopDir "codex_home"
$stopDaemonFlag = Join-Path $loopDir "STOP_DAEMON"
$stopCurrentRunFlag = Join-Path $loopDir "STOP_CURRENT_RUN"
$threadIdFile = Join-Path $artifactsDir "codex_thread_id.txt"
$cycleScript = Join-Path $repoRoot "scripts\llm_cycle.py"

if (-not (Test-Path $cycleScript)) { throw "Missing cycle script: $cycleScript" }
if ($null -eq (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "codex CLI not found in PATH. Install/update Codex CLI first."
}
$codexExe = (Get-Command codex -ErrorAction Stop).Source
if ($codexExe -like "*.ps1") {
    $candidate = [System.IO.Path]::ChangeExtension($codexExe, ".cmd")
    if (Test-Path $candidate) { $codexExe = $candidate }
} elseif ($codexExe -notlike "*.cmd" -and (Test-Path ($codexExe + ".cmd"))) {
    $codexExe = $codexExe + ".cmd"
}

if (Test-Path $stopDaemonFlag) { Remove-Item -Path $stopDaemonFlag -Force -ErrorAction SilentlyContinue }
if (Test-Path $stopCurrentRunFlag) { Remove-Item -Path $stopCurrentRunFlag -Force -ErrorAction SilentlyContinue }
if (-not (Test-Path $stateFile)) { '{"active_run": null}' | Set-Content -Path $stateFile -Encoding UTF8 }
if (-not (Test-Path $codexHome)) { New-Item -ItemType Directory -Path $codexHome -Force | Out-Null }

$prevCodexHome = $env:CODEX_HOME
$env:CODEX_HOME = $codexHome
& $codexExe login status *> $null
$loginStatusCode = $LASTEXITCODE
if ([string]::IsNullOrWhiteSpace($prevCodexHome)) {
    Remove-Item Env:CODEX_HOME -ErrorAction SilentlyContinue
} else {
    $env:CODEX_HOME = $prevCodexHome
}
if ($loginStatusCode -ne 0) {
    throw "Codex is not logged in for loop CODEX_HOME ($codexHome). Run .\\scripts\\login_loop_codex.ps1 first."
}

$runId = [string]$cfg.run_id
$pollSeconds = [int]$cfg.daemon_poll_seconds
if ($pollSeconds -lt 5) { $pollSeconds = 5 }

$cycle = 0
Write-Host ("Starting LLM daemon run_id=" + $runId + " model=" + [string]$cfg.model + " reasoning=" + [string]$cfg.reasoning_effort)
Write-Host ("Workspace: " + $workspaceRoot)
Write-Host ("CODEX_HOME: " + $codexHome)
Write-Host ("Stop flag: " + $stopDaemonFlag)

while ($true) {
    if (Test-Path $stopDaemonFlag) {
        Write-Host "STOP_DAEMON found. Exiting daemon loop."
        break
    }

    $cycle += 1
    $startedUtc = (Get-Date).ToUniversalTime().ToString("o")
    $args = @(
        $cycleScript,
        "--workspace-root", $workspaceRoot,
        "--config-path", $cfgAbs,
        "--codex-exe", $codexExe,
        "--codex-home", $codexHome,
        "--thread-id-file", $threadIdFile,
        "--state-file", $stateFile,
        "--events-file", $eventsFile,
        "--stop-daemon-flag", $stopDaemonFlag,
        "--stop-current-run-flag", $stopCurrentRunFlag
    )
    $raw = & $pythonExe @args 2>&1
    $exitCode = $LASTEXITCODE
    $endedUtc = (Get-Date).ToUniversalTime().ToString("o")

    $rawText = if ($raw -is [System.Array]) { ($raw -join "`n") } else { [string]$raw }
    if ($null -eq $rawText) { $rawText = "" }
    if ($rawText.Length -gt 4000) { $rawText = $rawText.Substring($rawText.Length - 4000) }

    $heartbeat = [ordered]@{
        daemon_status = if ($exitCode -eq 0) { "running" } else { "degraded" }
        run_id = $runId
        cycle = $cycle
        cycle_started_utc = $startedUtc
        cycle_ended_utc = $endedUtc
        last_exit_code = $exitCode
        last_cycle_output = $rawText
        state_file = $stateFile
        events_file = $eventsFile
        codex_home = $codexHome
        updated_utc = (Get-Date).ToUniversalTime().ToString("o")
        stop_daemon_flag = $stopDaemonFlag
        stop_current_run_flag = $stopCurrentRunFlag
    }
    $heartbeat | ConvertTo-Json -Depth 6 | Set-Content -Path $heartbeatFile -Encoding UTF8

    if ($exitCode -ne 0) {
        Write-Warning ("Cycle failed with exit code " + $exitCode)
    }
    Start-Sleep -Seconds $pollSeconds
}

$finalHeartbeat = [ordered]@{
    daemon_status = "stopped"
    run_id = $runId
    cycle = $cycle
    updated_utc = (Get-Date).ToUniversalTime().ToString("o")
    stop_daemon_flag = $stopDaemonFlag
}
$finalHeartbeat | ConvertTo-Json -Depth 4 | Set-Content -Path $heartbeatFile -Encoding UTF8
Write-Host "LLM daemon stopped."
