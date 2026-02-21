param(
    [string]$ConfigPath = "config/daemon_config.json",
    [switch]$StartInNewWindow,
    [double]$RunHours = 0,
    [int]$FinishupMinutes = 60,
    [int]$FinishupFinalTrainingRounds = 1,
    [int]$FinishupTopK = 10,
    [string]$FinishupNote = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$cfgAbs = if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $repoRoot $ConfigPath }
if (-not (Test-Path $cfgAbs)) { throw "Config not found: $cfgAbs" }

$cfg = Get-Content -Path $cfgAbs -Raw | ConvertFrom-Json
$workspaceRoot = if ([string]::IsNullOrWhiteSpace([string]$cfg.workspace_root)) { $repoRoot } else { [string]$cfg.workspace_root }
$dataSourceRoot = [string]$cfg.data_source_root
$effectiveCfgAbs = $cfgAbs

& (Join-Path $repoRoot "scripts\\link_data.ps1") -WorkspaceRoot $workspaceRoot -DataSourceRoot $dataSourceRoot

$requestFinishupScript = Join-Path $repoRoot "scripts\\request_finishup.ps1"
if ($RunHours -gt 0) {
    if (-not (Test-Path $requestFinishupScript)) {
        throw "Missing finish-up request script: $requestFinishupScript"
    }
    & $requestFinishupScript `
        -WorkspaceRoot $workspaceRoot `
        -RunHours $RunHours `
        -MinutesLeft $FinishupMinutes `
        -FinalTrainingRounds $FinishupFinalTrainingRounds `
        -TopK $FinishupTopK `
        -Note $FinishupNote
    Write-Host ("Scheduled finish-up via startup: RunHours=" + $RunHours + ", FinishupMinutes=" + $FinishupMinutes)
}

$daemonScript = Join-Path $repoRoot "scripts\\start_llm_daemon.ps1"
if ($StartInNewWindow) {
    Start-Process -FilePath "pwsh.exe" -WorkingDirectory $repoRoot -ArgumentList @("-NoProfile", "-File", $daemonScript, "-ConfigPath", $effectiveCfgAbs) | Out-Null
    Write-Host "Daemon started in new window."
} else {
    & $daemonScript -ConfigPath $effectiveCfgAbs
}
