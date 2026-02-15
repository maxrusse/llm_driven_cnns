param(
    [string]$ConfigPath = "config/daemon_config.json",
    [switch]$StartInNewWindow
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$cfgAbs = if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $repoRoot $ConfigPath }
if (-not (Test-Path $cfgAbs)) { throw "Config not found: $cfgAbs" }

$cfg = Get-Content -Path $cfgAbs -Raw | ConvertFrom-Json
$workspaceRoot = if ([string]::IsNullOrWhiteSpace([string]$cfg.workspace_root)) { $repoRoot } else { [string]$cfg.workspace_root }
$dataSourceRoot = [string]$cfg.data_source_root

& (Join-Path $repoRoot "scripts\\link_data.ps1") -WorkspaceRoot $workspaceRoot -DataSourceRoot $dataSourceRoot

$daemonScript = Join-Path $repoRoot "scripts\\start_llm_daemon.ps1"
if ($StartInNewWindow) {
    Start-Process -FilePath "pwsh.exe" -WorkingDirectory $repoRoot -ArgumentList @("-NoProfile", "-File", $daemonScript, "-ConfigPath", $cfgAbs) | Out-Null
    Write-Host "Daemon started in new window."
} else {
    & $daemonScript -ConfigPath $cfgAbs
}
