param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns"
)

$ErrorActionPreference = "Stop"
$loopDir = Join-Path $WorkspaceRoot ".llm_loop"
if (-not (Test-Path $loopDir)) { New-Item -ItemType Directory -Path $loopDir -Force | Out-Null }

New-Item -ItemType File -Path (Join-Path $loopDir "STOP_CURRENT_RUN") -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $loopDir "STOP_DAEMON") -Force | Out-Null

Write-Host "Stop flags created."
