param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns"
)

$ErrorActionPreference = "Stop"
$hb = Join-Path $WorkspaceRoot ".llm_loop\\logs\\daemon_heartbeat.json"
$state = Join-Path $WorkspaceRoot ".llm_loop\\state.json"
$events = Join-Path $WorkspaceRoot ".llm_loop\\logs\\events.jsonl"
$summary = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\cycle_summary.md"

if (Test-Path $hb) {
    Write-Host "Heartbeat:"
    Get-Content -Path $hb
} else {
    Write-Host "Heartbeat file missing."
}

if (Test-Path $state) {
    Write-Host ""
    Write-Host "State:"
    Get-Content -Path $state
}

if (Test-Path $summary) {
    Write-Host ""
    Write-Host "Cycle summary (6-point):"
    Get-Content -Path $summary
}

if (Test-Path $events) {
    Write-Host ""
    Write-Host "Recent events:"
    Get-Content -Path $events -Tail 30
}
