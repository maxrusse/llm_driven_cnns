param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns"
)

$ErrorActionPreference = "Stop"
$hb = Join-Path $WorkspaceRoot ".llm_loop\\logs\\daemon_heartbeat.json"
$state = Join-Path $WorkspaceRoot ".llm_loop\\state.json"
$events = Join-Path $WorkspaceRoot ".llm_loop\\logs\\events.jsonl"
$summary = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\cycle_summary.md"
$storyline = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\storyline.md"
$condensed = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\condensed_log.md"
$notes = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\notes.md"
$todo = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\todo.md"

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

if (Test-Path $storyline) {
    Write-Host ""
    Write-Host "Storyline (latest 30 lines):"
    Get-Content -Path $storyline -Tail 30
}

if (Test-Path $condensed) {
    Write-Host ""
    Write-Host "Condensed log (latest 20 lines):"
    Get-Content -Path $condensed -Tail 20
}

if (Test-Path $todo) {
    Write-Host ""
    Write-Host "TODO (latest 20 lines):"
    Get-Content -Path $todo -Tail 20
}

if (Test-Path $notes) {
    Write-Host ""
    Write-Host "Notes (latest 20 lines):"
    Get-Content -Path $notes -Tail 20
}

if (Test-Path $events) {
    Write-Host ""
    Write-Host "Recent events:"
    Get-Content -Path $events -Tail 30
}
