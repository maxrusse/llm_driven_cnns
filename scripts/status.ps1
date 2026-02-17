param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns"
)

$ErrorActionPreference = "Stop"
$cfgPath = Join-Path $WorkspaceRoot "config\\daemon_config.json"
$hb = Join-Path $WorkspaceRoot ".llm_loop\\logs\\daemon_heartbeat.json"
$state = Join-Path $WorkspaceRoot ".llm_loop\\state.json"
$events = Join-Path $WorkspaceRoot ".llm_loop\\logs\\events.jsonl"
$storyline = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\storyline.md"
$workpad = Join-Path $WorkspaceRoot ".llm_loop\\artifacts\\workpad.md"
$stateObj = $null
if (Test-Path $state) {
    try { $stateObj = Get-Content -Path $state -Raw | ConvertFrom-Json } catch {}
}

if (Test-Path $hb) {
    Write-Host "Heartbeat:"
    $hbRaw = Get-Content -Path $hb -Raw
    Write-Host $hbRaw
    try {
        $hbObj = $hbRaw | ConvertFrom-Json
        $pollSeconds = 20
        if (Test-Path $cfgPath) {
            try {
                $cfgObj = Get-Content -Path $cfgPath -Raw | ConvertFrom-Json
                if ($cfgObj.daemon_poll_seconds) {
                    $pollSeconds = [int]$cfgObj.daemon_poll_seconds
                }
            } catch {}
        }
        $staleAfter = [Math]::Max(60, $pollSeconds * 3)
        $updatedUtc = $null
        if ($hbObj.updated_utc) {
            $updatedUtc = [DateTimeOffset]::Parse([string]$hbObj.updated_utc)
        }
        if ($updatedUtc -ne $null) {
            $ageSeconds = [int](([DateTimeOffset]::UtcNow - $updatedUtc).TotalSeconds)
            if ($ageSeconds -gt $staleAfter -and ($hbObj.daemon_status -eq "running" -or $hbObj.daemon_status -eq "degraded")) {
                $activePid = $null
                $activeRunAlive = $false
                if ($stateObj -and $stateObj.active_run -and $stateObj.active_run.pid) {
                    $activePid = [int]$stateObj.active_run.pid
                    try {
                        $p = Get-Process -Id $activePid -ErrorAction Stop
                        if ($null -ne $p) { $activeRunAlive = $true }
                    } catch {}
                }
                if ($activeRunAlive) {
                    Write-Warning ("Heartbeat is stale (" + $ageSeconds + "s old), but active_run PID " + $activePid + " is alive. Daemon may be inside a long monitor window.")
                } else {
                    Write-Warning ("Heartbeat is stale (" + $ageSeconds + "s old). Daemon likely not running anymore.")
                }
            }
        }
    } catch {}
} else {
    Write-Host "Heartbeat file missing."
}

if (Test-Path $state) {
    Write-Host ""
    Write-Host "State:"
    if ($null -ne $stateObj) {
        $stateObj | ConvertTo-Json -Depth 8
    } else {
        Get-Content -Path $state
    }
}

if (Test-Path $storyline) {
    Write-Host ""
    Write-Host "Storyline (latest 40 lines):"
    Get-Content -Path $storyline -Tail 40
}

if (Test-Path $workpad) {
    Write-Host ""
    Write-Host "Workpad (latest 80 lines):"
    Get-Content -Path $workpad -Tail 80
}

if (Test-Path $events) {
    Write-Host ""
    Write-Host "Recent events:"
    Get-Content -Path $events -Tail 30
}
