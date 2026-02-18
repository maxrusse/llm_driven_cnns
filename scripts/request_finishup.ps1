param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns",
    [int]$MinutesLeft = 60,
    [int]$FinalTrainingRounds = 1,
    [int]$TopK = 10,
    [string]$Note = "",
    [int]$ActivateInMinutes = 0,
    [string]$ActivateAtUtc = "",
    [double]$RunHours = 0,
    [switch]$ForceReportNow,
    [switch]$Cancel,
    [switch]$Show
)

$ErrorActionPreference = "Stop"

$minutes = [Math]::Max(5, $MinutesLeft)
$rounds = [Math]::Max(0, $FinalTrainingRounds)
$k = [Math]::Min(20, [Math]::Max(3, $TopK))

$loopDir = Join-Path $WorkspaceRoot ".llm_loop"
if (-not (Test-Path $loopDir)) {
    New-Item -ItemType Directory -Path $loopDir -Force | Out-Null
}
$controlPath = Join-Path $loopDir "FINISH_UP.json"

if ($Cancel) {
    if (Test-Path $controlPath) {
        Remove-Item -Path $controlPath -Force
        Write-Host "Finish-up request removed: $controlPath"
    } else {
        Write-Host "No finish-up request file found."
    }
    return
}

if ($Show) {
    if (Test-Path $controlPath) {
        Get-Content -Path $controlPath
    } else {
        Write-Host "No finish-up request file found at $controlPath"
    }
    return
}

$now = [DateTimeOffset]::UtcNow
$activateAt = $now
$activationMode = "immediate"

if ($RunHours -gt 0) {
    $totalMinutes = [int][Math]::Round($RunHours * 60.0)
    $ActivateInMinutes = [Math]::Max(0, $totalMinutes - $minutes)
}

if (-not [string]::IsNullOrWhiteSpace($ActivateAtUtc)) {
    try {
        $activateAt = [DateTimeOffset]::Parse($ActivateAtUtc).ToUniversalTime()
        $activationMode = "scheduled_at_utc"
    } catch {
        throw "Invalid -ActivateAtUtc value: $ActivateAtUtc"
    }
} elseif ($ActivateInMinutes -gt 0) {
    $activateAt = $now.AddMinutes($ActivateInMinutes)
    $activationMode = "scheduled_in_minutes"
}

$deadline = $activateAt.AddMinutes($minutes)
$status = if ($activateAt -gt $now) { "scheduled" } else { "requested" }

$payload = [ordered]@{
    enabled = $true
    status = $status
    requested_utc = $now.ToString("o")
    activate_at_utc = $activateAt.ToString("o")
    deadline_utc = $deadline.ToString("o")
    minutes_left = $minutes
    total_minutes_window = $minutes
    final_training_rounds_target = $rounds
    report_top_k = $k
    force_report_now = [bool]$ForceReportNow
    activation_mode = $activationMode
    note = [string]$Note
}

$payload | ConvertTo-Json -Depth 4 | Set-Content -Path $controlPath -Encoding UTF8

Write-Host "Finish-up request written:"
Write-Host ("  file: " + $controlPath)
Write-Host ("  deadline_utc: " + $payload.deadline_utc)
Write-Host ("  final_training_rounds_target: " + $payload.final_training_rounds_target)
Write-Host ("  report_top_k: " + $payload.report_top_k)
Write-Host ("  status: " + $payload.status)
Write-Host ("  activate_at_utc: " + $payload.activate_at_utc)
if ($ForceReportNow) {
    Write-Host "  mode: report_now"
} else {
    if ($payload.status -eq "scheduled") {
        Write-Host "  mode: scheduled_finishup_then_report"
    } else {
        Write-Host "  mode: final_training_then_report"
    }
}
