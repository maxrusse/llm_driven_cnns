param(
    [string]$WorkspaceRoot = "C:\Users\Max\code\llm_driven_cnns",
    [int]$IntervalSeconds = 60,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
if ($IntervalSeconds -lt 5) { $IntervalSeconds = 5 }

$statusScript = Join-Path $PSScriptRoot "status.ps1"
if (-not (Test-Path $statusScript)) {
    throw "Missing status script: $statusScript"
}

function Clear-ScreenSafe {
    try {
        Clear-Host
    } catch {
        Write-Host ("`n" + ("=" * 80) + "`n")
    }
}

while ($true) {
    Clear-ScreenSafe
    Write-Host ("LLM Daemon Watch | " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host ("WorkspaceRoot: " + $WorkspaceRoot)
    Write-Host ""

    & $statusScript -WorkspaceRoot $WorkspaceRoot

    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
