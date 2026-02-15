param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns",
    [switch]$KeepDataLink,
    [switch]$KeepCodexLogin
)

$ErrorActionPreference = "Stop"
$loopDir = Join-Path $WorkspaceRoot ".llm_loop"
$runsDir = Join-Path $WorkspaceRoot "runs"
$dataLink = Join-Path $WorkspaceRoot "data"
$codexHomeDir = Join-Path $loopDir "codex_home"

if (Test-Path $loopDir) {
    if ($KeepCodexLogin -and (Test-Path $codexHomeDir)) {
        Get-ChildItem -Path $loopDir -Force -ErrorAction SilentlyContinue | Where-Object {
            $_.FullName -ne $codexHomeDir
        } | ForEach-Object {
            Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    } else {
        Remove-Item -Path $loopDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
New-Item -ItemType Directory -Path (Join-Path $loopDir "logs") -Force | Out-Null

if (Test-Path $runsDir) {
    Get-ChildItem -Path $runsDir -Force -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    New-Item -ItemType Directory -Path $runsDir -Force | Out-Null
}

if (-not $KeepDataLink -and (Test-Path $dataLink)) {
    Remove-Item -Path $dataLink -Recurse -Force -ErrorAction SilentlyContinue
}

if ($KeepCodexLogin -and (Test-Path $codexHomeDir)) {
    Write-Host "Fresh cleanup complete. Preserved Codex loop login at: $codexHomeDir"
} elseif ($KeepCodexLogin) {
    Write-Host "Fresh cleanup complete. KeepCodexLogin was set, but no existing codex_home was found."
} else {
    Write-Host "Fresh cleanup complete."
}
