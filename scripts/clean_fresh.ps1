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
$codexBackupDir = Join-Path $WorkspaceRoot (".codex_home_preserve_" + [Guid]::NewGuid().ToString("N"))
$preservedCodexHome = $false

try {
    if ($KeepCodexLogin -and (Test-Path $codexHomeDir)) {
        Move-Item -Path $codexHomeDir -Destination $codexBackupDir -Force
        $preservedCodexHome = $true
    }

    if (Test-Path $loopDir) {
        Remove-Item -Path $loopDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    New-Item -ItemType Directory -Path (Join-Path $loopDir "logs") -Force | Out-Null

    if ($KeepCodexLogin) {
        if ($preservedCodexHome -and (Test-Path $codexBackupDir)) {
            Move-Item -Path $codexBackupDir -Destination $codexHomeDir -Force
        } else {
            # Ensure expected path exists even if no prior login was present.
            New-Item -ItemType Directory -Path $codexHomeDir -Force | Out-Null
        }
    }
} finally {
    if (Test-Path $codexBackupDir) {
        Remove-Item -Path $codexBackupDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

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

if ($KeepCodexLogin -and $preservedCodexHome -and (Test-Path $codexHomeDir)) {
    Write-Host "Fresh cleanup complete. Preserved Codex loop login at: $codexHomeDir"
} elseif ($KeepCodexLogin -and (Test-Path $codexHomeDir)) {
    Write-Host "Fresh cleanup complete. KeepCodexLogin was set; codex_home path exists at: $codexHomeDir"
    Write-Host "If login was not previously stored there, run .\\scripts\\login_loop_codex.ps1 once."
} else {
    Write-Host "Fresh cleanup complete."
}
