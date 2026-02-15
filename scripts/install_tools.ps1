param(
    [string]$VenvPath = "C:\\Users\\Max\\code\\llm_driven_cnns_venv"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Get-PythonBootstrap {
    $candidates = @("python", "py")
    foreach ($c in $candidates) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($null -ne $cmd) { return $c }
    }
    return $null
}

function Ensure-Pip {
    param([string]$PythonExe)
    & $PythonExe -m pip --version *> $null
    if ($LASTEXITCODE -eq 0) { return $true }
    & $PythonExe -m ensurepip --upgrade *> $null
    & $PythonExe -m pip --version *> $null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-Path $VenvPath)) {
    Write-Host ("Creating venv: " + $VenvPath)
    $bootstrap = Get-PythonBootstrap
    if ($null -eq $bootstrap) { throw "No Python bootstrap executable found (python/py)." }
    & $bootstrap -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv at $VenvPath" }
}

$py = Join-Path $VenvPath "Scripts\\python.exe"
if (-not (Test-Path $py)) { throw "Python exe not found in venv: $py" }

if (Ensure-Pip -PythonExe $py) {
    Write-Host "Installing Python requirements..."
    & $py -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in venv." }
    & $py -m pip install -r (Join-Path $repoRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements.txt." }
} else {
    Write-Warning "pip is unavailable in the venv. Skipping Python package installation."
}

if ($null -eq (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "codex CLI not found in PATH. Install Codex CLI before starting the daemon."
}

Write-Host "Codex CLI detected."
Write-Host "Tool install complete."
