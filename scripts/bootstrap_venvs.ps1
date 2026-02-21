param(
    [string]$WorkspaceRoot = "C:\Users\Max\code",
    [string]$LoopRepoRoot = "",
    [string]$TrainRepoRoot = "",
    [string]$LoopVenvPath = "",
    [string]$TrainVenvPath = "",
    [switch]$SkipCuda,
    [switch]$SkipAutoRepairExtras
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($LoopRepoRoot)) {
    $LoopRepoRoot = Split-Path -Parent $PSScriptRoot
}
if ([string]::IsNullOrWhiteSpace($TrainRepoRoot)) {
    $TrainRepoRoot = Join-Path $WorkspaceRoot "xray_fracture_benchmark"
}
if ([string]::IsNullOrWhiteSpace($LoopVenvPath)) {
    $LoopVenvPath = Join-Path $WorkspaceRoot "llm_driven_cnns_venv"
}
if ([string]::IsNullOrWhiteSpace($TrainVenvPath)) {
    $TrainVenvPath = Join-Path $WorkspaceRoot "xray_fracture_benchmark_venv"
}

if (-not (Test-Path $LoopRepoRoot)) { throw "Loop repo root not found: $LoopRepoRoot" }
if (-not (Test-Path $TrainRepoRoot)) { throw "Training repo root not found: $TrainRepoRoot" }

$loopReq = Join-Path $LoopRepoRoot "requirements_wrapper.txt"
$trainReq = Join-Path $TrainRepoRoot "requirements.txt"
$trainReqCuda = Join-Path $TrainRepoRoot "requirements-cu128.txt"
$daemonCfg = Join-Path $LoopRepoRoot "config\daemon_config.json"

if (-not (Test-Path $loopReq)) { throw "Missing loop requirements file: $loopReq" }
if (-not (Test-Path $trainReq)) { throw "Missing training requirements file: $trainReq" }

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

function Ensure-VenvPython {
    param([string]$VenvPath)
    if (-not (Test-Path $VenvPath)) {
        Write-Host ("Creating venv: " + $VenvPath)
        $bootstrap = Get-PythonBootstrap
        if ($null -eq $bootstrap) { throw "No Python bootstrap executable found (python/py)." }
        & $bootstrap -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Failed to create venv at $VenvPath" }
    }
    $py = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path $py)) { throw "Python exe not found in venv: $py" }
    if (-not (Ensure-Pip -PythonExe $py)) { throw "pip unavailable in venv: $VenvPath" }
    return $py
}

function Install-ReqFile {
    param(
        [string]$PythonExe,
        [string]$ReqFile,
        [string]$Label
    )
    Write-Host ("Installing requirements (" + $Label + "): " + $ReqFile)
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip for $Label venv." }
    & $PythonExe -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements for $Label venv: $ReqFile" }
}

function Collect-PkgsFromMap {
    param([object]$MapObj)
    $out = New-Object System.Collections.Generic.List[string]
    if ($null -eq $MapObj) { return $out }
    foreach ($entry in $MapObj.PSObject.Properties) {
        $val = $entry.Value
        if ($val -is [System.Array]) {
            foreach ($pkg in $val) {
                $p = [string]$pkg
                if (-not [string]::IsNullOrWhiteSpace($p)) { $out.Add($p.Trim()) }
            }
        } else {
            $p = [string]$val
            if (-not [string]::IsNullOrWhiteSpace($p)) { $out.Add($p.Trim()) }
        }
    }
    return $out
}

$loopPy = Ensure-VenvPython -VenvPath $LoopVenvPath
$trainPy = Ensure-VenvPython -VenvPath $TrainVenvPath

Install-ReqFile -PythonExe $loopPy -ReqFile $loopReq -Label "loop"
Install-ReqFile -PythonExe $trainPy -ReqFile $trainReq -Label "training-base"

if (-not $SkipCuda) {
    if (Test-Path $trainReqCuda) {
        Install-ReqFile -PythonExe $trainPy -ReqFile $trainReqCuda -Label "training-cuda"
    } else {
        Write-Warning "CUDA requirements file not found; skipping: $trainReqCuda"
    }
} else {
    Write-Host "SkipCuda set; not installing training CUDA wheel requirements."
}

if (-not $SkipAutoRepairExtras) {
    if (Test-Path $daemonCfg) {
        Write-Host ("Preinstalling auto-repair extras from: " + $daemonCfg)
        $cfg = Get-Content -Path $daemonCfg -Raw | ConvertFrom-Json
        $allPkgs = New-Object System.Collections.Generic.List[string]
        foreach ($pkg in (Collect-PkgsFromMap -MapObj $cfg.auto_repair_module_package_map)) { $allPkgs.Add($pkg) }
        foreach ($pkg in (Collect-PkgsFromMap -MapObj $cfg.auto_repair_module_alias_map)) { $allPkgs.Add($pkg) }
        foreach ($pkg in (Collect-PkgsFromMap -MapObj $cfg.auto_repair_model_package_map)) { $allPkgs.Add($pkg) }
        $pkgList = @($allPkgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
        if ($pkgList.Count -gt 0) {
            & $trainPy -m pip install --disable-pip-version-check @pkgList
            if ($LASTEXITCODE -ne 0) { throw "Failed to install auto-repair extras into training venv." }
            Write-Host ("Installed auto-repair extras: " + ($pkgList -join ", "))
        } else {
            Write-Host "No auto-repair package extras configured."
        }
    } else {
        Write-Warning "daemon_config.json missing; skipping auto-repair extras preinstall."
    }
} else {
    Write-Host "SkipAutoRepairExtras set; not preinstalling auto-repair packages."
}

$loopCount = (& $loopPy -m pip list --format=freeze | Measure-Object).Count
$trainCount = (& $trainPy -m pip list --format=freeze | Measure-Object).Count

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host ("Loop venv: " + $LoopVenvPath + " (packages: " + $loopCount + ")")
Write-Host ("Training venv: " + $TrainVenvPath + " (packages: " + $trainCount + ")")
