param(
    [string]$WorkspaceRoot = "C:\\Users\\Max\\code\\llm_driven_cnns",
    [string]$DataSourceRoot = "C:\\Users\\Max\\code\\xray_fracture_benchmark\\data"
)

$ErrorActionPreference = "Stop"
$target = Join-Path $WorkspaceRoot "data"

if (Test-Path $target) {
    Write-Host ("Data link already exists: " + $target)
    exit 0
}

if (-not (Test-Path $DataSourceRoot)) {
    throw "Data source not found: $DataSourceRoot"
}

New-Item -ItemType Junction -Path $target -Target $DataSourceRoot | Out-Null
Write-Host ("Created data junction: " + $target + " -> " + $DataSourceRoot)
