# Full pipeline on Windows PowerShell (same steps as run_pipeline.sh).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path .env) {
  Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    if ($v.Trim()) { [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process") }
  }
}
foreach ($k in "UFFCS_RAW_DIR", "UFFCS_SELECTED_DIR", "UFFCS_DATASET_CSV") {
  if (-not [Environment]::GetEnvironmentVariable($k)) { throw "Set $k in .env (see .env.example)" }
}
New-Item -ItemType Directory -Force data, results\logs | Out-Null
python src\fa_data.py $env:UFFCS_RAW_DIR data $env:UFFCS_SELECTED_DIR
python src\fa_run.py $env:UFFCS_RAW_DIR $env:UFFCS_SELECTED_DIR data results
python src\fa_run.py $env:UFFCS_RAW_DIR $env:UFFCS_SELECTED_DIR data results checks control hgbreg posctl_reg null500 null_lr500
python -c "import xgboost" 2>$null
if ($LASTEXITCODE -eq 0) { python src\fa_vrlens.py $env:UFFCS_RAW_DIR $env:UFFCS_SELECTED_DIR $env:UFFCS_DATASET_CSV data results }
else { Write-Host "xgboost not installed: VR-LENS re-implementation skipped" }
python src\fa_stats.py data results
python src\fa_report.py . $env:UFFCS_RAW_DIR $env:UFFCS_SELECTED_DIR $env:UFFCS_DATASET_CSV
python src\fa_verify.py .
