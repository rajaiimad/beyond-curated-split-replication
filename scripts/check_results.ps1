# Recompute every number of the paper from the archived predictions, without refitting (Windows).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path .env) {
  Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    if ($v.Trim()) { [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process") }
  }
}
python src\fa_stats.py data results
python src\fa_report.py . $env:UFFCS_RAW_DIR $env:UFFCS_SELECTED_DIR $env:UFFCS_DATASET_CSV
python src\fa_verify.py .
