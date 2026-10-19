# verify_c_drive.ps1 - assert the C-drive net footprint stayed within budget.
#
# Re-snapshots the same C-drive targets as preflight.ps1 and compares them with
# the baseline in D:\mm\logs\c_drive_baseline.json. The project is defined to
# write NOTHING to C: (except the read-only use of the system python to *create*
# the venv), so the allowed net increase is a small tolerance for OS churn
# (browser caches, telemetry, other apps) -- default 0.5 GB.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\mm\scripts\verify_c_drive.ps1 [-BudgetGB 0.5]
#
# Exit codes:
#   0 = within budget
#   2 = budget exceeded (net increase > BudgetGB on a tracked target)
#   3 = baseline missing (run preflight.ps1 first)
#
# Writes D:\mm\logs\c_drive_verify.json
param(
    [double]$BudgetGB = 0.5,
    [string]$BaselinePath = "D:\mm\logs\c_drive_baseline.json",
    [string]$OutPath = "D:\mm\logs\c_drive_report.json"
)

$ErrorActionPreference = "Continue"
$BudgetBytes = [double]($BudgetGB * 1GB)

function Measure-Target([string]$Path) {
    $entry = [ordered]@{ path = $Path; exists = (Test-Path $Path); files = 0; bytes = 0 }
    if (Test-Path $Path) {
        try {
            $files = Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue
            if ($files) {
                $entry.files = ($files | Measure-Object).Count
                $entry.bytes = [double]($files | Measure-Object -Property Length -Sum).Sum
            }
        } catch { $entry["error"] = "$_" }
    }
    return $entry
}

Write-Host "==================== C-drive write check ===================="

if (-not (Test-Path $BaselinePath)) {
    Write-Error "[verify_c_drive] baseline not found: $BaselinePath (run preflight.ps1 first)"
    exit 3
}
$baseline = Get-Content -Raw -Path $BaselinePath | ConvertFrom-Json

$targets = @(
    "$env:USERPROFILE\.cache",
    "$env:LOCALAPPDATA\Temp",
    "$env:LOCALAPPDATA\pip",
    "$env:APPDATA\uv"
)

$report = [ordered]@{
    ts           = (Get-Date).ToUniversalTime().ToString("o")
    budget_gb    = $BudgetGB
    baseline_ts  = $baseline.ts
    targets      = @()
    ok           = $true
}

foreach ($t in $targets) {
    $now = Measure-Target $t
    $base = $baseline.targets | Where-Object { $_.path -eq $t } | Select-Object -First 1
    $baseBytes = if ($base) { [double]$base.bytes } else { 0.0 }
    $deltaBytes = [double]($now.bytes - $baseBytes)
    $deltaGB = [math]::Round($deltaBytes / 1GB, 6)
    $within = ($deltaBytes -le $BudgetBytes)

    Write-Host ("  {0}" -f $t)
    Write-Host ("      baseline bytes = {0:N0}   now bytes = {1:N0}   delta = {2:N0} ({3} GB)" -f $baseBytes, $now.bytes, $deltaBytes, $deltaGB)
    if (-not $within) {
        Write-Warning ("      OVER BUDGET (> {0} GB)" -f $BudgetGB)
        $report.ok = $false
    }

    $report.targets += [ordered]@{
        path            = $t
        base_bytes      = $baseBytes
        now_bytes       = $now.bytes
        delta_bytes     = $deltaBytes
        delta_gb        = $deltaGB
        within_budget   = $within
    }
}

$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $OutPath
# Rewrite without BOM so Python's json.loads(utf-8) is happy.
$noBom = New-Object System.Text.UTF8Encoding($false)
$json = $report | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText($OutPath, $json, $noBom)
# Also keep a legacy copy for validate_artifacts.py.
[System.IO.File]::WriteAllText("D:\mm\logs\c_drive_verify.json", $json, $noBom)
Write-Host ("  report written: {0}" -f $OutPath)

Write-Host "=============================================================="
if ($report.ok) {
    Write-Host (" C-DRIVE WRITE CHECK: PASS (all tracked targets within {0} GB)" -f $BudgetGB)
    exit 0
} else {
    Write-Warning (" C-DRIVE WRITE CHECK: FAIL (a tracked target grew by more than {0} GB)" -f $BudgetGB)
    exit 2
}
