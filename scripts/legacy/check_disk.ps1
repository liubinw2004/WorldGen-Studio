# check_disk.ps1 - D-drive water-level guard (standalone, re-runnable).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\mm\scripts\check_disk.ps1 [-RequiredGB 80] [-Target D:\mm]
#
# Exit codes:
#   0 = OK (free - (required + margin) >= 0)
#   2 = INSUFFICIENT (would breach safety margin)
#   3 = DRIVE MISSING
#
# Contract (docs/system_design.md section 8.6):
#   any download/install MUST call this first; abort when free < required + 20GB.
param(
    [double]$RequiredGB = 0,
    [string]$Target = "D:\mm",
    [double]$MarginGB = 20,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

try {
    $driveLetter = ([System.IO.Path]::GetPathRoot($Target)).TrimEnd('\').TrimEnd(':')
    if ([string]::IsNullOrEmpty($driveLetter)) { $driveLetter = "D" }
    $drive = Get-PSDrive -Name $driveLetter -ErrorAction Stop
} catch {
    Write-Error "[check_disk] DRIVE MISSING: $Target ($driveLetter)"
    exit 3
}

$freeGB = [math]::Round(($drive.Free / 1GB), 2)
$usedGB = [math]::Round(($drive.Used / 1GB), 2)
$totalGB = [math]::Round((($drive.Used + $drive.Free) / 1GB), 2)
$needGB = $RequiredGB + $MarginGB

$result = [ordered]@{
    ts         = (Get-Date).ToUniversalTime().ToString("o")
    target     = $Target
    drive      = "$driveLetter`:"
    total_gb   = $totalGB
    used_gb    = $usedGB
    free_gb    = $freeGB
    required_gb = $RequiredGB
    margin_gb  = $MarginGB
    need_gb    = $needGB
    ok         = ($freeGB -ge $needGB)
}

if (-not $Quiet) {
    Write-Host ("[check_disk] {0}  free={1} GB (need {2} GB = required {3} + margin {4})" -f `
        $Target, $freeGB, $needGB, $RequiredGB, $MarginGB)
}

if ($result.ok) {
    if (-not $Quiet) { Write-Host "[check_disk] OK" }
    exit 0
} else {
    Write-Warning ("[check_disk] INSUFFICIENT: free {0} GB < need {1} GB on {2}" -f $freeGB, $needGB, $Target)
    exit 2
}
