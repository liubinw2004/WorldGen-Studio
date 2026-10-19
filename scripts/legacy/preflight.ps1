# preflight.ps1 - one-shot readiness check before any heavy install.
#
# Checks:
#   1. D-drive water level (delegates to check_disk.ps1)
#   2. GPU presence, free VRAM (>= 10 GiB wanted for demo; informational here)
#   3. C-drive baseline snapshot (file count + bytes + newest mtime) for the
#      later zero-C-write assertion
#   4. Required tools: git, system python
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\mm\scripts\preflight.ps1
#
# Writes a JSON snapshot to D:\mm\logs\c_drive_baseline.json
param(
    [double]$RequiredGB = 80,
    [string]$Target = "D:\mm",
    [int]$MinFreeVramMiB = 10240
)

$ErrorActionPreference = "Continue"   # never abort mid-snapshot
$LogDir = "D:\mm\logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$overallOk = $true

Write-Host "==================== WorldGen preflight ===================="

# --- 1. disk ---------------------------------------------------------------
Write-Host "`n[1/4] D-drive water level"
& powershell -ExecutionPolicy Bypass -File (Join-Path $scriptDir "check_disk.ps1") -RequiredGB $RequiredGB -Target $Target
$diskOk = ($LASTEXITCODE -eq 0)
if (-not $diskOk) { $overallOk = $false }
Write-Host ("      disk_ok = {0}" -f $diskOk)

# --- 2. GPU ----------------------------------------------------------------
Write-Host "`n[2/4] GPU / VRAM"
$gpuOk = $false
$freeVramMiB = $null
$gpuName = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        $freeVramMiB = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        $totalVramMiB = [int](& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        $gpuName = $gpuName.Trim()
        Write-Host ("      GPU    : {0} (total {1} MiB)" -f $gpuName, $totalVramMiB)
        Write-Host ("      free   : {0} MiB (want >= {1} MiB for demo)" -f $freeVramMiB, $MinFreeVramMiB)
        $gpuOk = $true
    } catch {
        Write-Warning "      nvidia-smi query failed: $_"
    }
} else {
    Write-Warning "      nvidia-smi not found on PATH"
}
if (-not $gpuOk) { $overallOk = $false }

# --- 3. C-drive baseline ---------------------------------------------------
Write-Host "`n[3/4] C-drive baseline snapshot"
$targets = @(
    "$env:USERPROFILE\.cache",
    "$env:LOCALAPPDATA\Temp",
    "$env:LOCALAPPDATA\pip",
    "$env:APPDATA\uv"
)
$snapshot = [ordered]@{
    ts      = (Get-Date).ToUniversalTime().ToString("o")
    targets = @()
}
$cdrive = Get-PSDrive -Name C
$snapshot["c_drive_free_gb"] = [math]::Round(($cdrive.Free / 1GB), 4)
$snapshot["c_drive_used_gb"] = [math]::Round(($cdrive.Used / 1GB), 4)

foreach ($t in $targets) {
    $entry = [ordered]@{ path = $t; exists = (Test-Path $t); files = 0; bytes = 0; newest_mtime = $null }
    if (Test-Path $t) {
        try {
            $files = Get-ChildItem -LiteralPath $t -Recurse -File -ErrorAction SilentlyContinue
            if ($files) {
                $entry.files = ($files | Measure-Object).Count
                $entry.bytes = ($files | Measure-Object -Property Length -Sum).Sum
                $newest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($newest) { $entry.newest_mtime = $newest.LastWriteTime.ToUniversalTime().ToString("o") }
            }
        } catch {
            $entry["error"] = "$_"
        }
    }
    $snapshot["targets"] += $entry
    Write-Host ("      {0} -> exists={1} files={2} bytes={3}" -f $t, $entry.exists, $entry.files, $entry.bytes)
}

$snapshotPath = Join-Path $LogDir "c_drive_baseline.json"
$snapshot | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $snapshotPath
Write-Host ("      baseline written: {0}" -f $snapshotPath)

# --- 4. tools --------------------------------------------------------------
Write-Host "`n[4/4] required tools"
$gitOk = $false
if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitVer = (& git --version)
    Write-Host ("      git    : {0}" -f $gitVer)
    $gitOk = $true
} else {
    Write-Warning "      git NOT found"
}
$py312 = "C:\Users\Han\AppData\Local\Programs\Python\Python312\python.exe"
$pyOk = $false
if (Test-Path $py312) {
    $pyVer = (& $py312 --version 2>&1)
    Write-Host ("      python : {0} ({1})" -f $py312, $pyVer)
    $pyOk = $true
} else {
    Write-Warning ("      system python missing: {0}" -f $py312)
}
if (-not ($gitOk -and $pyOk)) { $overallOk = $false }

Write-Host "`n==================== preflight result ===================="
Write-Host (" disk={0}  gpu={1}  git={2}  python={3}" -f $diskOk, $gpuOk, $gitOk, $pyOk)
if ($overallOk) { Write-Host " PREFLIGHT: OK" } else { Write-Warning " PREFLIGHT: ISSUES FOUND (see above)" }
if ($overallOk) { exit 0 } else { exit 1 }
