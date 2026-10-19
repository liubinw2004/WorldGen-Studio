# gpu_precheck.ps1 - fail-fast GPU readiness gate for the WorldGen demo.
#
# Asserts free VRAM >= NeedMiB (default 10240 MiB ~ 10 GiB) and, when short,
# prints the processes currently holding GPU memory so the user can free them.
#
# Exit codes:
#   0 = ready (free >= NeedMiB)
#   2 = NOT ready (insufficient free VRAM) -> caller must abort
#   3 = nvidia-smi unavailable
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\mm\scripts\gpu_precheck.ps1 [-NeedMiB 10240]
param(
    [int]$NeedMiB = 10240,
    [int]$GpuIndex = 0
)

$ErrorActionPreference = "Continue"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Error "[gpu_precheck] nvidia-smi not found on PATH"
    exit 3
}

Write-Host "==================== GPU precheck ===================="
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader -i $GpuIndex 2>$null | Select-Object -First 1)
    $freeMiB = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GpuIndex 2>$null | Select-Object -First 1)
    $totalMiB = [int](& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GpuIndex 2>$null | Select-Object -First 1)
    $usedMiB = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GpuIndex 2>$null | Select-Object -First 1)
} catch {
    Write-Error "[gpu_precheck] failed to query nvidia-smi: $_"
    exit 3
}

Write-Host ("  GPU      : {0} (total {1} MiB)" -f $gpuName.Trim(), $totalMiB)
Write-Host ("  used     : {0} MiB" -f $usedMiB)
Write-Host ("  free     : {0} MiB (need >= {1} MiB)" -f $freeMiB, $NeedMiB)

if ($freeMiB -ge $NeedMiB) {
    Write-Host "[gpu_precheck] READY"
    exit 0
}

Write-Warning ("[gpu_precheck] NOT READY: free {0} MiB < required {1} MiB" -f $freeMiB, $NeedMiB)
Write-Host "  Processes currently holding GPU memory:"
try {
    & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} catch {
    Write-Host "  (could not enumerate compute apps)"
}
Write-Host "  -> Please close the apps above to free VRAM, then retry."
exit 2
