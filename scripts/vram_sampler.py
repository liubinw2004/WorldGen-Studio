"""T04 helper - sample GPU VRAM usage and record the peak during a demo run.

Polls `nvidia-smi` every `--interval` seconds and tracks the maximum
`memory.used` (MiB) for the target GPU. Stops when the sentinel file
`--until-file` appears (the demo runner touches it after the artifact is
produced) or when `--max-seconds` elapses.

Writes logs/vram_peak.json:
    {"gpu":..., "total_mib":..., "peak_used_mib":..., "peak_free_mib":...,
     "baseline_used_mib":..., "samples":..., "seconds":...}

Usage:
    python D:\\mm\\scripts\\vram_sampler.py --until-file D:\\mm\\logs\\.demo_done --interval 0.5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402


def query(gpu: int) -> tuple[int, int, int, str]:
    out = subprocess.run(
        ["nvidia-smi",
         f"--id={gpu}",
         "--query-gpu=name,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    name, total, used, free = [p.strip() for p in out.split(",")]
    return int(total), int(used), int(free), name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--until-file", default=str(cfg.LOG_DIR / ".demo_done"))
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    ap.add_argument("--out", default=str(cfg.LOG_DIR / "vram_peak.json"))
    args = ap.parse_args()

    until = Path(args.until_file)
    if until.exists():
        until.unlink()

    samples = 0
    peak_used = 0
    peak_free = 10**9
    baseline_used = None
    total = 0
    name = "?"
    t0 = time.time()

    print(f"[vram_sampler] start; until={until} interval={args.interval}s", flush=True)
    while True:
        try:
            total, used, free, name = query(args.gpu)
            if baseline_used is None:
                baseline_used = used
            peak_used = max(peak_used, used)
            peak_free = min(peak_free, free)
            samples += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[vram_sampler] query failed: {exc}", flush=True)

        if until.exists():
            break
        if time.time() - t0 > args.max_seconds:
            print("[vram_sampler] max-seconds reached", flush=True)
            break
        time.sleep(args.interval)

    report = {
        "gpu": name,
        "total_mib": total,
        "baseline_used_mib": baseline_used,
        "peak_used_mib": peak_used,
        "peak_free_mib": (peak_free if peak_free != 10**9 else None),
        "samples": samples,
        "seconds": round(time.time() - t0, 2),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[vram_sampler] peak_used={peak_used} MiB / total={total} MiB "
          f"(baseline={baseline_used}); samples={samples}; -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
