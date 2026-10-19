"""Watchdog wrapper for the WorldGen i2s demo.

Purpose: the documented i2s entry point must fail LOUDLY, not hang silently.
On a 12 GB card a VRAM-paging deadlock (9.5 GB T5 never evicted -> VAE/latent
allocations squeezed into ~125 MiB -> Windows WDDM paging) manifests as:
process alive, GPU 100%, ~120 MiB free, and *zero* log progress forever.

This wrapper:
  * launches the demo (via scripts/run_i2s_patched.py, or raw demo.py when
    I2S_NO_SHIM=1) with stdout+stderr redirected to a log file;
  * streams new log bytes to our own stdout (so the caller's redirect sees the
    live log, including tqdm carriage-return progress);
  * enforces a no-progress timeout: if no new output for I2S_WATCHDOG_SECS
    (default 300 s) AND generation has not finished, it kills the child and
    exits **124** with a self-diagnosis message;
  * optional I2S_KILL_AFTER_ARTIFACT=1: terminate cleanly (exit 0) a few seconds
    after "successfully generated" appears (for automated/CI runs; otherwise the
    process keeps serving viser on :8080, which is the interactive behaviour).

Usage (same args as demo.py):
    python D:\\mm\\scripts\\run_i2s_watchdog.py -i <img> -p "<prompt>" \
        -o <out> -r 1024 --low_vram --save_scene
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# <root>/scripts/run_i2s_watchdog.py -> <root>
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

COMPLETION_MARKER = b"successfully generated"
EXIT_HANG = 124


def _tail(path: Path, n: int = 4000) -> bytes:
    try:
        data = path.read_bytes()
        return data[-n:]
    except OSError:
        return b""


def main() -> int:
    passthrough = sys.argv[1:]
    py = sys.executable
    if os.environ.get("I2S_NO_SHIM") == "1":
        cmd = [py, str(cfg.REPO_DIR / "demo.py")] + passthrough
    else:
        cmd = [py, str(cfg.SCRIPTS_DIR / "run_i2s_patched.py")] + passthrough

    watchdog = float(os.environ.get("I2S_WATCHDOG_SECS", "300"))
    kill_after = os.environ.get("I2S_KILL_AFTER_ARTIFACT") == "1"

    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    # I2S_LOG_NAME lets automated runs (e.g. the 3x stability matrix) keep one
    # log per run instead of overwriting demo_i2s_live.log.
    logfile = cfg.LOG_DIR / os.environ.get("I2S_LOG_NAME", "demo_i2s_live.log")
    if logfile.exists():
        logfile.unlink()

    print(f"[i2s-watchdog] launching: {' '.join(cmd)}", flush=True)
    print(f"[i2s-watchdog] watchdog={watchdog:.0f}s  kill_after_artifact={kill_after}  "
          f"log={logfile}", flush=True)

    env = os.environ.copy()
    # Child print() is block-buffered when redirected to a file, so our
    # "successfully generated" completion marker would never reach the log
    # (only stderr/tqdm flush). Force line-buffering so we can detect completion
    # and exit promptly (otherwise the watchdog would sit until its timeout).
    env["PYTHONUNBUFFERED"] = "1"
    fh = open(logfile, "wb")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(cfg.REPO_DIR), env=env)

    out = sys.stdout.buffer
    offset = 0
    last_size = 0
    last_activity = time.time()
    done = False
    rc = None
    try:
        while True:
            size = logfile.stat().st_size if logfile.exists() else 0
            if size > offset:
                with open(logfile, "rb") as rf:
                    rf.seek(offset)
                    out.write(rf.read(size - offset))
                    out.flush()
                offset = size
            if size != last_size:
                last_size = size
                last_activity = time.time()

            rc = proc.poll()

            if not done and COMPLETION_MARKER in _tail(logfile):
                done = True
                last_activity = time.time()
                print("\n[i2s-watchdog] generation complete (artifact should be on disk).",
                      flush=True)
                if kill_after:
                    time.sleep(3.0)
                    proc.terminate()
                    print("[i2s-watchdog] kill_after_artifact=1 -> terminating viser.",
                          flush=True)

            if rc is not None:
                break

            if (not done) and (time.time() - last_activity) > watchdog:
                print(f"\n[i2s-watchdog] WATCHDOG TRIGGERED: {watchdog:.0f}s with no log "
                      f"progress and generation not finished.", flush=True)
                print("[i2s-watchdog] 疑似显存换页静默卡死：请检查 "
                      "`nvidia-smi` 余量是否 <200 MiB，并确认 post-encode 已将 "
                      "T5 撤离（text_encoder/text_encoder_2 -> CPU）。", flush=True)
                print("[i2s-watchdog] ------- last 20 log lines -------", flush=True)
                tail_txt = _tail(logfile, 6000).decode("utf-8", "replace").splitlines()[-20:]
                for ln in tail_txt:
                    print("   " + ln, flush=True)
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    proc.kill()
                return EXIT_HANG

            time.sleep(0.5)

        # final drain
        rc = proc.wait()
        size = logfile.stat().st_size if logfile.exists() else 0
        if size > offset:
            with open(logfile, "rb") as rf:
                rf.seek(offset)
                out.write(rf.read(size - offset))
                out.flush()
        print(f"\n[i2s-watchdog] child exited rc={rc}", flush=True)
        if done and kill_after:
            # We terminated viser ourselves *after* a verified completion; that
            # is a success, not a failure. Report 0 so CI/automation is correct.
            print("[i2s-watchdog] completion verified; treating post-artifact "
                  "termination as SUCCESS (rc=0)", flush=True)
            return 0
        return rc
    except KeyboardInterrupt:
        print("\n[i2s-watchdog] KeyboardInterrupt -> terminating child.", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        return 130
    finally:
        fh.close()


if __name__ == "__main__":
    sys.exit(main())
