r"""WorldGen Studio - inference job runner.

Wraps WorldGen's demo.py as a supervised child process and turns its very
chatty (tqdm / ANSI / carriage-return) console output into a clean event
stream that the GUI can render as a stage timeline.

Design notes
------------
* One job at a time (hard VRAM constraint: ~11.9 GiB peak on a 12 GiB card).
* stdout is read with raw os.read() and split on \r as well as \n, because
  tqdm redraws its bar with carriage returns -- naive readline() would block
  until the whole 50-step loop finished.
* tqdm progress lines are collapsed into progress events instead of being
  appended to the log, otherwise the UI drowns in 50 near-identical lines.
* i2s goes through scripts/run_i2s_watchdog.py (the T5-eviction shim plus
  silent-hang watchdog) exactly like the CLI entry point does.
* Nothing here writes outside the project root -- every path resolves from it.
"""
from __future__ import annotations

import codecs
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

PY = str(ROOT / "venv" / "Scripts" / "python.exe")
WORLDGEN_DIR = ROOT / "WorldGen"
STUDIO_OUT = ROOT / "outputs" / "studio"
UPLOAD_DIR = ROOT / "app" / "uploads"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
PCT_RE = re.compile(r"(\d+)%\|")
LOAD_PCT_RE = re.compile(r"(?:Loading|Converting)[^|]*\|\s*(\d+)/(\d+)")
SR_DONE_RE = re.compile(r"\[sr-shim\] pano (\d+)x(\d+) -> (\d+)x(\d+)")

MAX_EVENTS = 6000

STAGES = [
    ("init", "装载模型"),
    ("think", "理解输入"),
    ("diffuse", "扩散生成"),
    ("upscale", "超分增强"),
    ("reconstruct", "重建三维"),
    ("ready", "场景就绪"),
]
STAGE_INDEX = {k: i for i, (k, _) in enumerate(STAGES)}

# Panorama upscalers the app can drive. Keys match srlab.upscale.MODELS and the
# WG_SR_MODEL env var the child reads.
SR_MODELS = {
    "ultrasharp": "4x-UltraSharp",
    "realesrgan": "Real-ESRGAN x4plus",
}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _clean(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def _slug(text: str, limit: int = 28) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text or "").strip("-")
    return (text[:limit] or "scene").lower()


class Job:
    """A single generation run plus the event buffer the UI subscribes to.

    Locking: there is none, by design. An earlier version guarded every mutation
    with a ``threading.Lock`` and deadlocked on the first stage transition --
    ``set_stage`` held the lock and then called ``push``, which tried to take it
    again. Because the deadlock hit the log pump, the child's stdout pipe filled
    up and the inference process itself blocked with the GPU pinned at 100%.

    The buffer is a fixed-size deque plus a monotonic counter. ``deque.append``
    and ``int += 1`` are atomic under the GIL, the counter is only ever written
    by the pump thread, and readers take a snapshot with ``list()``. No reader
    can ever stall the writer, which is the property that matters.
    """

    def __init__(self, mode: str, prompt: str, image: str | None,
                 resolution: int, out_dir: Path, sr: str = ""):
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.mode = mode
        self.prompt = prompt
        self.image = image
        self.resolution = resolution
        self.sr = (sr or "").lower()
        self.sr_result = ""          # e.g. "2048x1024", filled from the SR log
        self.out_dir = out_dir
        self.status = "running"
        self.stage = "init"
        self.step = 0
        self.total_steps = 50
        self.pct = 0.0
        self.message = "正在初始化"
        self.error = ""
        self.rc: int | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.total = 0
        self.proc: subprocess.Popen | None = None
        self.last_activity = time.time()
        self._cancelled = False
        self._stalled = False
        # demo.py never exits after generation -- it keeps serving viser on
        # :8080 with `while True: sleep(1)`. So "the child exited" is the wrong
        # completion signal; this flag is set when the log reports success and
        # lets the pump finish while the process stays alive for the viewer.
        self._finished_marker = False

    # -- event plumbing -------------------------------------------------
    def push(self, kind: str, **payload) -> None:
        ev = {"i": self.total, "kind": kind,
              "t": round(time.time() - self.started_at, 2), **payload}
        self.last_activity = time.time()
        self.total += 1
        self.events.append(ev)

    def log(self, text: str, level: str = "info") -> None:
        if text:
            self.push("log", text=text[:2000], level=level)

    def set_stage(self, stage: str, message: str | None = None) -> None:
        """Advance the stage. Never moves backwards."""
        if STAGE_INDEX.get(stage, 0) < STAGE_INDEX.get(self.stage, 0):
            return
        if stage != self.stage:
            self.stage = stage
            if stage == "ready":
                self.pct = max(self.pct, 96.0)
            elif stage == "upscale":
                # Diffusion ends around 86%; the upscale step reports no steps of
                # its own, so pin the ring here rather than letting it look stuck.
                self.pct = max(self.pct, 87.0)
            self.push("stage", stage=stage, label=dict(STAGES).get(stage, stage),
                      pct=self.pct)
        if message:
            self.message = message
            self.push("message", text=message)

    def set_progress(self, step: int, total: int) -> None:
        total = total or 50
        step = max(0, min(step, total))
        self.step, self.total_steps = step, total
        # 6% is reserved for startup so the ring never sits at exactly 0.
        self.pct = round((0.06 + 0.80 * (step / total)) * 100, 1)
        self.push("progress", step=step, total=total, pct=self.pct)

    @property
    def base(self) -> int:
        """Sequence number of the oldest event still buffered."""
        return self.total - len(self.events)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "prompt": self.prompt,
            "image": self.image,
            "resolution": self.resolution,
            "sr": self.sr,
            "sr_label": SR_MODELS.get(self.sr, ""),
            "sr_result": self.sr_result,
            "out_dir": str(self.out_dir),
            "status": self.status,
            "stage": self.stage,
            "stage_label": dict(STAGES).get(self.stage, self.stage),
            "stage_index": STAGE_INDEX.get(self.stage, 0),
            "stage_count": len(STAGES),
            "step": self.step,
            "total_steps": self.total_steps,
            "pct": self.pct,
            "message": self.message,
            "error": self.error,
            "rc": self.rc,
            "started_at": self.started_at,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
            "event_base": self.base,
            "event_count": self.total,
        }

    def drain(self, cursor: int) -> tuple[list[dict], int]:
        """Return events at or after ``cursor`` plus the next cursor."""
        start = self.base
        if cursor < start:
            cursor = start
        snapshot = list(self.events)
        offset = cursor - start
        return snapshot[offset:], self.total


class Runner:
    def __init__(self) -> None:
        self.job: Job | None = None
        self.lock = threading.Lock()
        self.gpu = {"used": 0, "total": 0, "util": 0, "temp": 0, "name": "", "ok": False}
        self.gpu_ts = 0.0
        self._gpu_thread = None
        self._start_gpu_poll()

    # -- gpu telemetry --------------------------------------------------
    def _start_gpu_poll(self) -> None:
        """Poll nvidia-smi without ever blocking on it.

        nvidia-smi takes ~0.3 s on an idle card but was measured at >10 s while
        the GPU runs at 100% -- which is exactly when the UI polls most. A
        blocking ``subprocess.run`` there piles up processes and starves the
        request threads, so the child is spawned and reaped asynchronously and
        a slow sample is simply skipped.
        """
        q = "memory.used,memory.total,utilization.gpu,temperature.gpu,name"
        cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]

        def loop() -> None:
            proc = None
            deadline = 0.0
            while True:
                try:
                    if proc is None:
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        deadline = time.time() + 8.0
                    elif proc.poll() is not None:
                        out = proc.stdout.read() if proc.stdout else ""
                        proc.wait()
                        self._apply_gpu(out)
                        proc = None
                    elif time.time() > deadline:
                        proc.kill()
                        proc = None
                except Exception:  # noqa: BLE001 - telemetry must never crash the app
                    proc = None
                time.sleep(0.4)

        self._gpu_thread = threading.Thread(target=loop, daemon=True)
        self._gpu_thread.start()

    def _apply_gpu(self, out: str) -> None:
        if not out or not out.strip():
            return
        try:
            parts = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            self.gpu_ts = time.time()
            self.gpu = {
                "used": int(float(parts[0])),
                "total": int(float(parts[1])),
                "util": int(float(parts[2])),
                "temp": int(float(parts[3])),
                "name": parts[4],
                "ok": True,
            }
        except (ValueError, IndexError):
            pass

    # -- job control ----------------------------------------------------
    def current(self) -> Job | None:
        with self.lock:
            return self.job

    def busy(self) -> bool:
        j = self.current()
        return bool(j and j.status == "running")

    def reap_idle_child(self) -> bool:
        """Public wrapper: free the viewer of a finished job.

        A completed run keeps demo.py alive to serve viser, which holds ~6.6 GB.
        Callers must release it before asking "is there enough VRAM for a new
        run?", otherwise the honest answer is always no.
        """
        prev = self.job
        if prev and prev.status != "running" and prev.proc and prev.proc.poll() is None:
            self._reap_previous()
            return True
        return False

    def _reap_previous(self) -> None:
        """Kill the previous child, which is still serving viser on :8080.

        demo.py is designed to stay alive after generating so the browser can
        explore the scene. That leftover process holds both port 8080 and
        several GB of VRAM, so it has to go before the next run can start.
        """
        prev = self.job
        if prev and prev.proc and prev.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(prev.proc.pid)],
                               capture_output=True, timeout=25,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:  # noqa: BLE001
                try:
                    prev.proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def start(self, mode: str, prompt: str, image: str | None,
              resolution: int, sr: str = "") -> Job:
        with self.lock:
            if self.job and self.job.status == "running":
                raise RuntimeError("已有任务在运行，请先停止或等待完成")
            self._reap_previous()

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            suffix = f"-sr{self._sr_short(sr)}" if sr else ""
            out_dir = STUDIO_OUT / f"{stamp}-{mode}-{_slug(prompt)}{suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)

            job = Job(mode, prompt, image, resolution, out_dir, sr=sr)
            self.job = job

        t = threading.Thread(target=self._run, args=(job,), daemon=True)
        t.start()
        return job

    @staticmethod
    def _sr_short(sr: str) -> str:
        return {"ultrasharp": "ultra", "realesrgan": "esrgan"}.get((sr or "").lower(), "")

    def _build_cmd(self, job: Job) -> list[str]:
        common = [
            "-p", job.prompt,
            "-o", str(job.out_dir),
            "-r", str(job.resolution),
            "--low_vram",
            "--save_scene",
        ]
        if job.mode == "i2s":
            common = ["-i", job.image or ""] + common
            return [PY, str(ROOT / "scripts" / "run_i2s_watchdog.py")] + common

        # t2s runs through the same VRAM shim: it evicts the 9.5 GB T5-XXL
        # after prompt encoding. Without that eviction the peak sits at
        # ~11.95 GB of 12.29 GB -- close enough that the desktop compositor
        # alone can tip it into silent WDDM paging. Set WG_T2S_SHIM=0 to
        # reproduce the vanilla path.
        if os.environ.get("WG_T2S_SHIM", "1") != "0":
            return [PY, str(ROOT / "scripts" / "run_i2s_patched.py")] + common
        return [PY, str(WORLDGEN_DIR / "demo.py")] + common

    def _env(self, job: Job) -> dict:
        env = os.environ.copy()
        env.update({k: str(v) for k, v in cfg.ENV_VARS.items()})
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # One log file per run keeps the CLI logs intact for later forensics.
        env["I2S_LOG_NAME"] = f"studio-{job.id}.log"
        env["PYTHONWARNINGS"] = "ignore"
        # Panorama super-resolution. The child reads these in
        # vendor/worldgen_sr_shim; an empty WG_SR_MODEL means "leave the
        # panorama at the resolution the diffusion model produced".
        if job.sr:
            env["WG_SR_MODEL"] = job.sr
            env["WG_SR_SCALE"] = os.environ.get("WG_SR_SCALE", "2")
            env["WG_SR_MAX_WIDTH"] = os.environ.get("WG_SR_MAX_WIDTH", "2048")
        else:
            env.pop("WG_SR_MODEL", None)
        return env

    def _run(self, job: Job) -> None:
        cmd = self._build_cmd(job)
        job.log(f"$ {' '.join(cmd)}", "cmd")
        job.set_stage("init", "正在装载模型权重" if job.mode == "t2s" else "正在装载 FLUX.1-Fill 与分片")

        # The child writes to a file rather than a pipe: a pipe applies
        # back-pressure (if the reader ever stalls, the child blocks inside
        # print() and the GPU sits at 100% forever), and it can also stay open
        # via an inherited handle after the process is killed, which would hang
        # the pump and leave the job stuck in "running".
        log_path = job.out_dir / "run.log"
        job.log_path = log_path
        log_fh = open(log_path, "wb")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORLDGEN_DIR),
                env=self._env(job),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # noqa: BLE001
            log_fh.close()
            job.status = "failed"
            job.error = f"无法启动推理进程: {exc}"
            job.log(job.error, "error")
            job.finished_at = time.time()
            job.push("done", status="failed", error=job.error)
            return

        job.proc = proc
        threading.Thread(target=self._stall_watch, args=(job, proc), daemon=True).start()
        self._pump(job, proc, log_path)
        if job._finished_marker and proc.poll() is None:
            # Generated successfully; the child keeps running to serve viser.
            rc = 0
        else:
            rc = proc.wait()
        try:
            log_fh.close()
        except OSError:
            pass
        job.rc = rc
        job.finished_at = time.time()

        ply = job.out_dir / "splat.ply"
        if job._stalled:
            job.status = "failed"
            job.error = job.error or (
                "推理进程长时间没有任何输出，已判定为显存换页卡死并自动终止。"
                "请关闭占用显存的程序，或改用 1024 分辨率重试。")
        elif job._cancelled:
            job.status = "cancelled"
            job.message = "已停止"
        elif rc == 0 and ply.exists() and ply.stat().st_size > 0:
            job.status = "succeeded"
            job.set_stage("ready", "场景已生成")
            job.pct = 100.0
        elif rc == 124:
            job.status = "failed"
            job.error = "看门狗判定显存卡死（无日志进展超过阈值）。请确认其他 GPU 程序已关闭后重试。"
        else:
            job.status = "failed"
            job.error = job.error or f"推理进程退出码 {rc}"

        if job.status != "succeeded" and not job.error:
            job.error = "生成未产出 splat.ply"
        if job.status != "succeeded":
            job.log(job.error, "error")

        meta = {
            "id": job.id,
            "mode": job.mode,
            "prompt": job.prompt,
            "image": job.image,
            "resolution": job.resolution,
            "sr": job.sr,
            "sr_result": job.sr_result,
            "created": _now_iso(),
            "elapsed": round((job.finished_at or time.time()) - job.started_at, 1),
            "status": job.status,
            "rc": rc,
            "splat_bytes": ply.stat().st_size if ply.exists() else 0,
        }
        try:
            (job.out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

        job.push("done", status=job.status, error=job.error,
                 elapsed=meta["elapsed"], out_dir=str(job.out_dir),
                 splat=str(ply) if ply.exists() else "",
                 splat_bytes=meta["splat_bytes"])

    def _stall_watch(self, job: Job, proc: subprocess.Popen) -> None:
        """Kill the child when it stops producing output.

        On Windows a VRAM overcommit does not raise OOM: the process simply
        pages through WDDM and sits at 100% GPU with no output, forever. t2s
        has no watchdog script of its own, so the app must supply one.
        Without this the UI would spin until the user gives up.
        """
        limit = float(os.environ.get("WG_STALL_SECS", "480"))
        while proc.poll() is None:
            time.sleep(10.0)
            if job._finished_marker:
                return
            idle = time.time() - job.last_activity
            if idle <= limit:
                continue
            job._stalled = True
            job.error = (
                f"推理进程连续 {int(idle)} 秒没有任何输出，已判定为显存换页卡死并自动终止。"
                "请关闭其它占用显卡的程序，或降低分辨率后重试。")
            job.log(job.error, "error")
            job.push("message", text="检测到卡死，正在终止进程")
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=25,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            return

    # -- output pump ----------------------------------------------------
    def _pump(self, job: Job, proc: subprocess.Popen, log_path: Path) -> None:
        """Stream the child's log file into the event buffer.

        tqdm redraws its bar with carriage returns, so lines are split on \r as
        well as \n -- reading line-by-line would only surface the bar once every
        50 steps.
        """
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        offset = 0
        buf = ""
        idle = 0
        while True:
            try:
                size = log_path.stat().st_size
            except OSError:
                size = 0
            if size > offset:
                try:
                    with open(log_path, "rb") as rf:
                        rf.seek(offset)
                        data = rf.read(size - offset)
                    offset = size
                    buf += dec.decode(data)
                    parts = re.split(r"\r\n|\r|\n", buf)
                    buf = parts.pop()
                    for raw in parts:
                        self._on_line(job, raw)
                except OSError:
                    pass
            else:
                if proc.poll() is not None:
                    # Process gone and no new bytes: flush the tail and stop.
                    if buf:
                        self._on_line(job, buf)
                    return
                idle += 1
                # Success marker seen and nothing new for ~0.7 s -> the artifact
                # is on disk. Stop tailing but leave the child serving viser.
                if job._finished_marker and idle >= 6:
                    if buf:
                        self._on_line(job, buf)
                    return
            time.sleep(0.12)

    def _on_line(self, job: Job, raw: str) -> None:
        """Classify one child log line into a stage, progress or log event."""
        line = _clean(raw)
        if not line:
            return

        # 1. the 50-step denoise bar -> progress event (never a log line)
        m = STEP_RE.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            if total >= 10:
                job.set_stage("diffuse", "正在扩散生成全景图")
                job.set_progress(step, total)
                return

        # 2. component/shard loading bars -> startup progress, no log spam
        lm = LOAD_PCT_RE.search(line)
        if lm:
            if job.stage == "init":
                job.set_progress(int(lm.group(1)), int(lm.group(2)) or 7)
            return

        if not re.search(r"[\w\u4e00-\u9fff]", line):
            return

        # 3. stage transitions from known WorldGen markers
        low = line.lower()
        if "generating the world" in low:
            job.set_stage("think", "正在解析提示词并编码")
        elif "pre-encode" in low or "post-encode" in low:
            job.set_stage("think", "文本编码完成，准备扩散")
        elif "[sr-shim] diffusion released" in low:
            job.set_stage("upscale", "正在把全景超分到高分辨率")
        elif SR_DONE_RE.search(line):
            m = SR_DONE_RE.search(line)
            job.sr_result = f"{m.group(3)}x{m.group(4)}"
            job.set_stage("upscale", f"超分完成 · 全景 {job.sr_result}")
        elif ("depth" in low and ("->" in line or "pred" in low)
              and "-shim]" not in low):
            # The shim lines ("depth model -> CPU before diffusion") also contain
            # "depth" and "->"; they are VRAM bookkeeping, not the depth stage.
            job.set_stage("think", "正在估计单目深度")
        elif "splat_utils" in low or "gaussian" in low or "covariance" in low:
            job.set_stage("reconstruct", "正在构建高斯点云")
        elif "successfully generated" in low:
            job._finished_marker = True
            job.set_stage("ready", "场景已生成")
        elif "open your browser" in low:
            job.set_stage("ready", "查看器已就绪")

        # 4. log level
        if any(k in low for k in ("error", "failed", "traceback", "out of memory",
                                  "watchdog triggered", "cuda error")):
            level = "error"
            if "failed to load safetensors using method mio" in low:
                level = "warn"  # known-benign nunchaku fallback
        elif any(k in low for k in ("warning", "warn", "deprecat")):
            level = "warn"
        else:
            level = "info"
        job.log(line, level)

    # -- cancellation ---------------------------------------------------
    def cancel(self) -> bool:
        job = self.current()
        if not job or job.status != "running" or not job.proc:
            return False
        job._cancelled = True
        job.message = "正在停止…"
        job.push("message", text="收到停止指令，正在终止推理进程")
        pid = job.proc.pid
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:  # noqa: BLE001
            try:
                job.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        return True


RUNNER = Runner()
