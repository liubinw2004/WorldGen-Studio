r"""WorldGen Studio - local web application server.

A thin Flask layer over `runner.py`:
  * serves the single-page UI
  * exposes /api/generate, /api/cancel, /api/upload, /api/history
  * streams job events over SSE (/api/events)
  * proxies GPU telemetry and drives Explorer / downloads

Runs on 127.0.0.1 only. No outbound network, no C: writes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from flask import Flask, Response, jsonify, request, send_file, send_from_directory  # noqa: E402

import runner  # noqa: E402
import translate as translator  # noqa: E402

APP_DIR = ROOT / "app"
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR / "uploads"
STUDIO_OUT = ROOT / "outputs" / "studio"

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ASSET_ROOTS = [UPLOAD_DIR.resolve(), (ROOT / "cache" / "hf").resolve()]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _safe_under(path: Path, roots: list[Path]) -> bool:
    try:
        p = path.resolve()
    except OSError:
        return False
    return any(p == r or r in p.parents for r in roots)


def _human_size(n: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if n < step or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= step
    return f"{n:.1f} GB"


def _read_meta(d: Path) -> dict:
    mp = d / "meta.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _prompt_from_log(d: Path) -> str:
    """Recover the prompt from a run log when meta.json was never written.

    Older runs (and runs that failed mid-way) have no meta.json, but the log's
    first line always echoes the full argv.
    """
    try:
        with open(d / "run.log", "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
        m = re.search(r'-p\s+"([^"]{1,400})"', head)
        return m.group(1) if m else ""
    except OSError:
        return ""


def _shot_count(ply: Path) -> int:
    """Gaussian count == vertex count of the ply header (cheap, no parsing)."""
    try:
        with open(ply, "rb") as fh:
            head = fh.read(4096).decode("ascii", "replace")
        m = re.search(r"element vertex (\d+)", head)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/uploads/<path:name>")
def uploads(name: str):
    return send_from_directory(UPLOAD_DIR, name)


@app.get("/api/asset")
def api_asset():
    raw = request.args.get("path", "")
    p = Path(raw)
    if not p.exists() or not _safe_under(p, ASSET_ROOTS):
        return jsonify({"ok": False, "error": "not allowed"}), 403
    return send_file(p)


# --------------------------------------------------------------------------
# state / telemetry
# --------------------------------------------------------------------------
@app.get("/api/state")
def api_state():
    job = runner.RUNNER.current()
    return jsonify({
        "ok": True,
        "job": job.snapshot() if job else None,
        "gpu": runner.RUNNER.gpu,
        "server_time": datetime.now().strftime("%H:%M:%S"),
    })


@app.get("/api/system")
def api_system():
    venv_py = ROOT / "venv" / "Scripts" / "python.exe"
    models = [
        ("FLUX.1-dev (t2s)", ROOT / "models" / "FLUX.1-dev"),
        ("FLUX.1-Fill-dev (i2s)", ROOT / "models" / "FLUX.1-Fill-dev"),
    ]
    packs = []
    for label, path in models:
        packs.append({
            "label": label,
            "ready": path.exists() and any(path.iterdir()) if path.exists() else False,
            "path": str(path),
        })

    # Panorama upscalers: the UI greys out a switch whose weights are missing
    # rather than letting the user pick it and fail 40 s into a run.
    sr = []
    for key, meta in runner.SR_MODELS.items():
        p = ROOT / "models" / "sr" / (
            "4x-UltraSharp.pth" if key == "ultrasharp" else "RealESRGAN_x4plus.pth")
        sr.append({"key": key, "label": meta, "ready": p.exists(),
                   "bytes": p.stat().st_size if p.exists() else 0})

    return jsonify({
        "ok": True,
        "gpu": runner.RUNNER.gpu,
        "python": str(venv_py),
        "python_ok": venv_py.exists(),
        "offline": True,
        "models": packs,
        "sr": sr,
        # Chinese prompts are machine-translated before they reach FLUX; the UI
        # shows this in the system panel so a missing model is diagnosable.
        "translator": {
            "ready": translator.model_ready(),
            "label": "中译英 (opus-mt-zh-en)",
            "path": str(translator.MODEL_DIR),
        },
        "outputs": str(STUDIO_OUT),
    })


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@app.post("/api/generate")
def api_generate():
    data = request.get_json(silent=True) or {}
    mode = (data.get("mode") or "t2s").lower()
    prompt = (data.get("prompt") or "").strip()
    image = data.get("image") or None
    try:
        resolution = int(data.get("resolution") or (1024 if mode == "i2s" else 1600))
    except (TypeError, ValueError):
        resolution = 1024

    sr = (data.get("sr") or "").strip().lower()
    if sr in ("off", "none", "0", "false"):
        sr = ""
    if sr and sr not in runner.SR_MODELS:
        return jsonify({"ok": False, "error": f"未知的超分模型 {sr}"}), 400

    if mode not in {"t2s", "i2s"}:
        return jsonify({"ok": False, "error": "未知模式"}), 400

    if mode == "i2s":
        if not image or not Path(image).exists():
            return jsonify({"ok": False, "error": "请先选择一张图片"}), 400
        if resolution > 1024:
            return jsonify({
                "ok": False,
                "error": "12GB 显存下图像模式必须使用 1024 分辨率（1600 会直接 OOM）",
            }), 400
        if not prompt:
            prompt = "a beautiful scene"
        # map_image_to_pano already produces a 2048x1024 panorama, which is the
        # upscaler's ceiling -- running SR here would load a model and return the
        # input unchanged. Ignore the flag instead of pretending it did something.
        sr = ""
    else:
        if not prompt:
            return jsonify({"ok": False, "error": "请输入场景描述"}), 400
        if resolution not in (1024, 1280, 1600):
            resolution = 1600

    # Chinese has no representation in FLUX's English-only T5 encoder (a typical
    # sentence tokenises to a handful of <unk>), so translate before the job is
    # built and report the English that was actually used back to the UI.
    prompt_zh = ""
    if translator.has_chinese(prompt):
        prompt_zh = prompt
        prompt, translated = translator.translate_prompt(prompt)
        if not translated:
            # Refuse rather than silently generate something unrelated.
            return jsonify({
                "ok": False,
                "error": (
                    "中文提示词需要翻译模型，但当前不可用。"
                    "请运行 python scripts/fetch_translate_model.py 下载（约 300MB），"
                    "或直接改用英文描述。"
                ),
            }), 400

    if not runner.RUNNER.busy():
        # A finished run keeps its viewer alive (~6.6 GB). Release it first,
        # then let the telemetry thread take a fresh sample.
        if runner.RUNNER.reap_idle_child():
            time.sleep(1.6)

    gpu = runner.RUNNER.gpu
    stale = (time.time() - runner.RUNNER.gpu_ts) > 8.0
    if gpu.get("ok") and gpu.get("total") and not stale:
        free = gpu["total"] - gpu["used"]
        # 1600 saturates the card on its own (measured peak 11,972 MiB); the
        # app window and the desktop need headroom too. 1024 leaves ~2 GB spare.
        need = 11700 if resolution >= 1600 else 10200
        if free < need and not runner.RUNNER.busy():
            return jsonify({
                "ok": False,
                "error": (
                    f"可用显存仅 {free} MiB，低于 {need} MiB 门槛。"
                    "请关闭其它占用 GPU 的程序（浏览器硬件加速、游戏、其它推理进程）"
                    f"或改用 1024 分辨率后重试。"
                ),
            }), 409

    try:
        job = runner.RUNNER.start(mode, prompt, image, resolution, sr=sr,
                                  prompt_zh=prompt_zh)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409

    return jsonify({"ok": True, "job": job.snapshot()})


@app.post("/api/cancel")
def api_cancel():
    ok = runner.RUNNER.cancel()
    return jsonify({"ok": ok, "message": "已发送停止指令" if ok else "当前没有可停止的任务"})


@app.get("/api/events")
def api_events():
    since = request.args.get("since", default=0, type=int)

    def gen():
        cursor = since
        job_id = None
        last_ping = time.time()
        while True:
            job = runner.RUNNER.current()
            if job is None:
                if time.time() - last_ping > 10:
                    yield ": idle\n\n"
                    last_ping = time.time()
                time.sleep(0.3)
                continue

            if job.id != job_id:
                job_id = job.id
                cursor = job.base
                yield sse({"kind": "reset", "job": job.snapshot()})

            events, nxt = job.drain(cursor)
            if events:
                cursor = nxt
                last_ping = time.time()
                for ev in events:
                    yield sse(ev)
            else:
                if time.time() - last_ping > 10:
                    yield ": ping\n\n"
                    last_ping = time.time()
                time.sleep(0.18)

    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------
@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "没有收到文件"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"ok": False, "error": f"不支持的格式 {ext}"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}{ext}"
    dest = UPLOAD_DIR / name
    f.save(dest)

    try:
        from PIL import Image
        with Image.open(dest) as im:
            im.verify()
        with Image.open(dest) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": "图片无法解码，请换一张"}), 400

    return jsonify({
        "ok": True,
        "name": name,
        "path": str(dest),
        "url": f"/uploads/{name}",
        "width": w,
        "height": h,
        "size": dest.stat().st_size,
    })


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------
@app.get("/api/history")
def api_history():
    items = []
    if STUDIO_OUT.exists():
        for d in sorted(STUDIO_OUT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            ply = d / "splat.ply"
            if not ply.exists():
                continue
            meta = _read_meta(d)
            try:
                st = ply.stat()
            except OSError:
                continue
            prompt = meta.get("prompt") or _prompt_from_log(d)
            img = meta.get("image") or ""
            thumb = f"/api/asset?path={img}" if img and Path(img).exists() else ""
            items.append({
                "id": meta.get("id") or d.name,
                "dir": str(d),
                "mode": meta.get("mode", "t2s"),
                "prompt": prompt,
                # Kept separate from `prompt` (the English that ran) so the
                # library can show what the user actually typed.
                "prompt_zh": meta.get("prompt_zh") or "",
                "resolution": meta.get("resolution", 1024),
                "sr": meta.get("sr", ""),
                "sr_result": meta.get("sr_result", ""),
                "created": meta.get("created") or datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%S"),
                "mtime": st.st_mtime,
                "bytes": st.st_size,
                "size": _human_size(st.st_size),
                "gaussians": _shot_count(ply),
                "elapsed": meta.get("elapsed"),
                "status": meta.get("status", "succeeded"),
                "thumb": thumb,
            })
    return jsonify({"ok": True, "items": items[:60]})


@app.get("/api/download")
def api_download():
    raw = request.args.get("path", "")
    p = Path(raw)
    roots = [(ROOT / "outputs").resolve()]
    if not p.exists() or not _safe_under(p, roots):
        return jsonify({"ok": False, "error": "not allowed"}), 403
    return send_file(p, as_attachment=True, download_name=p.name)


@app.post("/api/reveal")
def api_reveal():
    data = request.get_json(silent=True) or {}
    raw = data.get("path", "")
    p = Path(raw)
    if not p.exists():
        return jsonify({"ok": False, "error": "路径不存在"}), 404
    if not _safe_under(p, [(ROOT / "outputs").resolve(), UPLOAD_DIR.resolve()]):
        return jsonify({"ok": False, "error": "不允许访问该路径"}), 403
    try:
        if p.is_dir():
            subprocess.Popen(["explorer", str(p)])
        else:
            subprocess.Popen(["explorer", "/select,", str(p)])
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/open-external")
def api_open_external():
    """Open the viser viewer in a real browser window (outside the app frame)."""
    import webbrowser
    webbrowser.open("http://localhost:8080")
    return jsonify({"ok": True})


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "pid": os.getpid()})


def main() -> int:
    for d in (STUDIO_OUT, UPLOAD_DIR, APP_DIR / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    # Under pythonw.exe both streams are None; redirect to a log file so that
    # print() never raises and the launcher has something to show on failure.
    if sys.stdout is None or sys.stderr is None:
        logfile = (APP_DIR / "logs" / "server.log").open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or logfile
        sys.stderr = sys.stderr or logfile

    # Restore the D-drive-only environment before anything spawns.
    os.environ.update({k: str(v) for k, v in cfg.ENV_VARS.items()})

    port = int(os.environ.get("WG_STUDIO_PORT", "7870"))
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] WorldGen Studio -> http://127.0.0.1:{port}", flush=True)
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
