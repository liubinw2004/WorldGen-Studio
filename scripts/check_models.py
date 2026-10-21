"""Check that every required model file is in place, and say exactly what to do
about the ones that are not.

Models are not downloaded by this project (they are large, and FLUX.1-dev is
gated behind a licence), so this script is the contract between "I downloaded
something" and "the app can run".

    python scripts/check_models.py
    python scripts/check_models.py --json     # machine-readable

Exit code is 0 when everything required for t2s is present, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

HAS_UI = True
try:
    from colorama import Fore, Style  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001 - colour is optional
    HAS_UI = False


def _snapshot_ok(repo_dir: Path, must_contain: str | None = None) -> bool:
    """An HF cache repo is usable when a snapshot holds real files.

    ``must_contain`` is a glob pattern (e.g. ``*.safetensors``) -- snapshots use
    symlinks into ../../blobs, and on Windows those may be real files, so the
    check matches by pattern rather than by name.
    """
    if not repo_dir.exists():
        return False
    snaps = repo_dir / "snapshots"
    if not snaps.exists():
        return False
    for snap in snaps.iterdir():
        if not snap.is_dir():
            continue
        if must_contain:
            if any(p.is_file() or p.is_symlink() for p in snap.rglob(must_contain)):
                return True
        elif any(p.is_file() or p.is_symlink() for p in snap.rglob("*")):
            return True
    return False


def _dirsize_gb(p: Path) -> float:
    total = 0
    try:
        for f in p.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
    except OSError:
        pass
    return round(total / 1024 ** 3, 2)


def collect() -> list[dict]:
    items: list[dict] = []

    # --- diffusers checkouts -------------------------------------------------
    dev = cfg.MODELS_DIR / "FLUX.1-dev"
    items.append({
        "key": "flux-dev",
        "label": "FLUX.1-dev（文本编码器 + VAE + 调度器）",
        "required": True,
        "ok": (dev / "model_index.json").exists() and (dev / "text_encoder").exists()
              and (dev / "vae").exists(),
        "path": str(dev),
        "size_gb": _dirsize_gb(dev) if dev.exists() else 0,
        "hint": ("从 https://huggingface.co/black-forest-labs/FLUX.1-dev 下载 diffusers 格式\n"
                 "        需要：model_index.json, text_encoder/, text_encoder_2/, tokenizer/,\n"
                 "              tokenizer_2/, vae/, scheduler/\n"
                 "        不需要：flux1-dev.safetensors（23.8GB，4bit 主干另行提供）\n"
                 f"        放置到：{dev}"),
    })

    fill = cfg.MODELS_DIR / "FLUX.1-Fill-dev"
    items.append({
        "key": "flux-fill",
        "label": "FLUX.1-Fill-dev（仅「图生场景」需要）",
        "required": False,
        "ok": (fill / "model_index.json").exists(),
        "path": str(fill),
        "size_gb": _dirsize_gb(fill) if fill.exists() else 0,
        "hint": ("从 https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev 下载\n"
                 f"        放置到：{fill}\n"
                 "        text_encoder 可从 FLUX.1-dev 复制，省 9.5GB"),
    })

    # --- HF hub cache entries ------------------------------------------------
    hf = [
        ("lora", "WorldGen LoRA（t2s + i2s）", True,
         "models--LeoXie--WorldGen", "*.safetensors",
         "huggingface-cli download LeoXie/WorldGen"),
        ("da2", "DA-2 全景深度模型", True,
         "models--haodongli--DA-2", "*.safetensors",
         "huggingface-cli download haodongli/DA-2"),
        ("svdq", "svdq-int4 FLUX 主干（4bit）", True,
         "models--mit-han-lab--svdq-int4-flux.1-dev", "*.safetensors",
         "huggingface-cli download mit-han-lab/svdq-int4-flux.1-dev"),
        ("svdq-fill", "svdq-int4 FLUX Fill 主干（i2s 用）", False,
         "models--mit-han-lab--svdq-int4-flux.1-fill-dev", "*.safetensors",
         "huggingface-cli download mit-han-lab/svdq-int4-flux.1-fill-dev"),
    ]
    for key, label, required, dirname, need, cmd in hf:
        p = cfg.HF_HUB_CACHE / dirname
        items.append({
            "key": key,
            "label": label,
            "required": required,
            "ok": _snapshot_ok(p, need),
            "path": str(p),
            "size_gb": _dirsize_gb(p) if p.exists() else 0,
            "hint": f"运行：{cmd}\n        缓存目录已设为：{cfg.HF_HUB_CACHE}",
        })

    # --- optional super-resolution weights -----------------------------------
    sr_files = {
        "sr-ultra": ("4x-UltraSharp.pth", "4x-UltraSharp（超分·锐利）"),
        "sr-esrgan": ("RealESRGAN_x4plus.pth", "Real-ESRGAN x4plus（超分·柔和）"),
    }
    for key, (fname, label) in sr_files.items():
        p = cfg.SR_DIR / fname
        items.append({
            "key": key,
            "label": label + " [可选]",
            "required": False,
            "ok": p.exists(),
            "path": str(p),
            "size_gb": round(p.stat().st_size / 1024 ** 3, 3) if p.exists() else 0,
            "hint": f"放到 {p}（约 67MB）\n        缺失只影响「画质增强」，不影响正常生成",
        })

    # --- optional Chinese -> English translator ------------------------------
    # Without it a Chinese prompt reaches FLUX as a handful of <unk> tokens and
    # the panorama silently ignores the description, so the failure is invisible
    # unless the checklist calls it out.
    mt_dir = cfg.ROOT / "models" / "mt" / "opus-mt-zh-en"
    mt_ok = (mt_dir / "config.json").exists() and (
        any(mt_dir.glob("*.safetensors")) or any(mt_dir.glob("pytorch_model*.bin")))
    items.append({
        "key": "translate",
        "label": "中文提示词翻译（opus-mt-zh-en）[可选]",
        "required": False,
        "ok": mt_ok,
        "path": str(mt_dir),
        "size_gb": _dirsize_gb(mt_dir) if mt_dir.exists() else 0,
        "hint": "缺失时仍可用英文提示词；中文会被模型忽略\n"
                "        下载：python scripts/fetch_translate_model.py（约 300MB）",
    })

    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()

    items = collect()
    missing_required = [i for i in items if i["required"] and not i["ok"]]

    if args.json:
        print(json.dumps({
            "root": str(cfg.ROOT),
            "ok": not missing_required,
            "items": items,
        }, ensure_ascii=False, indent=2))
        return 0 if not missing_required else 1

    print()
    print(f"  WorldGen Studio 模型检查   项目目录: {cfg.ROOT}")
    print("  " + "=" * 72)
    for it in items:
        mark = "OK  " if it["ok"] else ("缺失" if it["required"] else "可选")
        size = f"{it['size_gb']:.2f} GB" if it["ok"] else "—"
        print(f"  [{mark}] {it['label']}")
        print(f"          {it['path']}")
        print(f"          占用 {size}")
        if not it["ok"]:
            for line in it["hint"].splitlines():
                print(f"          {line}")
        print()

    print("  " + "-" * 72)
    if missing_required:
        print(f"  还缺 {len(missing_required)} 项必需模型，补齐后即可使用。")
    else:
        print("  必需模型齐全，双击 start.bat 启动。")
        optional_missing = [i for i in items if not i["required"] and not i["ok"]]
        if optional_missing:
            names = "、".join(i["label"] for i in optional_missing)
            print(f"  可选未安装：{names}")
    print()
    return 0 if not missing_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
