"""Download the Chinese -> English prompt translator (~300 MB).

Optional component: without it the app still generates from English prompts, but
a Chinese prompt reaches FLUX's English-only T5 encoder as a few ``<unk>``
tokens and the description is silently ignored.

Usage::

    python scripts/fetch_translate_model.py            # official endpoint
    python scripts/fetch_translate_model.py --mirror   # hf-mirror.com

The model lands in ``models/mt/opus-mt-zh-en`` and is loaded lazily by
``app/translate.py`` on the first Chinese prompt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

REPO_ID = "Helsinki-NLP/opus-mt-zh-en"
TARGET = ROOT / "models" / "mt" / "opus-mt-zh-en"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true",
                    help="use hf-mirror.com (mainland China)")
    args = ap.parse_args()

    # Keep every byte on the project drive, mirroring config.py's policy.
    os.environ.setdefault("HF_HOME", str(cfg.ROOT / "cache" / "hf"))
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"[fetch] endpoint : {endpoint}")
    print(f"[fetch] target   : {TARGET}")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET),
        allow_patterns=["*.json", "*.txt", "*.model", "*.spm", "*.bin",
                        "*.safetensors"],
        ignore_patterns=["*.gguf", "*.onnx*", "*.msgpack"],
    )

    weight = (any(TARGET.glob("*.safetensors"))
              or any(TARGET.glob("pytorch_model*.bin")))
    if not weight:
        print("[fetch] FAILED: 权重文件没有下载完整")
        return 1
    print("[fetch] done. 中文提示词现在会自动翻译成英文。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
