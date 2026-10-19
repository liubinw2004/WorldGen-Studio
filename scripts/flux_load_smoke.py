"""T03/T07 load smoke test - prove the assembled FLUX components are loadable.

Layered verification (each layer reported separately, so we know exactly how
far the pipeline path was validated without paying the full 23.8 GB transformer
cost). ``--variant`` selects which assembled repo to test (default ``dev``,
fully backward compatible; ``fill`` covers the i2s path):

  L1  DiffusionPipeline.load_config(local_dir)      -> model_index.json parse + class resolution
  L2  scheduler / tokenizer / tokenizer_2 load      -> config + vocab assets
  L3  VAE (AutoencoderKL) load                       -> real safetensors weights
  L4  text_encoder (CLIP-L) load                     -> real safetensors weights
  L5  Plan-A offline resolution: with HF_HUB_OFFLINE=1,
      DiffusionPipeline.load_config("<variant repo id>") must resolve from the
      local HF cache mirror (no network).
  L6  T5 text_encoder_2 shard presence (weights load is optional / heavy)

Run:
  D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\flux_load_smoke.py [--variant dev|fill] [--load-t5]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)

RESULTS: dict[str, tuple[bool, str]] = {}

SMOKE_VARIANTS: Dict[str, dict] = {
    "dev": {"local": cfg.FLUX_DEV_DIR, "repo_id": "black-forest-labs/FLUX.1-dev",
            "title": "FLUX.1-dev"},
    "fill": {"local": cfg.FLUX_FILL_DIR, "repo_id": "black-forest-labs/FLUX.1-Fill-dev",
             "title": "FLUX.1-Fill-dev"},
}


def rec(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = (ok, detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def l1_load_config(local_dir: Path) -> bool:
    try:
        from diffusers import DiffusionPipeline

        info = DiffusionPipeline.load_config(str(local_dir))
        rec("L1 load_config(local)", True, f"components={sorted(k for k in info if not k.startswith('_'))}")
        return True
    except Exception as exc:  # noqa: BLE001
        rec("L1 load_config(local)", False, f"{exc}\n{traceback.format_exc()}")
        return False


def l2_light(local_dir: Path) -> bool:
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTokenizer, T5TokenizerFast

        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(str(local_dir), subfolder="scheduler")
        tok = CLIPTokenizer.from_pretrained(str(local_dir), subfolder="tokenizer")
        tok2 = T5TokenizerFast.from_pretrained(str(local_dir), subfolder="tokenizer_2")
        rec("L2 scheduler+tokenizers", True,
            f"scheduler={type(sched).__name__}, clip_vocab={tok.vocab_size}, t5_vocab={tok2.vocab_size}")
        return True
    except Exception as exc:  # noqa: BLE001
        rec("L2 scheduler+tokenizers", False, f"{exc}\n{traceback.format_exc()}")
        return False


def l3_vae(local_dir: Path) -> bool:
    try:
        from diffusers import AutoencoderKL
        import torch

        vae = AutoencoderKL.from_pretrained(str(local_dir), subfolder="vae", torch_dtype=torch.float32)
        n = sum(p.numel() for p in vae.parameters())
        rec("L3 vae weights", True, f"AutoencoderKL loaded, params={n:,}")
        return True
    except Exception as exc:  # noqa: BLE001
        rec("L3 vae weights", False, f"{exc}\n{traceback.format_exc()}")
        return False


def l4_clip_text_encoder(local_dir: Path) -> bool:
    try:
        from transformers import CLIPTextModel
        import torch

        te = CLIPTextModel.from_pretrained(str(local_dir), subfolder="text_encoder", torch_dtype=torch.float32)
        n = sum(p.numel() for p in te.parameters())
        rec("L4 text_encoder (CLIP-L)", True, f"CLIPTextModel loaded, params={n:,}")
        return True
    except Exception as exc:  # noqa: BLE001
        rec("L4 text_encoder (CLIP-L)", False, f"{exc}\n{traceback.format_exc()}")
        return False


def l5_plan_a_offline(repo_id: str) -> bool:
    """Plan A: the variant repo id must resolve from the local cache mirror while offline."""
    try:
        from diffusers import DiffusionPipeline

        old = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            info = DiffusionPipeline.load_config(repo_id)
        finally:
            if old is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old
        rec("L5 Plan-A offline resolve", True,
            f"{repo_id} resolved from cache; components={sorted(k for k in info if not k.startswith('_'))}")
        return True
    except Exception as exc:  # noqa: BLE001
        rec("L5 Plan-A offline resolve", False, f"{exc}\n{traceback.format_exc()}")
        return False


def l6_t5_shards(local_dir: Path, load_t5: bool) -> bool:
    try:
        import json

        idx = local_dir / "text_encoder_2/model.safetensors.index.json"
        manifest = json.loads(idx.read_text(encoding="utf-8"))
        shards = sorted(set(manifest.get("weight_map", {}).values()))
        missing = [s for s in shards if not (local_dir / "text_encoder_2" / s).exists()]
        if missing:
            rec("L6 T5 shards", False, f"missing shards: {missing}")
            return False
        if load_t5:
            from transformers import T5EncoderModel
            import torch

            te2 = T5EncoderModel.from_pretrained(str(local_dir), subfolder="text_encoder_2",
                                                 torch_dtype=torch.float16)
            n = sum(p.numel() for p in te2.parameters())
            rec("L6 T5 shards", True, f"shards={shards}; T5EncoderModel loaded params={n:,}")
        else:
            rec("L6 T5 shards", True, f"shards present: {shards} (weights load skipped)")
        return True
    except Exception as exc:  # noqa: BLE001
        rec("L6 T5 shards", False, f"{exc}\n{traceback.format_exc()}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(SMOKE_VARIANTS), default="dev",
                        help="which assembled variant to smoke-test (default: dev)")
    parser.add_argument("--load-t5", action="store_true", help="also load the 9.5GB T5 weights")
    args = parser.parse_args()

    variant = SMOKE_VARIANTS[args.variant]
    local_dir: Path = variant["local"]
    repo_id: str = variant["repo_id"]
    print(f"variant = {args.variant} ({variant['title']})", flush=True)
    print(f"local_dir = {local_dir}", flush=True)
    print(f"repo_id = {repo_id}", flush=True)

    l1_load_config(local_dir)
    l2_light(local_dir)
    l3_vae(local_dir)
    l4_clip_text_encoder(local_dir)
    l5_plan_a_offline(repo_id)
    l6_t5_shards(local_dir, args.load_t5)

    print("-" * 60)
    for k, (ok, _) in RESULTS.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    all_ok = all(v[0] for v in RESULTS.values())
    print(f"FLUX LOAD SMOKE [{args.variant}]: " + ("ALL LAYERS PASSED" if all_ok else "SOME LAYERS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
