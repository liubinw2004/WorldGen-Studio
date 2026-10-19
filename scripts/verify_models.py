"""T03 verification - assert every tier-1 model is present and complete.

Checks (against real measured byte sizes, no fabrication):
    1. LeoXie/WorldGen   : both LoRA safetensors present & non-trivial
    2. haodongli/DA-2    : model.safetensors present
    3. svdq-int4-flux.1-dev : transformer_blocks + unquantized_layers + configs
    4. FLUX.1-dev components : all 18 files present with expected sizes

Results are written to logs/T03-verify.json and written back into state.json.

Run:  D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\verify_models.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)

# Expected FLUX component sizes (from the ModelScope repo listing, bytes).
FLUX_EXPECTED: Dict[str, int] = {
    "model_index.json": 536,
    "scheduler/scheduler_config.json": 273,
    "text_encoder/config.json": 613,
    "text_encoder/model.safetensors": 246_144_352,
    "text_encoder_2/config.json": 782,
    "text_encoder_2/model-00001-of-00002.safetensors": 4_994_582_224,
    "text_encoder_2/model-00002-of-00002.safetensors": 4_530_066_360,
    "text_encoder_2/model.safetensors.index.json": 19_885,
    "tokenizer/merges.txt": 524_619,
    "tokenizer/special_tokens_map.json": 588,
    "tokenizer/tokenizer_config.json": 705,
    "tokenizer/vocab.json": 1_059_962,
    "tokenizer_2/special_tokens_map.json": 2_543,
    "tokenizer_2/spiece.model": 791_656,
    "tokenizer_2/tokenizer.json": 2_424_235,
    "tokenizer_2/tokenizer_config.json": 20_817,
    "vae/config.json": 820,
    "vae/diffusion_pytorch_model.safetensors": 167_666_902,
}

# Expected FLUX.1-Fill-dev component sizes (bytes, from ModelScope listing).
# NOTE: some configs differ from dev (e.g. model_index 540 vs 536); the VAE is
# 335 MB (vs dev 168 MB) -> downloaded, not reused.
FLUX_FILL_EXPECTED: Dict[str, int] = {
    "model_index.json": 540,
    "scheduler/scheduler_config.json": 299,
    "text_encoder/config.json": 561,
    "text_encoder/model.safetensors": 246_144_352,
    "text_encoder_2/config.json": 741,
    "text_encoder_2/model-00001-of-00002.safetensors": 4_994_582_224,
    "text_encoder_2/model-00002-of-00002.safetensors": 4_530_066_360,
    "text_encoder_2/model.safetensors.index.json": 19_885,
    "tokenizer/merges.txt": 524_619,
    "tokenizer/special_tokens_map.json": 588,
    "tokenizer/tokenizer_config.json": 705,
    "tokenizer/vocab.json": 1_059_962,
    "tokenizer_2/special_tokens_map.json": 2_543,
    "tokenizer_2/spiece.model": 791_656,
    "tokenizer_2/tokenizer.json": 2_424_235,
    "tokenizer_2/tokenizer_config.json": 20_817,
    "vae/config.json": 774,
    "vae/diffusion_pytorch_model.safetensors": 335_306_212,
}

# Expected svdq-int4-fill files (bytes).
SVDQ_FILL_EXPECTED: Dict[str, int] = {
    "transformer_blocks.safetensors": 6_640_056_352,
    "unquantized_layers.safetensors": 130_218_224,
    "config.json": 393,
    "comfy_config.json": 478,
    "demo.jpg": 458_662,
}


def log(msg: str) -> None:
    print(f"[verify_models] {msg}", flush=True)


def hf_repo_dir(repo_id: str) -> Path:
    return cfg.HF_HUB_CACHE / ("models--" + repo_id.replace("/", "--"))


def find_in_snapshot(repo_id: str, name: str) -> Path | None:
    repo = hf_repo_dir(repo_id)
    snaps = repo / "snapshots"
    if not snaps.exists():
        return None
    for snap in snaps.iterdir():
        cand = snap / name
        if cand.exists():
            return cand
    # fall back to a recursive search
    for cand in repo.rglob(Path(name).name):
        if cand.is_file():
            return cand
    return None


def check_hf_lora() -> Tuple[bool, List[str]]:
    msgs, ok = [], True
    for fname in [
        "models--WorldGen-Flux-Lora/worldgen_text2scene.safetensors",
        "models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors",
    ]:
        p = find_in_snapshot("LeoXie/WorldGen", fname)
        if p and p.stat().st_size > 1_000_000:
            msgs.append(f"OK  {fname} = {p.stat().st_size:,} B")
        else:
            ok = False
            msgs.append(f"MISSING/small  {fname} -> {p}")
    return ok, msgs


def check_hf_da2() -> Tuple[bool, List[str]]:
    p = find_in_snapshot("haodongli/DA-2", "model.safetensors")
    if p and p.stat().st_size > 100_000_000:
        return True, [f"OK  da2/model.safetensors = {p.stat().st_size:,} B"]
    return False, [f"MISSING/small  da2/model.safetensors -> {p}"]


def check_hf_svdq() -> Tuple[bool, List[str]]:
    msgs, ok = [], True
    for fname in [
        "transformer_blocks.safetensors",
        "unquantized_layers.safetensors",
        "config.json",
        "comfy_config.json",
    ]:
        p = find_in_snapshot("mit-han-lab/svdq-int4-flux.1-dev", fname)
        if p and p.stat().st_size > 0:
            msgs.append(f"OK  svdq/{fname} = {p.stat().st_size:,} B")
        else:
            ok = False
            msgs.append(f"MISSING  svdq/{fname} -> {p}")
    return ok, msgs


def check_flux_components() -> Tuple[bool, List[str]]:
    msgs, ok = [], True
    total = 0
    for rel, expected in FLUX_EXPECTED.items():
        p = cfg.FLUX_DEV_DIR / rel
        if p.exists() and p.stat().st_size == expected:
            total += expected
            msgs.append(f"OK  {rel} = {p.stat().st_size:,} B")
        elif p.exists():
            ok = False
            msgs.append(f"SIZE MISMATCH  {rel}: got {p.stat().st_size:,} expected {expected:,}")
        else:
            ok = False
            msgs.append(f"MISSING  {rel}")
    msgs.append(f"FLUX components total = {total:,} B = {total / 1024 ** 3:.2f} GB")
    # T5 shard integrity
    idx = cfg.FLUX_DEV_DIR / "text_encoder_2/model.safetensors.index.json"
    if idx.exists():
        manifest = json.loads(idx.read_text(encoding="utf-8"))
        shards = sorted(set(manifest.get("weight_map", {}).values()))
        msgs.append(f"T5 index references shards: {shards}")
        for s in shards:
            if not (cfg.FLUX_DEV_DIR / "text_encoder_2" / s).exists():
                ok = False
                msgs.append(f"T5 SHARD MISSING: {s}")
    return ok, msgs


def check_flux_fill_components() -> Tuple[bool, List[str]]:
    """Verify the assembled FLUX.1-Fill-dev (i2s) components (excludes transformer/)."""
    msgs, ok = [], True
    total = 0
    for rel, expected in FLUX_FILL_EXPECTED.items():
        p = cfg.FLUX_FILL_DIR / rel
        if p.exists() and p.stat().st_size == expected:
            total += expected
            msgs.append(f"OK  {rel} = {p.stat().st_size:,} B")
        elif p.exists():
            ok = False
            msgs.append(f"SIZE MISMATCH  {rel}: got {p.stat().st_size:,} expected {expected:,}")
        else:
            ok = False
            msgs.append(f"MISSING  {rel}")
    transformer = cfg.FLUX_FILL_DIR / "transformer"
    if transformer.exists():
        ok = False
        msgs.append("UNEXPECTED transformer/ present (should be skipped)")
    idx = cfg.FLUX_FILL_DIR / "text_encoder_2/model.safetensors.index.json"
    if idx.exists():
        manifest = json.loads(idx.read_text(encoding="utf-8"))
        shards = sorted(set(manifest.get("weight_map", {}).values()))
        for s in shards:
            if not (cfg.FLUX_FILL_DIR / "text_encoder_2" / s).exists():
                ok = False
                msgs.append(f"T5 SHARD MISSING: {s}")
    msgs.append(f"Fill components total = {total:,} B = {total / 1024 ** 3:.2f} GB")
    return ok, msgs


def check_hf_svdq_fill() -> Tuple[bool, List[str]]:
    """Verify the svdq-int4-flux.1-fill-dev 4-bit transformer cache."""
    msgs, ok = [], True
    repo_dir = hf_repo_dir("mit-han-lab/svdq-int4-flux.1-fill-dev")
    for fname, expected in SVDQ_FILL_EXPECTED.items():
        p = find_in_snapshot("mit-han-lab/svdq-int4-flux.1-fill-dev", fname)
        if p and p.stat().st_size == expected:
            msgs.append(f"OK  svdq-fill/{fname} = {p.stat().st_size:,} B")
        elif p and p.stat().st_size > 0:
            ok = False
            msgs.append(f"SIZE MISMATCH  svdq-fill/{fname}: got {p.stat().st_size:,} expected {expected:,}")
        else:
            ok = False
            msgs.append(f"MISSING  svdq-fill/{fname} -> {p}")
    if repo_dir.exists():
        msgs.append(f"svdq-fill cache: {repo_dir}")
    return ok, msgs


def main() -> int:
    report: Dict[str, dict] = {}
    checks = {
        "LeoXie/WorldGen": check_hf_lora,
        "haodongli/DA-2": check_hf_da2,
        "mit-han-lab/svdq-int4-flux.1-dev": check_hf_svdq,
        "FLUX.1-dev components": check_flux_components,
    }
    # i2s (optional) checks only when the fill assets are present.
    if cfg.FLUX_FILL_DIR.exists():
        checks["FLUX.1-Fill-dev components (i2s)"] = check_flux_fill_components
    if (cfg.HF_HUB_CACHE / "models--mit-han-lab--svdq-int4-flux.1-fill-dev").exists():
        checks["mit-han-lab/svdq-int4-flux.1-fill-dev (i2s)"] = check_hf_svdq_fill
    all_ok = True
    for name, fn in checks.items():
        log(f"--- {name} ---")
        ok, msgs = fn()
        all_ok = all_ok and ok
        for m in msgs:
            log("  " + m)
        report[name] = {"ok": ok, "details": msgs}

    out = cfg.LOG_DIR / "T03-verify.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"report written: {out}")

    state = cfg.load_state()
    state.setdefault("models", {})["verify"] = {k: v["ok"] for k, v in report.items()}
    cfg.save_state(state)

    log("VERIFY MODELS: " + ("ALL OK" if all_ok else "ISSUES FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
