"""T03 ModelFetcher - fetch the non-gated tier-1 models from the HF mirror.

Models (all gated=False, verified reachable via https://hf-mirror.com):
    * LeoXie/WorldGen                 (~80 MB)  2 LoRA: t2s / i2s
    * haodongli/DA-2                  (~1.28 GB) 360 depth SphereViT
    * mit-han-lab/svdq-int4-flux.1-dev (~4 GB)  nunchaku 4-bit FLUX transformer
      NOTE: this repo is NOT a single file - it ships
            transformer_blocks.safetensors + unquantized_layers.safetensors
            (+ config.json, comfy_config.json), so we snapshot the whole repo.

Tier-2 (``--tier 2`` / ``--tier all``) adds the i2s fill transformer:
    * mit-han-lab/svdq-int4-flux.1-fill-dev (~6.3 GB)  nunchaku 4-bit FLUX Fill
      transformer (transformer_blocks + unquantized_layers + configs + demo.jpg,
      the latter kept as a real i2s test input image).

Design contracts (docs/system_design.md section 8):
    * every download is preceded by a disk water-level guard (required + 20 GB)
    * 3x exponential-backoff retry
    * resume_download / snapshot_download are inherently resumable
    * actual byte sizes are measured and written back into state.json

Run with the D-drive venv interpreter:
    D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\fetch_models.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)
os.environ.setdefault("HF_ENDPOINT", cfg.HF_ENDPOINT)

# The host shell exports HTTP(S)_PROXY=http://127.0.0.1:10131; its CONNECT tunnel
# to hf-mirror.com returns 502, while direct access works. Bypass the proxy for
# our own outbound requests (curl --noproxy verified direct access is fine).
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

SAFETY_MARGIN_GB = 20.0

# repo_id -> approx size (GB) and allow_patterns (None = whole repo)
TIER1_HF: Dict[str, dict] = {
    "LeoXie/WorldGen": {"approx_gb": 0.08, "allow": None},
    "haodongli/DA-2": {"approx_gb": 1.28, "allow": ["model.safetensors", "*.json"]},
    "mit-han-lab/svdq-int4-flux.1-dev": {
        "approx_gb": 4.0,
        "allow": [
            "transformer_blocks.safetensors",
            "unquantized_layers.safetensors",
            "config.json",
            "comfy_config.json",
        ],
    },
}

# Tier-2 / i2s extras. The fill transformer is a separate 4-bit nunchaku repo
# (build_pano_fill_model -> "mit-han-lab/svdq-{precision}-flux.1-fill-dev").
TIER2_HF: Dict[str, dict] = {
    "mit-han-lab/svdq-int4-flux.1-fill-dev": {
        "approx_gb": 6.3,
        "allow": [
            "transformer_blocks.safetensors",
            "unquantized_layers.safetensors",
            "config.json",
            "comfy_config.json",
            "demo.jpg",          # kept as a real i2s test input image
        ],
    },
}

# Convenience: all HF-mirror repos keyed by tier.
HF_BY_TIER: Dict[str, Dict[str, dict]] = {"1": TIER1_HF, "2": TIER2_HF}


def log(msg: str) -> None:
    print(f"[fetch_models] {msg}", flush=True)


def free_gb(path: Path = ROOT) -> float:
    target = path if path.exists() else path.anchor
    return shutil.disk_usage(str(target)).free / (1024 ** 3)


def check_disk(required_gb: float, target: Path = ROOT) -> bool:
    """Abort if free < required + margin (system_design section 8.6)."""
    f = free_gb(target)
    need = required_gb + SAFETY_MARGIN_GB
    ok = f >= need
    log(f"disk check: free={f:.2f}GB need={need:.2f}GB (required {required_gb:.2f} + margin {SAFETY_MARGIN_GB}) -> {'OK' if ok else 'INSUFFICIENT'}")
    return ok


def dir_size_bytes(path: Path) -> int:
    """Sum unique file bytes under ``path`` (hardlinks counted once via inode)."""
    total = 0
    seen: set = set()
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    st = p.stat()
                except OSError:
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += st.st_size
    return total


def _hf_tree(repo_id: str) -> list:
    import requests

    url = f"{cfg.HF_ENDPOINT}/api/models/{repo_id}/tree/main?recursive=true"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def repair_repo_cache(repo_id: str) -> int:
    """Repair 0-byte snapshot placeholders left by Windows symlink fallback.

    On Windows, huggingface_hub's snapshot<->blob linking produced 0-byte
    snapshot entries while the real bytes correctly landed in blobs/. This
    rewrites each empty snapshot file as a hardlink (or copy) of its blob.

    The file -> blob mapping is recovered deterministically:
      * LFS files : blob name == the LFS oid reported by the HF tree API
      * others    : matched by exact size against the remaining blobs
    Returns the number of entries repaired.
    """
    repo_dir = cfg.HF_HUB_CACHE / ("models--" + repo_id.replace("/", "--"))
    snaps_root = repo_dir / "snapshots"
    blobs = repo_dir / "blobs"
    if not snaps_root.exists() or not blobs.exists():
        return 0

    entries = _hf_tree(repo_id)
    size_by_path = {e["path"]: e.get("size") for e in entries if e.get("type") == "file"}
    oid_by_path = {e["path"]: (e.get("lfs") or {}).get("oid") for e in entries if e.get("type") == "file"}
    blob_sizes = {b.name: b.stat().st_size for b in blobs.glob("*")
                  if b.is_file() and not b.name.endswith(".metadata")}

    repaired = 0
    used: set[str] = set()
    for snap in sorted(snaps_root.iterdir()):
        for f in sorted(snap.rglob("*")):
            if not f.is_file() or f.stat().st_size != 0:
                continue
            rel = f.relative_to(snap).as_posix()
            target: Optional[Path] = None
            oid = oid_by_path.get(rel)
            if oid and (blobs / oid).exists():
                target = blobs / oid
            else:
                sz = size_by_path.get(rel)
                cands = [n for n, s in blob_sizes.items() if s == sz and n not in used]
                if len(cands) == 1:
                    target = blobs / cands[0]
            if target is None:
                log(f"  WARN: cannot resolve blob for {rel}")
                continue
            f.unlink()
            try:
                os.link(target, f)
            except OSError:
                shutil.copy2(target, f)
            used.add(target.name)
            repaired += 1
            log(f"  repaired {rel} -> blob {target.name[:12]}... ({target.stat().st_size:,} B)")
    return repaired


def fetch_hf(repo_id: str, allow: Optional[List[str]], retries: int = 3) -> Optional[Path]:
    """snapshot_download a repo from the HF mirror with retry + backoff."""
    from huggingface_hub import snapshot_download

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            log(f"downloading {repo_id} (attempt {attempt}/{retries}) ...")
            t0 = time.time()
            path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow,
                cache_dir=str(cfg.HF_HUB_CACHE),
                endpoint=cfg.HF_ENDPOINT,
                max_workers=2,
            )
            log(f"  OK {repo_id} -> {path}  ({time.time() - t0:.1f}s)")
            return Path(path)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = 2 ** attempt
            log(f"  ERROR {repo_id}: {exc!r} (retry in {wait}s)")
            time.sleep(wait)
    log(f"  FATAL {repo_id} after {retries} attempts: {last_err!r}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--only", default=None, help="comma-separated repo ids to fetch")
    parser.add_argument("--tier", choices=["1", "2", "all"], default="1",
                        help="which HF-mirror tier(s) to fetch (default: 1)")
    parser.add_argument("--repair-only", action="store_true",
                        help="only repair 0-byte snapshot placeholders (no download)")
    args = parser.parse_args()

    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    log(f"HF_HUB_CACHE={cfg.HF_HUB_CACHE}  TMPDIR={os.environ.get('TMPDIR')}")

    tiers = ["1", "2"] if args.tier == "all" else [args.tier]
    repos: Dict[str, dict] = {}
    for t in tiers:
        repos.update(HF_BY_TIER[t])

    if args.repair_only:
        total = 0
        for repo_id in repos:
            n = repair_repo_cache(repo_id)
            total += n
            log(f"repair {repo_id}: {n} entries fixed")
        log(f"REPAIR DONE: {total} entries")
        return 0

    if not check_disk(sum(m["approx_gb"] for m in repos.values())):
        log("FATAL: insufficient disk space.")
        return 2

    only = set(args.only.split(",")) if args.only else None
    results: Dict[str, dict] = {}
    all_ok = True

    for repo_id, meta in repos.items():
        if only and repo_id not in only:
            continue
        repo_dir = cfg.HF_HUB_CACHE / ("models--" + repo_id.replace("/", "--"))
        # snapshot_download is idempotent: it completes/verifies an existing cache.
        path = fetch_hf(repo_id, meta["allow"])
        if path is None:
            all_ok = False
            results[repo_id] = {"ok": False}
            continue
        nrep = repair_repo_cache(repo_id)
        if nrep:
            log(f"  repaired {nrep} zero-byte snapshot entries for {repo_id}")
        size = dir_size_bytes(repo_dir)
        results[repo_id] = {
            "ok": True,
            "snapshot": str(path),
            "cache_dir": str(repo_dir),
            "bytes": size,
            "gb": round(size / 1024 ** 3, 3),
        }
        log(f"measured {repo_id}: {size:,} B = {size / 1024 ** 3:.3f} GB")

    # write back to state.json (kept split per tier for backward compatibility)
    state = cfg.load_state()
    models = state.setdefault("models", {})
    for t in tiers:
        models[f"tier{t}_hf"] = {rid: res for rid, res in results.items()
                                 if rid in HF_BY_TIER[t]}
    cfg.save_state(state)
    log(f"state.json updated: {list(results)}")

    if all_ok:
        log(f"ALL TIER{args.tier}-HF MODELS OK")
        return 0
    log(f"SOME TIER{args.tier}-HF MODELS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
