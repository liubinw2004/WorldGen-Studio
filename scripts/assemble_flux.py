"""T03/T06 FLUX component assembly (gated models via ModelScope).

Two variants are supported through ``--variant`` (default ``dev`` -> fully
backward compatible with the original T03 behaviour):

  * ``dev``  -> FLUX.1-dev  (t2s):        ``AI-ModelScope/FLUX.1-dev``
  * ``fill`` -> FLUX.1-Fill-dev (i2s):    ``AI-ModelScope/FLUX.1-Fill-dev``

Both repos are gated on the Hub (hf-mirror returns 403), so we assemble ONLY
the diffusers components the pipeline needs locally, skipping the native
bf16 transformer (supplied at runtime by the nunchaku 4-bit nunchaku model).

``dev`` downloads (~9.94 GB):
    model_index.json
    scheduler/scheduler_config.json
    text_encoder/{config.json, model.safetensors}                 # CLIP-L  ~246 MB
    text_encoder_2/{config.json, model-0000{1,2}-of-00002.safetensors,
                    model.safetensors.index.json}                 # T5-XXL  ~9.52 GB
    tokenizer/{vocab.json, merges.txt, special_tokens_map.json, tokenizer_config.json}
    tokenizer_2/{spiece.model, tokenizer.json, special_tokens_map.json, tokenizer_config.json}
    vae/{config.json, diffusion_pytorch_model.safetensors}        # ~168 MB

``fill`` downloads only ~0.6 GB of NEW bytes: the T5-XXL encoder (two shards,
~9.5 GB) and CLIP-L (~246 MB) are byte-for-byte identical to the ``dev`` copy,
so after a full sha256 verification against the ModelScope-declared digest we
HARDLINK them from ``models/FLUX.1-dev`` instead of re-downloading. Everything
else (scheduler, model_index, tokenizers, the 335 MB Fill VAE) is downloaded.

Explicitly SKIPPED (would waste 50+ GB):
    transformer/**                      (supplied by nunchaku at runtime)
    flux1-dev.safetensors  (23.8 GB) / flux1-fill-dev.safetensors (23.8 GB)
    ae.safetensors         (335 MB)     (duplicate of vae/)
    README.md / LICENSE.md / configuration.json / .gitattributes / images/
    dev_grid.jpg (dev only)

Plan A (default): after assembling into ``models/<variant>`` we ALSO build a
HF-hub cache mirror at ``models--black-forest-labs--FLUX.1-<variant>`` so that
the hard-coded ``from_pretrained("black-forest-labs/FLUX.1-<variant>")`` in
pano_gen.py resolves locally with HF_HUB_OFFLINE=1 (zero source changes).

Run with the venv interpreter:
    D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\assemble_flux.py [--variant dev|fill]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)

MS_API_FILES = ("https://www.modelscope.cn/api/v1/models/{ns}/repo/files"
                "?Revision=master&Recursive=true")
MS_FILE_URL = "https://www.modelscope.cn/api/v1/models/{ns}/repo?Revision=master&FilePath={path}"
REVISION = "master"
SAFETY_MARGIN_GB = 20.0

# --------------------------------------------------------------------------- #
# Variant definitions. Each entry is the single source of truth for one target.
# --------------------------------------------------------------------------- #
VARIANTS: Dict[str, dict] = {
    "dev": {
        "title": "FLUX.1-dev",
        "ms_repo": "AI-ModelScope/FLUX.1-dev",
        "dest": cfg.FLUX_DEV_DIR,
        "mirror_repo_id": "black-forest-labs/FLUX.1-dev",
        "ignore_prefixes": ("transformer/",),
        "ignore_exact": {
            "flux1-dev.safetensors", "ae.safetensors", "dev_grid.jpg",
            "README.md", "LICENSE.md", "configuration.json", ".gitattributes",
        },
        # No cross-variant reuse for the primary variant.
        "reuse_from": {},
    },
    "fill": {
        "title": "FLUX.1-Fill-dev",
        "ms_repo": cfg.FLUX_FILL_MS_REPO,
        "dest": cfg.FLUX_FILL_DIR,
        "mirror_repo_id": "black-forest-labs/FLUX.1-Fill-dev",
        "ignore_prefixes": ("transformer/", "images/"),
        "ignore_exact": {
            "flux1-fill-dev.safetensors", "ae.safetensors",
            "README.md", "LICENSE.md", "configuration.json", ".gitattributes",
        },
        # rel_path -> (source_root, source_rel); hardlinked ONLY if the sha256 of
        # the source equals the ModelScope-declared sha256 of the target file.
        "reuse_from": {
            "text_encoder/model.safetensors": (
                cfg.FLUX_DEV_DIR, "text_encoder/model.safetensors"),
            "text_encoder_2/model-00001-of-00002.safetensors": (
                cfg.FLUX_DEV_DIR, "text_encoder_2/model-00001-of-00002.safetensors"),
            "text_encoder_2/model-00002-of-00002.safetensors": (
                cfg.FLUX_DEV_DIR, "text_encoder_2/model-00002-of-00002.safetensors"),
        },
    },
}


def log(msg: str) -> None:
    print(f"[assemble_flux] {msg}", flush=True)


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(str(path if path.exists() else path.anchor)).free / (1024 ** 3)


def check_disk(required_gb: float) -> bool:
    f = free_gb()
    need = required_gb + SAFETY_MARGIN_GB
    ok = f >= need
    log(f"disk check: free={f:.2f}GB need={need:.2f}GB -> {'OK' if ok else 'INSUFFICIENT'}")
    return ok


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """Full streaming sha256 of a file (never loads it into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_ms_files(ns_name: str) -> List[dict]:
    import requests

    r = requests.get(MS_API_FILES.format(ns=ns_name), timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("Data", {}).get("Files", [])
    out = []
    for f in files:
        p = f.get("Path", "")
        if f.get("Type") == "tree":  # directories have size 0 / Type tree
            continue
        if not p or p.endswith("/"):
            continue
        out.append({
            "path": p,
            "size": int(f.get("Size", 0) or 0),
            "sha256": (f.get("Sha256") or "").lower() or None,
        })
    return out


def select_components(all_files: List[dict], variant: dict) -> List[dict]:
    ignore_prefixes = variant["ignore_prefixes"]
    ignore_exact = variant["ignore_exact"]
    selected = []
    for f in all_files:
        p = f["path"]
        if p in ignore_exact:
            continue
        if any(p.startswith(pref) for pref in ignore_prefixes):
            continue
        if f["size"] == 0:  # directory placeholder
            continue
        selected.append(f)
    return selected


def download_file(rel_path: str, expected_size: int, dest_root: Path,
                  ns_name: str, retries: int = 3) -> bool:
    """Download one file with HTTP Range resume + exponential-backoff retry."""
    import requests

    dest = dest_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = MS_FILE_URL.format(ns=ns_name, path=urllib.parse.quote(rel_path))

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_size and have == expected_size:
            return True
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
                if have and r.status_code == 200:
                    have = 0  # server ignored Range -> restart
                r.raise_for_status()
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            final = dest.stat().st_size
            if expected_size and final != expected_size:
                log(f"  size mismatch {rel_path}: got {final:,} expected {expected_size:,}")
                raise IOError("size mismatch")
            return True
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            log(f"  ERROR {rel_path}: {exc!r} (retry {attempt}/{retries} in {wait}s)")
            time.sleep(wait)
    return False


def try_reuse(rel_path: str, expected_size: int, expected_sha: Optional[str],
              src_root: Path, src_rel: str, dest: Path) -> Tuple[bool, str]:
    """Hardlink ``src_root/src_rel`` -> ``dest`` iff it fully matches the target.

    Verification: the file size AND its streaming sha256 must equal the
    ModelScope-declared values. Any mismatch (or a missing digest) aborts the
    reuse so the caller falls back to a real download.
    """
    src = src_root / src_rel
    if not src.exists():
        return False, f"reuse source missing: {src}"
    if src.stat().st_size != expected_size:
        return False, f"reuse size mismatch ({src.stat().st_size:,} != {expected_size:,})"
    if not expected_sha:
        return False, "no ModelScope sha256 to verify against -> download"
    got = sha256_file(src)
    if got != expected_sha:
        return False, f"reuse sha256 mismatch ({got[:12]} != {expected_sha[:12]}) -> download"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)                      # same volume -> no extra bytes
        how = "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        how = "copy"
    return True, f"reused via {how} from {src} (sha256 verified)"


def build_hf_cache_mirror(src_root: Path, repo_id: str) -> Path:
    """Mirror a local diffusers dir into HF hub cache layout (blobs+snapshots+refs).

    This lets ``from_pretrained("<repo_id>")`` succeed with HF_HUB_OFFLINE=1 and
    no source patch (Plan A).
    """
    cache_repo = cfg.HF_HUB_CACHE / ("models--" + repo_id.replace("/", "--"))
    sha = hashlib.sha1(repo_id.encode()).hexdigest()  # 40-hex pseudo commit
    snap = cache_repo / "snapshots" / sha
    blobs = cache_repo / "blobs"
    refs = cache_repo / "refs"
    for d in (snap, blobs, refs):
        d.mkdir(parents=True, exist_ok=True)

    linked = copied = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        etag = hashlib.sha256(str(rel).encode()).hexdigest()
        blob = blobs / etag
        if not blob.exists():
            try:
                os.link(src, blob)          # hardlink (same volume) = no extra bytes
                linked += 1
            except OSError:
                shutil.copy2(src, blob)
                copied += 1
        target = snap / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(blob, target)
            except OSError:
                shutil.copy2(blob, target)

    # NOTE: no trailing newline (and no CRLF translation) - huggingface_hub does
    # not strip refs/main, so a stray "\r\n" would break the snapshot dir lookup.
    (refs / "main").write_text(sha, encoding="utf-8", newline="")
    log(f"HF cache mirror: {cache_repo} (sha={sha[:12]}..., hardlinked={linked}, copied={copied})")
    return cache_repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="dev",
                        help="which FLUX variant to assemble (default: dev)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--skip-mirror", action="store_true", help="do not build HF cache mirror")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    ns_name = variant["ms_repo"]
    dest_root: Path = variant["dest"]
    reuse_from: Dict[str, Tuple[Path, str]] = variant["reuse_from"]

    log(f"variant={args.variant} ({variant['title']})  ModelScope={ns_name}  "
        f"dest={dest_root}  TMPDIR={os.environ.get('TMPDIR')}")

    all_files = list_ms_files(ns_name)
    selected = select_components(all_files, variant)
    total = sum(f["size"] for f in selected)
    log(f"repo has {len(all_files)} files; selected {len(selected)} components "
        f"= {total:,} B = {total / 1024 ** 3:.2f} GB")
    for f in selected:
        tag = " [reuse-candidate]" if f["path"] in reuse_from else ""
        log(f"  {f['size']:>13,}  {f['path']}{tag}")

    # The reuse candidates do NOT consume download bytes; subtract them from the
    # disk water-level requirement (they are hardlinks to existing dev files).
    reuse_bytes = sum(f["size"] for f in selected if f["path"] in reuse_from)
    download_total = total - reuse_bytes
    log(f"of which {reuse_bytes / 1024 ** 3:.2f} GB is hardlink-reuse; "
        f"new download = {download_total / 1024 ** 3:.2f} GB")

    if not check_disk(download_total / 1024 ** 3):
        log("FATAL: insufficient disk space.")
        return 2

    if args.force and dest_root.exists():
        shutil.rmtree(dest_root)

    ok = True
    downloaded_bytes = 0
    reused_bytes = 0
    for f in selected:
        rel, size, sha = f["path"], f["size"], f.get("sha256")
        dest = dest_root / rel
        already = dest.exists() and dest.stat().st_size == size

        if already and not args.force:
            log(f"  [OK] {rel}  (cached, {size:,} B)")
            downloaded_bytes += size
            continue

        if rel in reuse_from and not args.force:
            src_root, src_rel = reuse_from[rel]
            good, why = try_reuse(rel, size, sha, src_root, src_rel, dest)
            if good:
                reused_bytes += size
                log(f"  [OK] {rel}  ({why})")
                continue
            log(f"  [--] {rel}  reuse skipped: {why}")

        t0 = time.time()
        good = download_file(rel, size, dest_root, ns_name)
        if good:
            downloaded_bytes += size
            log(f"  [{'OK' if good else 'XX'}] {rel}  ({time.time() - t0:.1f}s)")
        else:
            ok = False
            log(f"  [XX] {rel}  (FAILED)")

    state = cfg.load_state()
    state.setdefault("models", {})[f"flux_{args.variant}_components"] = {
        "source": f"modelscope:{ns_name}",
        "ok": ok,
        "dest": str(dest_root),
        "bytes": sum((dest_root / f["path"]).stat().st_size
                     for f in selected if (dest_root / f["path"]).exists()),
        "n_files": len(selected),
        "reused_bytes": reused_bytes,
        "downloaded_bytes": downloaded_bytes,
    }
    cfg.save_state(state)

    if ok and not args.skip_mirror:
        build_hf_cache_mirror(dest_root, variant["mirror_repo_id"])
        log("Plan A HF cache mirror ready (set HF_HUB_OFFLINE=1 to use it).")

    log(f"ASSEMBLE {variant['title']}: " + ("OK" if ok else "FAILED")
        + f"  (new download {downloaded_bytes / 1024 ** 3:.2f} GB, "
          f"reused {reused_bytes / 1024 ** 3:.2f} GB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
