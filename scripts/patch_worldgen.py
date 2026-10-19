"""Plan B minimal patch - redirect the hard-coded FLUX repo ids to local dirs.

Use ONLY if Plan A (HF cache mirror + HF_HUB_OFFLINE=1) fails. This changes the
*source* of the repo id string, never the algorithm/semantics.

Patches src/worldgen/pano_gen.py, replacing every occurrence of
    "black-forest-labs/FLUX.1-dev"      -> os.environ.get("WORLDGEN_FLUX_DEV_DIR", "<orig>")
    "black-forest-labs/FLUX.1-Fill-dev" -> os.environ.get("WORLDGEN_FLUX_FILL_DIR", "<orig>")

Idempotent; a one-time backup is saved as pano_gen.py.orig.

Run:  D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\patch_worldgen.py [--revert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

TARGET = cfg.REPO_DIR / "src" / "worldgen" / "pano_gen.py"
BACKUP = TARGET.with_suffix(".py.orig")

REPLACEMENTS = {
    '"black-forest-labs/FLUX.1-dev"':
        'os.environ.get("WORLDGEN_FLUX_DEV_DIR", "black-forest-labs/FLUX.1-dev")',
    '"black-forest-labs/FLUX.1-Fill-dev"':
        'os.environ.get("WORLDGEN_FLUX_FILL_DIR", "black-forest-labs/FLUX.1-Fill-dev")',
}


def log(msg: str) -> None:
    print(f"[patch_worldgen] {msg}", flush=True)


def apply() -> int:
    if not TARGET.exists():
        log(f"FATAL: {TARGET} not found")
        return 1
    src = TARGET.read_text(encoding="utf-8")

    if "os.environ.get(\"WORLDGEN_FLUX_DEV_DIR\"" in src:
        log("already patched - nothing to do.")
        return 0

    if not BACKUP.exists():
        BACKUP.write_text(src, encoding="utf-8")
        log(f"backup written: {BACKUP}")

    patched = src
    counts = {}
    for old, new in REPLACEMENTS.items():
        counts[old] = patched.count(old)
        patched = patched.replace(old, new)

    if patched == src:
        log("WARNING: no occurrences replaced (unexpected).")
        return 1

    if not patched.lstrip().startswith("import os"):
        patched = "import os\n" + patched

    TARGET.write_text(patched, encoding="utf-8")
    for old, n in counts.items():
        log(f"replaced {n}x  {old}")
    log("patch applied. NOTE: record this in docs/DEVIATIONS.md")
    return 0


def revert() -> int:
    if not BACKUP.exists():
        log("no backup to revert from.")
        return 1
    TARGET.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"reverted {TARGET} from {BACKUP}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    return revert() if args.revert else apply()


if __name__ == "__main__":
    sys.exit(main())
