"""T04 artifact collector / validator for WorldGen demo outputs.

Collects the generated assets (splat.ply / mesh.glb / *.png / rgb.mp4) into
D:\\mm\\outputs and asserts they are real, non-empty files (not placeholders).

Run:  D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\collect_outputs.py [--src DIR] [--dest DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

PATTERNS = ["*.ply", "*.glb", "*.png", "*.mp4", "*.safetensors"]
MIN_BYTES = {"*.ply": 1024, "*.glb": 1024, "*.png": 512, "*.mp4": 1024, "*.safetensors": 1024}


def log(msg: str) -> None:
    print(f"[collect_outputs] {msg}", flush=True)


def is_nonzero(path: Path) -> bool:
    """Heuristic: file must exceed a minimum size AND not be all zero bytes."""
    size = path.stat().st_size
    if size < MIN_BYTES.get("*" + path.suffix, 512):
        return False
    with open(path, "rb") as fh:
        chunk = fh.read(min(size, 65536))
    return any(b != 0 for b in chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(cfg.OUTPUT_DIR / "latest"))
    ap.add_argument("--dest", default=str(cfg.OUTPUT_DIR))
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log(f"source dir does not exist: {src}")
        log("COLLECT: NO OUTPUTS")
        return 1

    found: List[Path] = []
    for pat in PATTERNS:
        found.extend(sorted(src.rglob(pat)))

    report = {"src": str(src), "dest": str(dest), "artifacts": []}
    ok = False
    for f in found:
        good = f.is_file() and is_nonzero(f)
        target = dest / f.name
        if good and f.resolve() != target.resolve():
            try:
                shutil.copy2(f, target)
            except Exception as exc:  # noqa: BLE001
                log(f"copy failed {f.name}: {exc}")
        report["artifacts"].append({
            "name": f.name,
            "src": str(f),
            "bytes": f.stat().st_size if f.is_file() else 0,
            "non_zero": good,
            "dest": str(target),
        })
        if good:
            ok = True
        log(f"{'OK ' if good else 'BAD'} {f.name} ({f.stat().st_size if f.is_file() else 0:,} B)")

    out = cfg.LOG_DIR / "T04-artifacts.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"report written: {out}")

    state = cfg.load_state()
    state.setdefault("outputs", {})["last_collect"] = {"n": len(found), "ok": ok}
    cfg.save_state(state)

    log("COLLECT: " + ("VALID ARTIFACTS FOUND" if ok else "NO VALID ARTIFACTS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
