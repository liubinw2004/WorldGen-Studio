r"""T04 argv builder for WorldGen's demo.py.

Responsibilities:
  * assemble the demo.py command line (single source of truth for run_demo.bat/.sh)
  * ALWAYS force --low_vram (12GB card -> nunchaku SVDQuant path; demo.py also
    auto-sets it, but we are explicit)
  * inject PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True into the child env
  * guarantee the output dir exists

It prints ONE line: the argument list (without the interpreter), quoted, ready
to be appended after `python demo.py`.

Usage:
    python demo_args.py --prompt "..." [--resolution 1600] [--return_mesh] ...
    # -> -p "..." --output_dir "D:\mm\outputs\run_..." --resolution 1600 --low_vram --save_scene
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)


def q(s: str) -> str:
    return '"' + str(s) + '"'


def build_argv(args: argparse.Namespace) -> list[str]:
    out_dir = Path(args.output_dir) if args.output_dir else (cfg.OUTPUT_DIR / "latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = []
    if args.prompt:
        argv += ["-p", q(args.prompt)]
    if args.image:
        argv += ["-i", q(args.image)]
    if args.pano_image:
        argv += ["--pano_image", q(args.pano_image)]
    argv += ["-o", q(str(out_dir))]
    argv += ["-r", str(args.resolution)]
    if args.use_sharp:
        argv += ["--use_sharp"]
    if args.inpaint_bg:
        argv += ["--inpaint_bg"]
    if args.return_mesh:
        argv += ["--return_mesh"]
    if args.save_scene:
        argv += ["--save_scene"]
    # low_vram is ALWAYS on for a 12GB card (kept last so it is obvious).
    argv += ["--low_vram"]
    return argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default="A beautiful landscape with mountains and a lake")
    parser.add_argument("--image", "-i", type=str, default=None)
    parser.add_argument("--pano_image", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--resolution", "-r", type=int, default=1600)
    parser.add_argument("--use_sharp", action="store_true")
    parser.add_argument("--inpaint_bg", action="store_true")
    parser.add_argument("--return_mesh", action="store_true")
    parser.add_argument("--save_scene", action="store_true", default=True)
    parser.add_argument("--no_save_scene", dest="save_scene", action="store_false")
    args = parser.parse_args()

    # Ensure the alloc-conf is set for the child process (Windows: harmless no-op warning).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(" ".join(build_argv(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
