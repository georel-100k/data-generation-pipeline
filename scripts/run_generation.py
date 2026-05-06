#!/usr/bin/env python3
"""Reviewer-friendly launcher for the Blender synthetic-data generator."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_SCRIPT = REPO_ROOT / "pipeline" / "blender_generator.py"


def _resolve_blender(value: str | None) -> str:
    blender = value or shutil.which("blender")
    if not blender:
        raise SystemExit(
            "Could not find Blender. Install Blender 4.x or pass --blender /path/to/blender."
        )
    return blender


def build_command(args: argparse.Namespace) -> list[str]:
    script = Path(args.script).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not script.is_file():
        raise SystemExit(f"Generator script not found: {script}")
    if not csv_path.is_file():
        raise SystemExit(f"Asset manifest CSV not found: {csv_path}")

    cmd = [
        _resolve_blender(args.blender),
        "--background",
        "--python",
        str(script),
        "--",
        "--csv",
        str(csv_path),
        "--out",
        str(out_dir),
        "--num-scenes",
        str(args.num_scenes),
        "--workers",
        str(args.workers),
        "--gpu-ids",
        args.gpu_ids,
        "--seed",
        str(args.seed),
        "--resolution",
        args.resolution,
        "--samples",
        str(args.samples),
        "--aux-samples",
        str(args.aux_samples),
        "--object-pool-size",
        str(args.object_pool_size),
        "--mean-objects",
        str(args.mean_objects),
        "--max-objects",
        str(args.max_objects),
    ]

    if args.use_blenderkit and args.no_blenderkit:
        raise SystemExit("Pass at most one of --use-blenderkit or --no-blenderkit.")
    if args.use_blenderkit:
        cmd.append("--use-blenderkit")
    elif args.no_blenderkit:
        cmd.append("--no-blenderkit")
    if args.blenderkit_api_key:
        cmd.extend(["--blenderkit-api-key", args.blenderkit_api_key])

    return cmd


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Blender data-generation pipeline with explicit paths."
    )
    parser.add_argument("--csv", required=True, help="Asset manifest CSV")
    parser.add_argument("--out", default="outputs/scenes", help="Output directory")
    parser.add_argument("--blender", default=None, help="Path to Blender executable")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Blender script")
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated GPU ids. Empty/all/none leaves devices unpinned.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", default="512x512")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--aux-samples", type=int, default=1)
    parser.add_argument("--object-pool-size", type=int, default=500)
    parser.add_argument("--mean-objects", type=float, default=7.3)
    parser.add_argument("--max-objects", type=int, default=12)
    parser.add_argument("--use-blenderkit", action="store_true")
    parser.add_argument("--no-blenderkit", action="store_true")
    parser.add_argument("--blenderkit-api-key", default="")
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cmd = build_command(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
