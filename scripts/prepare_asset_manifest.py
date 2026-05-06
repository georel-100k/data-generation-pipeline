#!/usr/bin/env python3
"""Build a minimal asset manifest CSV from a directory of 3D assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ASSET_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx"}
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _object_name(asset_path: Path, root: Path) -> str:
    if asset_path.parent.name == "meshes" and asset_path.parent.parent != root:
        return asset_path.parent.parent.name
    return asset_path.stem


def _texture_for(asset_path: Path) -> str:
    if asset_path.parent.name != "meshes":
        return ""
    object_dir = asset_path.parent.parent
    texture_dir = object_dir / "materials" / "textures"
    if not texture_dir.is_dir():
        return ""
    for texture in sorted(texture_dir.rglob("*")):
        if texture.suffix.lower() in TEXTURE_EXTENSIONS:
            return str(texture.resolve())
    return ""


def iter_assets(root: Path) -> list[Path]:
    assets = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS
    ]
    return sorted(assets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the CSV asset manifest consumed by pipeline/blender_generator.py."
    )
    parser.add_argument("asset_root", help="Directory to scan recursively")
    parser.add_argument("--out", default="asset_manifest.csv", help="Output CSV path")
    parser.add_argument(
        "--default-location",
        choices=["indoor", "outdoor", "both"],
        default="indoor",
        help="Value for gemma_location_type when no classifier labels are available",
    )
    parser.add_argument(
        "--class-name",
        default="",
        help="Optional class label to apply to every discovered asset",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.asset_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Asset root not found: {root}")

    assets = iter_assets(root)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "object_name",
                "filepath",
                "texture_filepath",
                "gemma_location_type",
                "class",
            ],
        )
        writer.writeheader()
        for asset in assets:
            writer.writerow(
                {
                    "filename": asset.stem,
                    "object_name": _object_name(asset, root),
                    "filepath": str(asset.resolve()),
                    "texture_filepath": _texture_for(asset),
                    "gemma_location_type": args.default_location,
                    "class": args.class_name,
                }
            )

    print(f"Wrote {len(assets)} assets to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
