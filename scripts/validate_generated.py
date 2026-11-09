"""
Validate generated scenes (output of blender_pipeline.py).

Per view checks:
  - rgb, depth, normal and mask files exist and share the metadata resolution
  - euclidean depth == zdepth * |((u-cx)/fx, (v-cy)/fy, 1)|; background >= 1e10
  - worldspace (encoded 0.5*n+0.5) and camspace normals: same background, unit length,
    camspace == R_w2c_colmap @ decode(worldspace)
  - normal background agrees with zdepth background on >= --bg-agree-min of pixels
  - masks (8-bit PNG): combined mask non-empty with ids 0..N (N objects), and each
    per-object mask equal to (combined == id) * 255; zdepth is finite
Per arrangement: every object's bounding box (metadata.json) stays inside the walls,
|x|, |y| <= --floor-half (the walls' inner faces).

Writes a JSON report ([{"view": ..., "errors": [...]}, ...]) and exits 1 if any view
has errors. Needs numpy, OpenEXR and Pillow (e.g. the metric3d venv).

Usage (from the repository root):
    python scripts/validate_generated.py                    # <repo>/__artifacts/generated_test
    python scripts/validate_generated.py --data-root ../dataset/georel-cvpr --output generated
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import OpenEXR
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(path: str) -> str:
    """Command-line paths: absolute as given, relative ones from the repository root."""
    return os.path.abspath(os.path.join(REPO_ROOT, os.path.expanduser(path)))


def read_exr(path: str, names: str) -> np.ndarray:
    """Read channels by name ("R", "RGB"); OpenEXR is used because cv2 mis-reads X/Y/Z EXRs."""
    channels = OpenEXR.File(path, separate_channels=True).channels()
    arrays = [channels[n].pixels.astype(np.float32) for n in names]
    return arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=-1)


def check_view(arr_dir: str, cam: str, meta: dict, n_obj: int, bg_agree_min: float) -> list:
    errors = []
    K = meta["intrinsics"]
    R = np.asarray(meta["extrinsics"]["R_w2c_colmap"], dtype=np.float64)
    rgb = np.asarray(Image.open(f"{arr_dir}/rgb/{cam}.png"))
    z   = read_exr(f"{arr_dir}/depth/{cam}_zdepth.exr", "R")
    e   = read_exr(f"{arr_dir}/depth/{cam}_euclidean.exr", "R")
    ws  = read_exr(f"{arr_dir}/normals/{cam}_worldspace.exr", "RGB")
    cs  = read_exr(f"{arr_dir}/normals/{cam}_camspace.exr", "RGB")
    mk  = np.asarray(Image.open(f"{arr_dir}/masks/{cam}_combined.png"))

    H, W = z.shape
    if any(a.shape[:2] != (H, W) for a in (rgb, e, ws, cs, mk)):
        errors.append("resolution mismatch between outputs")
        return errors
    if not np.isfinite(z).all():
        errors.append("non-finite zdepth")

    fg = z < 1e9
    u = (np.arange(W) + 0.5 - K["cx_px"]) / K["fx_px"]
    v = (np.arange(H) + 0.5 - K["cy_px"]) / K["fy_px"]
    ray = np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)
    if not fg.any():
        errors.append("no geometry")
    elif np.abs(e[fg] / (z[fg] * ray[fg]) - 1).max() > 1e-5:
        errors.append("euclidean != zdepth * ray length")
    if (~fg).any() and not (e[~fg] >= 1e10).all():
        errors.append("euclidean background < 1e10")

    bg_ws = np.all(ws == 0, axis=-1)
    len_cs = np.linalg.norm(cs, axis=-1)
    bg_cs = len_cs == 0
    if not (bg_cs == bg_ws).all():
        errors.append("camspace/worldspace background mismatch")
    n = 2 * ws[~bg_ws] - 1
    if n.size:
        if np.abs(np.linalg.norm(n, axis=-1) - 1).max() > 1e-3 or np.abs(len_cs[~bg_cs] - 1).max() > 1e-3:
            errors.append("non-unit normals")
        if np.abs(cs[~bg_ws] - n @ R.T).max() > 1e-5:
            errors.append("camspace != R_w2c_colmap @ worldspace")
    agree = float((bg_cs == ~fg).mean())
    if agree < bg_agree_min:
        errors.append(f"normal/zdepth background agreement {agree:.5f} < {bg_agree_min}")
    if mk.ndim != 2 or mk.dtype != np.uint8:
        errors.append(f"combined mask is not 8-bit grayscale ({mk.dtype}, shape {mk.shape})")
    if not (mk > 0).any():
        errors.append("empty object mask")
    if mk.max() > n_obj:
        errors.append(f"mask id {int(mk.max())} > number of objects {n_obj}")
    for obj_id in range(1, n_obj + 1):
        om = np.asarray(Image.open(f"{arr_dir}/masks/{cam}_obj{obj_id:03d}.png"))
        if om.shape != mk.shape or not np.array_equal(om, np.where(mk == obj_id, 255, 0)):
            errors.append(f"obj{obj_id:03d} mask != (combined == {obj_id})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "__artifacts"),
                        help="Data root (default: <repo>/__artifacts)")
    parser.add_argument("--output", default="generated_test",
                        help="Output folder in the data root to check (generated_test or generated)")
    parser.add_argument("--out-json", default=None,
                        help="Report path (default: <data root>/logs/validate_<output>.json)")
    parser.add_argument("--bg-agree-min", type=float, default=0.999)
    parser.add_argument("--floor-half", type=float, default=2.2,
                        help="FLOOR_HALF of the generator: walls' inner faces at +-this (m)")
    args = parser.parse_args()
    data_root = repo_path(args.data_root)
    args.root = os.path.join(data_root, args.output)
    args.out_json = repo_path(args.out_json) if args.out_json else \
        os.path.join(data_root, "logs", f"validate_{args.output}.json")
    if not os.path.isdir(args.root):
        sys.exit(f"not found: {args.root}")

    report = []
    for arr_dir in sorted(glob.glob(os.path.join(args.root, "scene_*", "arr_*"))):
        meta_path = os.path.join(arr_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            report.append({"view": os.path.relpath(arr_dir, args.root), "errors": ["missing metadata.json"]})
            continue
        meta = json.load(open(meta_path))
        outside = []
        for obj in meta.get("objects", []):
            lo, hi = obj["bbox_min_m"], obj["bbox_max_m"]
            excess = max(-args.floor_half - lo[0], hi[0] - args.floor_half,
                         -args.floor_half - lo[1], hi[1] - args.floor_half)
            if excess > 1e-4:
                outside.append(f"object {obj['id']} crosses a wall by {excess:.3f} m")
        if outside:
            report.append({"view": os.path.relpath(arr_dir, args.root), "errors": outside})
        cams = {c["label"]: c for c in meta["cameras"]}
        for cam in sorted(cams):
            try:
                errors = check_view(arr_dir, cam, cams[cam], len(meta.get("objects", [])),
                                    args.bg_agree_min)
            except Exception as exc:
                errors = [repr(exc)]
            report.append({"view": os.path.relpath(os.path.join(arr_dir, cam), args.root), "errors": errors})

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)
    bad = [r for r in report if r["errors"]]
    print(f"views checked: {len(report)} | views with errors: {len(bad)} -> {args.out_json}")
    for r in bad[:10]:
        print("  ", r["view"], r["errors"])
    sys.exit(1 if bad or not report else 0)


if __name__ == "__main__":
    main()
