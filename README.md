# 3D Data Generation Pipeline

This repository contains the Blender-based pipeline used to generate synthetic
multi-view 3D scenes for the accompanying paper. It places classified 3D assets
in simple indoor rooms, renders RGB and geometric supervision, and writes camera
metadata, COLMAP pose files, per-object masks, and scene graphs.

The canonical generator is `pipeline/blender_generator.py`. Earlier pipeline
versions and one-off experiment scripts are retained under `legacy/` as research
provenance, but reviewers should start with the commands below.

## Quick Start

Requirements:

- Blender 4.x available as `blender` or passed with `--blender`.
- Python 3.10+ for the launcher and helper scripts.
- A CSV asset manifest with at least `filepath` and `gemma_location_type`
  columns. See `docs/DATA_MANIFEST.md`.

Install helper-script dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you have the optional sample asset archive, place it at
`examples/sample_assets.tar.gz` and build a small manifest from it:

```bash
mkdir -p data/sample_assets
tar -xzf examples/sample_assets.tar.gz -C data/sample_assets
python scripts/prepare_asset_manifest.py data/sample_assets --out data/sample_assets/assets.csv
```

Generate one reviewer smoke-test scene:

```bash
python scripts/run_generation.py \
  --csv data/sample_assets/assets.csv \
  --out outputs/reviewer_smoke \
  --num-scenes 1 \
  --workers 1 \
  --gpu-ids "" \
  --resolution 512x512 \
  --samples 16 \
  --no-blenderkit
```

The equivalent direct Blender invocation is:

```bash
blender --background --python pipeline/blender_generator.py -- \
  --csv data/sample_assets/assets.csv \
  --out outputs/reviewer_smoke \
  --num-scenes 1 \
  --workers 1 \
  --gpu-ids "" \
  --resolution 512x512 \
  --samples 16 \
  --no-blenderkit
```

Use `--gpu-ids 0,1` on a multi-GPU machine to pin workers to physical GPUs.
Use `--workers N` to control the number of concurrent Blender worker processes.

## Outputs

Each generated scene is written under:

```text
outputs/reviewer_smoke/
  scene_001/
    arr_001/
      rgb/*.png
      depth/*_zdepth.exr
      depth/*_euclidean.exr
      normals/*_camspace.exr
      normals/*_worldspace.exr
      masks/*_combined.exr
      masks/*_obj*.exr
      colmap/cameras.txt
      colmap/images.txt
      colmap/points3D.txt
      metadata.json
      scene_graph.json
      scene.blend
```

`metadata.json` stores camera intrinsics/extrinsics and object bounding boxes.
`scene_graph.json` stores object nodes, distances, and spatial relations.
COLMAP files contain camera poses only; `points3D.txt` is intentionally empty.

## Reproducibility Notes

- Set `--seed` to reproduce sampling and placement.
- Blender and GPU drivers can introduce small numerical/rendering differences
  across platforms.
- BlenderKit material downloads are off by default; pass `--use-blenderkit`
  only when you want to use the BlenderKit cache or API.
- The original large-scale run used higher resolution, more samples, and many
  worker processes. The quick-start command is intentionally small.

## Repository Layout

```text
.
  pipeline/                  Canonical Blender generator
  scripts/                   Reviewer-facing launchers and utility scripts
  scripts/download/          Dataset download helpers
  scripts/preprocess/        Dataset conversion and manifest-prep helpers
  docs/                      Manifest and output documentation
  examples/                  Tiny example manifest and optional sample archive
  notebooks/                 Classifier notebook and related artifacts
  legacy/                    Earlier experimental scripts kept for provenance
```

## Helper Scripts

- `scripts/prepare_asset_manifest.py`: scans a directory of `.glb`, `.gltf`,
  `.obj`, or `.fbx` assets and writes a minimal manifest CSV.
- `scripts/run_generation.py`: normal Python launcher for the Blender generator.
- `scripts/rerender_camspace_normals.py`: optional post-process launcher that
  re-renders camera-space normal EXRs from saved `scene.blend` files.
- `scripts/run_parallel_recons_norma.py`: original launcher name retained for
  backwards compatibility.
- `scripts/fix_camspace_normals.py`: Blender worker used by the normal
  re-renderer.
- `scripts/download/` and `scripts/preprocess/`: dataset acquisition and
  preprocessing utilities used during the original data build.

## Data Sources

The pipeline can consume local assets from sources such as Objaverse, Amazon
Berkeley Objects, Google Scanned Objects, ShapeNet, and CO3D after they have
been downloaded and represented in the manifest CSV. Reviewers do not need to
download all original sources for a smoke test.

Please follow the license and terms of each upstream dataset when redistributing
assets or generated derivatives.

## License

This code is released under the Apache-2.0 license. See `LICENSE`.
