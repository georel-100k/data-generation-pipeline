# GeoRel data generation pipeline

Blender pipeline that generates the GeoRel synthetic dataset: rooms of 3D assets (Objaverse,
Amazon Berkeley Objects, Google Scanned Objects) rendered from multiple views, with depth,
surface normals, instance masks, camera poses and a scene graph of object relations.

Each **scene** holds 1–12 objects, laid out in 3 **arrangements**, each rendered from several
**views**. Object sizes are intentionally not physical (cartoon-style rendering) but fixed
within a scene.

## Repository layout

```text
pipeline/blender_pipeline.py    the generator (runs inside Blender)
scripts/setup_assets.py         downloads the 3D assets and checks completeness
scripts/validate_generated.py   checks generated scenes
scripts/build_manifest.py       regenerates manifest/asset_manifest.csv
manifest/asset_manifest.csv     asset list (28,576 assets)
manifest/source/                inputs of build_manifest.py
docs/                           output format, asset manifest, changes from the original generator
```

Run everything from the repository root; relative paths are resolved from it.

## Requirements

- **Blender 4.5** with an NVIDIA GPU (only the driver is needed; Cycles ships its own CUDA
  kernels). Falls back to the CPU without a GPU.
- **Python 3.10+** for `scripts/`: `pip install -r requirements.txt`.
- **Headless Linux:** if Blender fails to start, check `ldd /path/to/blender | grep "not found"`.
  On minimal Ubuntu 24.04: `sudo apt-get install libxfixes3 libsm6 libice6 libgl1`.
- **Disk:** about 400 GB of assets; 10,000 generated scenes take about 2.2–2.7 TB.

### Tested environment

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-90 |
| CPU / RAM | AMD Threadripper PRO 9965WX (24 cores) / 256 GB |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM each |
| Driver / CUDA | 590.48.01 / 13.1 |
| Blender | 4.5.15 LTS (bundled Python 3.11.15) |
| Python (scripts) | 3.12.13 |

At full settings a view takes 3.8 s (1.2 s main render + 2.6 s normal render), using up to
3.7 GB RAM and 4.3 GB GPU memory per process.

## Quick start

```bash
pip install -r requirements.txt
B=/path/to/blender-4.5/blender

python scripts/setup_assets.py                                     # 1. download assets
$B --background --python pipeline/blender_pipeline.py -- --test    # 2. test run (4 scenes, 512²)
python scripts/validate_generated.py
$B --background --python pipeline/blender_pipeline.py              # 3. full run
python scripts/validate_generated.py --output generated
```

Data goes to `__artifacts/` (git-ignored) by default. To use another folder, pass the same
`--data-root PATH` to every step. Manifest paths are relative to the data root.

`setup_assets.py` downloads only missing assets, so it can be re-run; `--check` only reports.

**Reproducibility:** each scene is seeded by its index, so with the same assets and Blender
version the geometry (layouts, cameras, depth, normals, masks, metadata) is reproduced
exactly. RGB may differ by 1/255 from denoiser noise.

## Generating

```bash
$B --background --python pipeline/blender_pipeline.py -- [options]
```

| Option | Meaning |
|---|---|
| `--data-root PATH` | data root (default `__artifacts/`) |
| `--test` | test run (`TEST_*` settings) into `generated_test/` |
| `--num-scenes N` | number of scenes (overrides `M`) |
| `--start-index I` | first scene index; index `I` is `scene_{I+1}` |
| `--force` | re-render scenes that are already complete |
| `--gpus 0,1` / `cpu` | devices to use (default: all visible GPUs) |
| `--manifest PATH` | alternative asset manifest |

One scene renders at a time per GPU. **Runs are resumable:** re-running the same command
skips complete scenes and re-renders interrupted or failed ones, with identical results.
Ctrl-C (or closing the output, e.g. a `| tee` being killed) stops the run and its workers.

Other settings are constants at the top of `pipeline/blender_pipeline.py`:

| Setting | Value | Meaning |
|---|---|---|
| `M` | 10000 | scenes |
| `SEED` | 2030 | scene `idx` uses seed `SEED + idx·997` |
| `RENDER_RES_X/Y`, `RENDER_SAMPLES` | 1024², 32 | main render (denoised) |
| `NORMAL_SUPERSAMPLE` | 4 | normal rays per pixel per axis |
| `Z`, `MAX_OBJECTS_PER_SCENE` | 7.3, 12 | objects per scene (Poisson mean, cap) |
| `NUM_ARRANGEMENTS` | 3 | layouts per scene |
| `VIEWS_MIN/MEAN/MAX` | 6 / 12 / 24 | views per scene |
| `CAM_POLAR_MIN/MAX_DEG`, `CAM_FOV_DEG` | 30–60°, 70° | camera angle from zenith, field of view |
| `FLOOR_HALF`, `WALL_HEIGHT` | 2.2 m, 1.9 m | room size (4.4 × 4.4 m) |
| `BLEND_COMPRESS` | True | compress `scene.blend` |

## Output

```text
<data root>/generated/scene_00001/arr_001/
  rgb/<view>.png
  depth/<view>_{zdepth,euclidean}.exr       metres
  normals/<view>_{camspace,worldspace}.exr
  masks/<view>_combined.png                 object IDs (0 = background)
  masks/<view>_objNNN.png                   per-object binary masks
  colmap/{cameras,images,points3D}.txt
  metadata.json, scene_graph.json, scene.blend
```

Cameras follow the OpenCV/COLMAP convention (x right, y down, z forward); the world is Z-up,
in metres. Details: [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md).

`validate_generated.py` checks every view (resolutions, depth and normal consistency, masks,
objects inside the walls) and writes a report to `<data root>/logs/`.

## Asset manifest

`manifest/asset_manifest.csv` lists 28,576 assets; the 23,639 labelled `indoor` or `both`
are sampled. Regenerate it with `python scripts/build_manifest.py`.
Details: [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md).

## Changes from the original generator

Correct camera-space normals, exact Euclidean depth, supersampled normals, objects kept inside
the walls, overlap-free layouts, per-scene seeding, resumable runs, PNG masks and smaller
`.blend` files. Details: [docs/PIPELINE_CHANGES.md](docs/PIPELINE_CHANGES.md).

## Licenses

Assets: [Objaverse](https://objaverse.allenai.org/) (per-object Creative Commons licenses),
[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) and
[Google Scanned Objects](https://goo.gle/scanned-objects) (both CC BY 4.0). Each asset's
license is in the manifest's `license` column. `scene.blend` files embed the meshes, so
sharing them is subject to these licenses (about 9% of sampled assets are non-commercial).

The code is released under the Apache-2.0 license ([LICENSE](LICENSE)).
