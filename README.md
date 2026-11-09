# GeoRel data generation pipeline

Blender pipeline that generates the GeoRel synthetic dataset: rooms filled with 3D assets
(Objaverse, Amazon Berkeley Objects, Google Scanned Objects), rendered from several camera
views per layout, with pixel-aligned depth, surface normals, instance masks, camera poses,
and a scene graph of spatial relations between the objects.

Each **scene** is a set of 1–12 objects. Each scene is laid out three times
(**arrangements**), and each arrangement is rendered from several **views**. Object sizes
are intentionally not physical (every object is scaled to a similar size, for cartoon-style
scene rendering) but stay fixed within a scene.

## Repository layout

```text
pipeline/
  blender_pipeline.py          the generator (runs inside Blender)
scripts/
  setup_assets.py              downloads the manifest's 3D assets and checks they are complete
  validate_generated.py        per-view checks of generated scenes
  build_manifest.py            regenerates manifest/asset_manifest.csv
manifest/
  asset_manifest.csv           the generator's asset list (28,576 assets)
  source/                      inputs of build_manifest.py: recovered source manifest and
                               per-object Objaverse licenses
docs/
  OUTPUT_FORMAT.md             every output file and its conventions
  DATA_MANIFEST.md             the asset manifest: columns, contents, how it was built
  PIPELINE_CHANGES.md          changes from the original generator, with verification
```

All commands are run from the repository root, and relative paths given to any script
are resolved from the repository root.

## Requirements

- **Blender 4.5** with Cycles and an NVIDIA GPU (CUDA). Blender ships its own Cycles CUDA
  kernels, so only the NVIDIA driver is needed, not the CUDA toolkit. The generator runs
  inside Blender's bundled Python (which provides `bpy`, `mathutils` and `numpy`). Without a
  GPU it falls back to the CPU (much slower).
- **Python 3.10+** for `scripts/`: `pip install -r requirements.txt`.
- **Headless Linux**: Blender loads a few X11/OpenGL libraries even with `--background`.
  If it fails to start, list what is missing with `ldd /path/to/blender | grep "not found"`;
  on a minimal Ubuntu 24.04 these were `libxfixes3 libsm6 libice6 libgl1`
  (`sudo apt-get install libxfixes3 libsm6 libice6 libgl1`).
- **Disk**: ~400 GB of assets, plus ~1.75 GB per generated scene at 1024×1024.

### Tested environment

| Component | Version / spec |
|---|---|
| OS | Ubuntu 24.04.5 LTS (Linux kernel 6.8.0-90) |
| CPU | AMD Ryzen Threadripper PRO 9965WX, 24 cores / 48 threads |
| RAM | 256 GB (251 GiB usable) |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM each (97,887 MiB; compute capability 12.0) |
| NVIDIA driver / CUDA | driver 590.48.01, CUDA 13.1 (driver API; Cycles uses Blender's bundled kernels) |
| Blender | 4.5.15 LTS (official Linux x64 build), bundled Python 3.11.15 |
| Python (scripts) | 3.12.13 |
| Storage | ext4 RAID volume (data root, assets and outputs on the same filesystem) |

Resource use at the full settings (1024×1024, 32 samples, 4× normal supersampling): about
4.3 s per view (2.0 s main render + 2.3 s normal render), peak 3.7 GB RAM per Blender
process and 4.3 GB GPU memory, one scene at a time per GPU.

## Data root

Code and the asset manifest live in this repository. Assets and generated data live in a
**data root**: `__artifacts/` in the repository by default (ignored by git), or any folder
passed with `--data-root PATH` to `setup_assets.py`, `blender_pipeline.py` and
`validate_generated.py`.

```text
<data root>/
  assets/            objaverse/glbs/, abo/3dmodels/original/, gscanned/  (setup_assets.py)
  object-paths.json  Objaverse download index (fetched by setup_assets.py)
  generated/         output of full runs
  generated_test/    output of --test runs
  logs/              validation reports, download failure lists
  cache/             metadata cache (only if licenses are re-fetched)
```

Paths in `manifest/asset_manifest.csv` are relative to the data root
(`assets/objaverse/glbs/<uid>.glb`, …), so the same manifest works on any machine.

## Reproducing the dataset

```bash
git clone <repo-url> data-generation-pipeline && cd data-generation-pipeline
pip install -r requirements.txt
B=/path/to/blender-4.5/blender

# 1. download the 3D assets (~400 GB) into __artifacts/assets/ and check completeness
python scripts/setup_assets.py

# 2. small test run: 4 scenes, 512x512, 16 samples -> __artifacts/generated_test/
$B --background --python pipeline/blender_pipeline.py -- --test
python scripts/validate_generated.py

# 3. full run -> __artifacts/generated/
$B --background --python pipeline/blender_pipeline.py
python scripts/validate_generated.py --output generated
```

To keep the data elsewhere, pass the same `--data-root` to every step, e.g.
`--data-root ../dataset/georel-cvpr`.

`setup_assets.py` fetches Objaverse GLBs from the `allenai/objaverse` Hugging Face dataset,
streams the official ABO 3D-model archive (~166 GB) extracting only the models in the
manifest, and downloads Google Scanned Objects from Gazebo Fuel. It only fetches what is
missing, so it can be re-run to resume, and it exits with an error if any asset of the
generator's sampling pool is missing (`--check` only reports).

**Same seed, same dataset.** A scene's content depends only on `SEED`, its index, the
manifest and the set of pool assets present, not on how the run is split or resumed. With
all assets present and the same Blender version, the geometry (layouts, cameras, depth,
normals, masks, metadata) is reproduced exactly; RGB can differ by up to 1/255 from
denoiser noise.

## Generating

```bash
$B --background --python pipeline/blender_pipeline.py -- [options]
```

| Option | Meaning |
|---|---|
| `--data-root PATH` | data root (default `__artifacts/`) |
| `--test` | small test run (`TEST_*` settings) into `generated_test/` |
| `--num-scenes N` | render N scenes (overrides `M`) |
| `--gpus 0,1` / `--gpus cpu` | GPUs to use (default: all visible GPUs, from `CUDA_VISIBLE_DEVICES` or `nvidia-smi`) |
| `--manifest PATH` | another asset manifest with the same columns (see `docs/DATA_MANIFEST.md`) |

A coordinator process starts one Blender worker per scene, one scene at a time per GPU.
Per camera there are two renders: the main Cycles render (RGB, depth, masks) and a
supersampled normal render. Objects are placed and, only in the first arrangement, shrunk
if they do not fit; every arrangement is overlap-free and inside the walls; later
arrangements move and rotate objects without resizing them.

Other settings are constants at the top of `pipeline/blender_pipeline.py`:

| Setting | Value | Meaning |
|---|---|---|
| `M` | 10000 | scenes to render |
| `SEED` | 2030 | scene `idx` is seeded with `SEED + idx·997` |
| `SCENE_INDEX_OFFSET` | 0 | first scene index (to resume: first missing index, with `M` = scenes left) |
| `RENDER_RES_X/Y`, `RENDER_SAMPLES` | 1024², 32 | main render (OIDN denoised) |
| `NORMAL_SUPERSAMPLE` | 4 | normal pass rays per pixel per axis |
| `Z`, `MAX_OBJECTS_PER_SCENE` | 7.3, 12 | objects per scene (Poisson mean, cap) |
| `NUM_ARRANGEMENTS` | 3 | layouts per scene |
| `VIEWS_MIN/MEAN/MAX` | 6 / 12 / 24 | views per scene, split across arrangements |
| `CAM_POLAR_MIN/MAX_DEG`, `CAM_FOV_DEG` | 30–60°, 70° | camera elevation band (from zenith), field of view |
| `TARGET_SIZE_MIN/MAX` | 1.8–3.5 m | initial largest side of each object |
| `FLOOR_HALF`, `WALL_HEIGHT` | 2.2 m, 1.9 m | room: 4.4 × 4.4 m floor |
| `MIN_GAP`, `WALL_CLEARANCE` | 0.15 m, 0.05 m | spacing between objects / from walls |
| `TEST_*` | 4 scenes, 512², 16 samples | used by `--test` |

## Validating

`scripts/validate_generated.py [--data-root PATH] [--output generated_test|generated]`
checks every view and arrangement: files present at the right resolution, Euclidean depth
consistent with z-depth, unit normals with matching backgrounds, camera-space normals equal
to the rotated world-space normals, non-empty masks, and every object inside the walls. It
writes `<data root>/logs/validate_<output>.json` and exits with an error if anything fails.

## Asset manifest

`manifest/asset_manifest.csv` lists 28,576 assets (Objaverse 19,586, ABO 7,960, GSO 1,030),
of which the 23,639 labelled `indoor` or `both` form the generator's sampling pool. It is
generated in one step from the recovered source manifest in `manifest/source/`:

```bash
python scripts/build_manifest.py
```

Columns, licenses and the full build history: [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md).

## Output

```text
<data root>/generated/scene_001/arr_001/
  rgb/<view>.png                       colour render
  depth/<view>_{zdepth,euclidean}.exr  planar depth / ray distance, metres
  normals/<view>_{camspace,worldspace}.exr
  masks/<view>_combined.exr, masks/<view>_objNNN.exr
  colmap/{cameras,images,points3D}.txt
  metadata.json, scene_graph.json, scene.blend
```

Cameras use the OpenCV/COLMAP convention (x right, y down, z forward); the world is
Blender's (Z up, metres). File-by-file details: [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md).

## Changes from the original generator

Correct camera-space normals, exact Euclidean depth, supersampled normals, objects kept
inside the walls, fixed object sizes and overlap-free layouts in every arrangement,
per-scene seeding, portable paths, and fixes for long runs. Details and verification:
[docs/PIPELINE_CHANGES.md](docs/PIPELINE_CHANGES.md).

## Data sources and licenses

Assets come from [Objaverse](https://objaverse.allenai.org/) (per-object Creative Commons
licenses: `by`, `by-sa`, `by-nc`, `by-nc-sa`, `cc0`), the
[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
dataset (CC BY 4.0) and [Google Scanned Objects](https://goo.gle/scanned-objects) (CC BY 4.0).
Each asset's license is in the manifest's `license` column. `scene.blend` files embed the
source meshes, so redistributing them is subject to these licenses (about 9% of the sampling
pool is non-commercial).

## License

The code is released under the Apache-2.0 license; see [LICENSE](LICENSE).
