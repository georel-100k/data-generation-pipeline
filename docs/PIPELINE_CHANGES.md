# Changes from the original generator

`pipeline/blender_pipeline.py` started from the last version of the original generator
(`final_blender_pipeline6.py`, 2026-05-08). This file lists what changed and how each change
was verified. Test numbers refer to `--test` runs (512×512, 16 samples) on 6 scenes.

## Outputs

1. **Camera-space normals are in camera space.** The Cycles `Normal` pass is in world space
   and v6 wrote it unchanged as `*_camspace.exr`. Normals are now rotated into the
   OpenCV/COLMAP camera frame with the same `R_w2c_colmap` stored in `metadata.json`.
   *Verified:* `camspace = R_w2c_colmap · world` to 1e-7; floor normal matches the camera
   pose; 2.1° median difference from Metric3Dv2 predictions.
2. **Euclidean depth is exact.** v6 rendered it with an emission material: the background
   held the world colour (0.0509, a fake surface 5 cm away) and silhouette pixels averaged
   two samples. It is now `zdepth × ray length`, computed in the main render (same
   intrinsics as `metadata.json` / COLMAP via `_camera_intrinsics`), with zdepth's no-hit
   value (≥ 1e10). *Verified:* matches to 1e-7 on every pixel.
3. **Supersampled normal pass.** Both normal maps come from one render with the world
   detached (background 0) at `NORMAL_SUPERSAMPLE` (4×) resolution, one ray per sub-pixel;
   `_supersampled_normals` averages each 4×4 block over the rays of its majority object
   and renormalises. Object boundaries stay crisp and every foreground normal is unit
   length. *Why:* with one ray per pixel, dense meshes (mostly 3D scans, ~20% of objects)
   gave per-pixel random normals. *Verified:* noisy objects' median neighbour-pixel angle
   21.1° → 13.6° (the rest is real surface roughness); normal and zdepth backgrounds agree
   on ≥ 99.955% of pixels.
4. **Aux render always uses Cycles.** v6 switched to EEVEE whenever a libEGL was found; on a
   headless machine without a GL driver EEVEE produced garbage normals.

## Layout

5. **Objects stay inside the walls.** In v6, 4% of placements crossed a wall (one object
   ended up outside the room): the "no free spot" fallback ignored object size, and shrinking
   an object scaled it about its file origin, sliding the mesh. Now shrinking keeps the
   footprint centre fixed, the fallback respects the footprint (`fallback_position`), and
   `contain_and_resolve` pulls any object back inside (`WALL_CLEARANCE` = 5 cm).
6. **Object sizes are fixed per scene.** v6 re-ran overlap resolution in every arrangement,
   shrinking objects permanently each time (down to 38% of their first size) and adding a
   new random tilt each time. Now only arrangement 1 may shrink objects. Its layout is
   stored (`snapshot_layout`) and arrangements 2+ (`rearrange_fixed_size`) only move and
   rotate objects about the vertical axis. If no layout is found, arrangement 1's layout is
   rotated by 90°, 180° or 270° about the room centre (always fits).
7. **Every arrangement is overlap-free.** v6's overlap resolution gives up after 8 rounds
   and could leave objects intersecting. `settle_first_layout` keeps an already-clean first
   layout; otherwise it searches for an overlap-free layout at the current sizes
   (`_search_layout`, shared with later arrangements), shrinking the largest object only if
   none is found. *Verified (5–7):* across 18 arrangements, 0 of 510 object pairs closer than
   `MIN_GAP`, no object within 5 cm of a wall, identical mesh dimensions and heights in all
   arrangements of a scene.

## Determinism and running

8. **Per-scene seeding.** Every scene re-seeds with `SEED + idx·997` before it is built, and
   the resume offset (`SCENE_INDEX_OFFSET`) is applied once when listing scenes. v6 added
   the offset after seeding (resumed scenes got mismatched seeds) and inline mode used one
   random stream for all scenes. *Verified:* a scene rendered inline, alone with an offset,
   or by the 2-GPU coordinator gives byte-identical JSON/COLMAP files and bit-identical
   depth, normal and mask EXRs (RGB differs by ≤ 1/255 from denoiser noise).
9. **Portable paths and command-line options.** v6 hard-coded the manifest and output
   paths, and the manifest held absolute paths. The manifest now lives in the repository
   (`manifest/`) with paths relative to a data root (default `__artifacts/`, or
   `--data-root PATH`; relative command-line paths resolve from the repository root), so a
   clone can download the assets (`scripts/setup_assets.py`) and generate. Options
   `--test`, `--num-scenes N`, `--data-root`, `--manifest` and `--gpus` (default: all
   visible GPUs instead of v6's hard-coded `[0, 1]`; `cpu` to render on the CPU) are
   forwarded to worker processes; the coordinator is always used, so `--gpus` also applies
   to single-scene runs. `scene_graph.json` records the manifest's relative asset paths.
   *Verified:* run from the repository, scenes are identical to the previous outputs
   (apart from relative paths in the scene graph); a `--gpus cpu` run gives identical
   metadata.
10. **Robustness for long runs.** The coordinator uses one thread per GPU slot (v6 created
    one per scene); the per-process listing of the whole object pool was removed (~89% of
    log lines); the normal pass cleans up its temporary files in a `finally`; `[PASS1]` /
    `[PASS2]` log lines report seconds per view; unused functions with the old buggy logic
    were removed.
11. **Settings.** BlenderKit is off (`USE_BLENDERKIT = False`), so materials are procedural
    and the random stream does not depend on a material cache.

## Full-settings trial

2 scenes at 1024×1024, 32 samples, 4× normal supersampling, 2 GPUs: 33 views in 78 s;
per view 2.0 s main render + 2.3 s normal render; peak 3.7 GB RAM per Blender process and
4.3 GB GPU memory; all views pass `scripts/validate_generated.py`. Output ~1.75 GB per scene,
of which ~70% per-object masks and ~20% `scene.blend`.
