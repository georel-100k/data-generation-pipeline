# Asset manifest

`pipeline/blender_pipeline.py` samples objects from `manifest/asset_manifest.csv` (in this
repository). Each row is one candidate 3D object. The generator keeps rows whose file
exists in the data root and whose `gemma_location_type` is `indoor` or `both`.

## Columns

| Column | Used by the generator | Description |
|---|---|---|
| `filepath` | **required** | Path to a `.glb`, `.gltf`, `.obj` or `.fbx` file, relative to the data root (absolute paths are used as given). |
| `gemma_location_type` | **required** | `indoor`, `outdoor`, `both` or `unknown` (Gemma classification); only `indoor` and `both` are sampled. |
| `object_name` | scene graph | Human-readable name, written to `scene_graph.json`. |
| `class` | scene graph | Class label, written to `scene_graph.json`. |
| `texture_filepath` | materials | Texture image (GSO only), relative to the data root. If empty, the object gets a procedural material. |
| `filename` | — | Stable asset id: Objaverse UID or ABO item id (empty for GSO). |
| `license` | — | Asset license in Objaverse's codes: `by`, `by-sa`, `by-nc`, `by-nc-sa`, `cc0`. |
| `categories`, `item_id`, `description`, `gemma_class` | — | Source metadata kept from the classification step. |

## Current manifest

28,576 rows. Paths are relative to the data root; `scripts/setup_assets.py` downloads every
asset into place and reports anything missing.

| Source | Rows | In the generator pool (indoor/both) | Location |
|---|---|---|---|
| Objaverse | 19,586 | 15,122 (64%) | `assets/objaverse/glbs/<uid>.glb` |
| Amazon Berkeley Objects (ABO) | 7,960 | 7,495 (32%) | `assets/abo/3dmodels/original/<x>/<item_id>.glb` |
| Google Scanned Objects (GSO) | 1,030 | 1,022 (4%) | `assets/gscanned/<name>/meshes/model.obj` (+ texture) |
| **Total** | **28,576** | **23,639** | |

Licenses (manifest / pool): `by` 25,006 / 20,843, `by-nc-sa` 1,864 / 1,438,
`by-nc` 924 / 736, `by-sa` 628 / 509, `cc0` 154 / 113. There are no ND (no-derivatives)
licenses. ABO and GSO are CC BY 4.0; Objaverse licenses are per object.

Known limitations of the labels: ABO `class` is the last product keyword (e.g. `round`)
rather than a category, Objaverse `class` is the first user tag, and ABO indoor/outdoor
labels were inferred from the product description only.

## How it was built

Steps 1–3 are history: their scripts and the classification notebook
(`notebooks/indoor_outdoor_classifier.ipynb`, with its label checkpoint
`notebooks/artifacts/gemma_checkpoint_gscanned.parquet`) are no longer in the working tree
but remain in the repository history (commit `e75ea93`). Their result is
`manifest/source/gemma_classified_expanded.csv`; only step 4 is needed to reproduce the
manifest.

1. **Source metadata**: Objaverse metadata
   (19,586 objects with tags, from a 30,000-object sample), ABO listings and 3D models, and
   GSO metadata were merged into `final_metadata.csv` (28,630 rows, including 54 ShapeNet
   category rows).
2. **Classification** (classifier notebook): each row was labelled
   `indoor` / `outdoor` / `both` with `google/gemma-3-4b-it` (greedy decoding). The labels
   used are in the notebook's label checkpoint (matched by row index); notebook cell 28
   then expanded ShapeNet categories into one row per model,
   giving `gemma_classified_expanded.csv`.
3. **Recovery**: the original expanded manifest was not kept. It was rebuilt by re-running
   notebook cells 16 → 24 → 28 on `final_metadata.csv` and the checkpoint above; label
   counts match the notebook's saved output exactly (20,527 indoor / 4,936 outdoor /
   3,166 both / 1 unknown) and all objects of the released subset are present with an
   indoor/both label. The recovered expanded manifest is
   `manifest/source/gemma_classified_expanded.csv`.
4. **Build** (`scripts/build_manifest.py`, one step), from the recovered expanded manifest:
   - makes all paths relative to the data root (`assets/<source>/…`);
   - drops all ShapeNet rows (52,376): ShapeNet is not used for generation, and in the
     original every ShapeNet row had an empty `filepath` because notebook cell 28 copied
     `texture_filepath` instead of the exploded `to_expand_filepath`;
   - fixes two GSO folders whose names lost their non-ASCII characters
     (`Pokémon_…` on disk, `Pokmon_…` in the paths), recovering the name from
     `object_name`;
   - adds the `license` column: ABO and GSO `by`; Objaverse from
     `manifest/source/objaverse_licenses.csv`, which is fetched from the `allenai/objaverse`
     metadata if it is missing (or with `--refresh-licenses`; ≈550 MB of metadata shards,
     cached in `<data root>/cache/`).

```bash
python scripts/build_manifest.py        # writes manifest/asset_manifest.csv
```

It needs no assets on disk, keeps the row order (sampling depends on it), and is
idempotent: running it on its own output changes nothing.

## A manifest for other assets

The generator accepts any CSV with at least `filepath` and `gemma_location_type`
(`--manifest PATH`). Relative `filepath`s are resolved from the data root; absolute ones are
used as given. For example:

```csv
filename,object_name,filepath,texture_filepath,gemma_location_type,class,license
chair_001,wood chair,/absolute/path/to/chair.glb,,indoor,chair,by
```
