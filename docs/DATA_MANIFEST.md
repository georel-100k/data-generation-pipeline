# Asset Manifest

`pipeline/blender_generator.py` reads a CSV manifest before rendering. Each row is
one candidate 3D object.

## Required Columns

| Column | Description |
| --- | --- |
| `filepath` | Absolute or working-directory-relative path to a `.glb`, `.gltf`, `.obj`, or `.fbx` file. |
| `gemma_location_type` | Location label used for filtering. The canonical indoor generator keeps `indoor` and `both`. |

## Recommended Columns

| Column | Description |
| --- | --- |
| `filename` | Stable asset id, such as an Objaverse UID or ABO item id. |
| `object_name` | Human-readable name used in logs and `scene_graph.json`. |
| `texture_filepath` | Optional texture image path. If empty, the generator uses BlenderKit or procedural materials. |
| `class` | Optional semantic class written into `scene_graph.json`. |

## Example

See `examples/asset_manifest.example.csv`.

```csv
filename,object_name,filepath,texture_filepath,gemma_location_type,class
chair_001,wood chair,/abs/path/to/chair.glb,,indoor,chair
lamp_001,table lamp,/abs/path/to/lamp.obj,/abs/path/to/lamp_texture.png,both,lamp
```

## Building a Manifest from a Folder

For a quick smoke test, scan any local folder of assets:

```bash
python scripts/prepare_asset_manifest.py /path/to/assets --out data/assets.csv
```

This creates absolute `filepath` entries and assigns `indoor` to
`gemma_location_type` by default. Use `--default-location both` if the folder
contains assets suitable for both indoor and outdoor scenes.
