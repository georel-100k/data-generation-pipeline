# Output format

What `pipeline/blender_pipeline.py` writes, and the conventions every file follows.

## Directory layout

```text
<data root>/<output>/                      generated/ (full run) or generated_test/ (--test)
└── scene_00001/                             one set of objects (sizes and materials fixed)
    ├── arr_001/                           one layout of those objects (3 per scene)
    │   ├── rgb/<view>.png
    │   ├── depth/<view>_zdepth.exr
    │   ├── depth/<view>_euclidean.exr
    │   ├── normals/<view>_camspace.exr
    │   ├── normals/<view>_worldspace.exr
    │   ├── masks/<view>_combined.png
    │   ├── masks/<view>_obj001.png …       one per object
    │   ├── colmap/cameras.txt
    │   ├── colmap/images.txt
    │   ├── colmap/points3D.txt             empty (poses only)
    │   ├── metadata.json
    │   ├── scene_graph.json
    │   └── scene.blend
    ├── arr_002/
    └── arr_003/
```

- **Scene**: 1–12 objects sampled from the asset manifest (Poisson, mean 7.3), with their
  materials. Object sizes are set in the first arrangement and stay identical in all
  arrangements of the scene.
- **Arrangement**: a different layout of the same objects (position and rotation about the
  vertical axis), with its own cameras and lights. Each `arr_XXX/` folder is self-contained.
- **View**: one camera. Every file of a view starts with the view label
  `camNN_az<azimuth>_el<elevation>`, e.g. `cam00_az299.0_el45.7` (degrees). A scene has
  6–24 views in total, split across its arrangements.

## Coordinate conventions

| Frame | Convention |
|---|---|
| World | Blender world: right-handed, **Z up**, metres. Floor at z = 0, room centred on the origin, walls' inner faces at x, y = ±2.2 m (`FLOOR_HALF`), wall height 1.9 m. Scene-graph compass: +X = east, +Y = north. |
| Camera | **OpenCV / COLMAP**: x right, y down, z forward (viewing direction). |
| Pose | World-to-camera `x_cam = R · X + t`; camera centre `C = −Rᵀ t`. `R`, `t` are in `metadata.json` (`R_w2c_colmap`, `t_w2c_colmap_m`) and, as a quaternion, in `colmap/images.txt`. |
| Pixels | Pinhole: `u = fx·x/z + cx`, `v = fy·y/z + cy`; 70° field of view, principal point at the image centre. |

Object sizes are intentionally not physical: every object's largest side is scaled to
`TARGET_SIZE_MIN`–`TARGET_SIZE_MAX` regardless of its class (the dataset targets cartoon
scene rendering), then shrunk if needed to fit the room.

## Per-view files

All EXRs are 32-bit float with channels R, G, B (OpenCV reads them as BGR). Masks are 8-bit
grayscale PNGs (lossless).

| File | Content | Background (no geometry) |
|---|---|---|
| `rgb/<view>.png` | 8-bit RGBA colour render (alpha always 255). Cycles, OIDN denoised, Blender's default view transform. | world colour (grey) |
| `depth/<view>_zdepth.exr` | Planar depth: distance along the camera's optical axis, metres. R = G = B. | `1e10` |
| `depth/<view>_euclidean.exr` | Distance from the camera centre along each pixel's ray, metres: `zdepth · ‖((u+0.5−cx)/fx, (v+0.5−cy)/fy, 1)‖`. Pixel-aligned with zdepth. R = G = B. | `≥ 1e10` |
| `normals/<view>_camspace.exr` | Unit surface normal in the **camera frame** (OpenCV), R = x, G = y, B = z, in [−1, 1]. Surfaces facing the camera have z < 0. | `(0, 0, 0)` |
| `normals/<view>_worldspace.exr` | Unit surface normal in the **world frame**, encoded `v = 0.5·n + 0.5` (decode `n = 2v − 1`). | `(0, 0, 0)` (decodes to a non-unit vector; test for all-zero) |
| `masks/<view>_combined.png` | 8-bit grayscale; pixel value = object ID: `0` = room or background, `k` = object `k` (at most 12 per scene). | `0` |
| `masks/<view>_objNNN.png` | 8-bit grayscale binary mask of object `NNN`: 255 where visible, 0 elsewhere (equal to `combined == NNN`). | `0` |

Normals come from a supersampled pass: 4×4 rays per pixel (`NORMAL_SUPERSAMPLE`), each
pixel averaging only the rays that hit its majority object, then renormalised. Fine
geometry is averaged instead of aliased, object boundaries stay crisp, and
`camspace = R_w2c_colmap · decode(worldspace)` exactly. RGB, depth and masks come from the
main render (anti-aliased for RGB only).

## Per-arrangement files

### `colmap/` (COLMAP text format)

- `cameras.txt`: one line per view, `camera_id PINHOLE width height fx fy cx cy`. All views
  share the same intrinsics but each has its own `camera_id`.
- `images.txt`: two lines per view. Line 1:
  `image_id QW QX QY QZ TX TY TZ camera_id name`, the world-to-camera rotation
  (unit quaternion, scalar first) and translation in metres, `name` = `<view>.png`.
  Line 2 (2D keypoints) is always empty.
- `points3D.txt`: only a comment line (no sparse reconstruction).

### `metadata.json`

```text
render                   {resolution_x, resolution_y, num_cameras}
cameras[]                one per view
  id, label              0-based index; view label (file prefix)
  intrinsics             fx_px, fy_px, cx_px, cy_px, fov_h_deg, fov_v_deg
  extrinsics             position_world_m, R_w2c_colmap (3x3), t_w2c_colmap_m
objects[]                one per object
  id                     1-based; = mask value, objNNN, scene-graph node id
  bbox_min_m, bbox_max_m world axis-aligned bounding box (see note)
  center_3d_m            box centre
  dimensions_m           box extents (x, y change with the object's rotation; z does not)
  footprint_radius_m     half the diagonal of the box's x-y footprint
inter_object_distances[] one per unordered pair
  object_ids, center_to_center_3d_m, center_to_center_2d_m,
  surface_to_surface_approx_m (= max(0, 2D centre distance − both footprint radii)),
  delta_z_m
```

Bounding boxes are built from each mesh part's own bounding-box corners transformed to
world space, so they always contain the object but can be a few centimetres larger than
the mesh for rotated parts.

### `scene_graph.json`

```text
scene_id                 e.g. "scene_00001_arr_001"
num_objects
nodes[]                  one per object
  id                     same id as metadata.json / masks
  object_name, filepath, class      from the asset manifest (filepath relative to the data root)
  center_3d_m, bbox_min_m, bbox_max_m, dimensions_m, footprint_radius_m
  material_name
edges[]                  directed, both directions: n(n−1) edges
  source, target
  distances_m            center_3d, center_2d, surface_to_surface, delta_z
  relations              how TARGET sits relative to SOURCE:
    proximity            adjacent (surface gap < 0.3 m) | near (< 1.5 m) | far
    vertical             above | below | same_level
    horizontal           east_of | north_of | west_of | south_of (world compass)
adjacency                {source_id: {target_id: surface_to_surface_m}}, symmetric
```

Asset identity (`filepath`, `class`, `object_name`) is only in the scene graph;
`metadata.json` links to it through `id`.

### `scene.blend`

The Blender scene at the end of the arrangement: room, objects with materials (most
textures packed), cameras and lights. Useful for re-rendering; it embeds the source meshes, so it is
subject to the assets' licenses. Saved with Blender's file compression (`BLEND_COMPRESS`), which
Blender reads transparently; unused materials and images are removed before saving.

## Sizes

At 1024×1024 with ≈12 views per scene: ≈0.1 GB per scene of images (per view ≈5.8 MB depth,
≈1.4 MB normals, ≈1.1 MB RGB, ≈0.02 MB masks), plus `scene.blend` files, which depend on the
objects and average ≈0.1–0.16 GB per scene compressed (≈0.4–0.5 GB uncompressed).
