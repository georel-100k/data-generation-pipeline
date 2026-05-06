"""
fix_camspace_worker_gpu.py
--------------------------
GPU-aware Blender worker. Called by run_parallel_camspace_gpu.py.
Processes ONE arr folder. Pinned to a specific GPU via CUDA_VISIBLE_DEVICES
(set by the launcher) + explicit Cycles device selection inside Blender.

Usage (via launcher — don't call directly):
    blender --background --python fix_camspace_worker_gpu.py -- /path/to/arr --gpu-id 2
"""

import bpy
import os
import sys
import glob
import shutil


# ── Engine setup (Cycles/CUDA only) ──────────────────────────────────────────

def _configure_engine(scene: bpy.types.Scene, gpu_id: int) -> str:
    """
    Forces Cycles on the CUDA GPU pinned via CUDA_VISIBLE_DEVICES.
    CUDA_VISIBLE_DEVICES=N (set by the launcher) remaps that GPU to index 0
    inside this process, so we always enable cuda_devs[0].
    Returns "CYCLES".
    """
    scene.render.engine  = "CYCLES"
    scene.cycles.device  = "GPU"
    scene.cycles.samples = 1        # emission shader — 1 sample is exact, instant

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()             # force Blender to enumerate devices

    cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
    cpu_devs  = [d for d in prefs.devices if d.type == "CPU"]

    if cuda_devs:
        # Disable everything, then enable only the remapped GPU (index 0)
        for d in prefs.devices:
            d.use = False
        cuda_devs[0].use = True
        print(f"  [ENGINE] Cycles/CUDA  GPU-{gpu_id}  device: {cuda_devs[0].name}")
    else:
        # No CUDA visible in this process — shouldn't happen if launcher set
        # CUDA_VISIBLE_DEVICES correctly, but handle gracefully
        print(f"  [WARN] No CUDA devices found for GPU-{gpu_id}, falling back to CPU")
        for d in cpu_devs:
            d.use = True
        scene.cycles.device = "CPU"

    return "CYCLES"


# ── Material ──────────────────────────────────────────────────────────────────

def _make_camera_normal_material() -> bpy.types.Material:
    """
    Full 3-channel camera-space normal material.
    World → Camera transform, remap -1..1 → 0..1, output via Emission.
    """
    mat = bpy.data.materials.new("_fix_cam_normal_gpu")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    geo   = nodes.new("ShaderNodeNewGeometry")
    trans = nodes.new("ShaderNodeVectorTransform")
    remap = nodes.new("ShaderNodeVectorMath")
    emit  = nodes.new("ShaderNodeEmission")
    out   = nodes.new("ShaderNodeOutputMaterial")

    trans.vector_type  = "NORMAL"
    trans.convert_from = "WORLD"
    trans.convert_to   = "VIEW"     # Blender's enum for camera space — "CAMERA" is invalid

    # v * 0.5 + 0.5  →  maps -1..1 to 0..1
    remap.operation = "MULTIPLY_ADD"
    remap.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap.inputs[2].default_value = (0.5, 0.5, 0.5)

    links.new(geo.outputs["Normal"],    trans.inputs["Vector"])
    links.new(trans.outputs["Vector"],  remap.inputs[0])
    links.new(remap.outputs["Vector"],  emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    # Confirm enum took effect — will print valid values if something is wrong
    valid = list(trans.bl_rna.properties["convert_to"].enum_items.keys())
    print(f"  [MAT] VectorTransform: {trans.convert_from} → {trans.convert_to}  "
          f"(valid: {valid})")

    return mat


# ── Render setup ──────────────────────────────────────────────────────────────

def _setup_exr_output(scene: bpy.types.Scene) -> None:
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode  = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec   = "ZIP"
    scene.use_nodes = False


# ── Camera discovery ──────────────────────────────────────────────────────────

def _find_cameras(arr_dir: str, scene) -> list:
    """
    Returns [(camera_obj, label, out_stem), ...].
    Matches existing normals/*_camspace.exr filenames to scene camera names.
    Falls back to all scene cameras.
    """
    normals_dir = os.path.join(arr_dir, "normals")
    pairs       = []

    existing = sorted(glob.glob(os.path.join(normals_dir, "*_camspace.exr")))

    if existing:
        for exr_path in existing:
            label    = os.path.basename(exr_path).replace("_camspace.exr", "")
            out_stem = os.path.join(normals_dir, f"{label}_camspace")

            cam_obj = scene.objects.get(label)
            if cam_obj and cam_obj.type == "CAMERA":
                pairs.append((cam_obj, label, out_stem))
                continue

            # Partial match
            matches = [o for o in scene.objects
                       if o.type == "CAMERA" and label in o.name]
            if matches:
                pairs.append((matches[0], label, out_stem))
            else:
                print(f"  [WARN] No camera for label '{label}' — skipping")

    if not pairs:
        print("  [INFO] Fallback: using all scene cameras")
        os.makedirs(normals_dir, exist_ok=True)
        for obj in scene.objects:
            if obj.type == "CAMERA":
                out_stem = os.path.join(normals_dir, f"{obj.name}_camspace")
                pairs.append((obj, obj.name, out_stem))

    return pairs


# ── Core ──────────────────────────────────────────────────────────────────────

def process_arr(arr_dir: str, gpu_id: int) -> None:
    blend_path = os.path.join(arr_dir, "scene.blend")
    if not os.path.exists(blend_path):
        print(f"[SKIP] No scene.blend: {arr_dir}")
        return

    print(f"[START] GPU-{gpu_id}  {arr_dir}")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene

    engine = _configure_engine(scene, gpu_id)
    _setup_exr_output(scene)

    cameras = _find_cameras(arr_dir, scene)
    if not cameras:
        print(f"[SKIP] No cameras resolved in {arr_dir}")
        return

    mat = _make_camera_normal_material()
    vl  = scene.view_layers[0]

    for cam_obj, label, out_stem in cameras:
        canonical = out_stem + ".exr"

        # Backup broken single-channel file
        if os.path.exists(canonical):
            shutil.copy2(canonical, canonical + ".bak")

        # Render
        vl.material_override  = mat
        scene.camera          = cam_obj
        scene.render.filepath = out_stem
        bpy.ops.render.render(write_still=True)
        vl.material_override  = None

        # Rename Blender's auto-numbered output → canonical name
        numbered = out_stem + "0001.exr"
        if os.path.exists(numbered):
            os.replace(numbered, canonical)
            print(f"  [OK] {label}_camspace.exr  ({engine}, GPU {gpu_id})")
        elif os.path.exists(canonical):
            print(f"  [OK] {label}_camspace.exr  (no rename needed)")
        else:
            print(f"  [WARN] Expected output missing: {numbered}")

    bpy.data.materials.remove(mat)
    print(f"[DONE] GPU-{gpu_id}  {arr_dir}  ({len(cameras)} cameras)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python fix_camspace_worker_gpu.py "
              "-- /path/to/arr [--gpu-id N]")
        sys.exit(1)

    rest = argv[argv.index("--") + 1:]

    # Parse --gpu-id N
    gpu_id  = 0
    arr_dir = None
    i = 0
    while i < len(rest):
        if rest[i] == "--gpu-id" and i + 1 < len(rest):
            gpu_id = int(rest[i + 1])
            i += 2
        else:
            arr_dir = rest[i]
            i += 1

    if not arr_dir:
        print("ERROR: no arr_dir provided")
        sys.exit(1)

    arr_dir = os.path.abspath(arr_dir)
    if not os.path.isdir(arr_dir):
        print(f"ERROR: not a directory: {arr_dir}")
        sys.exit(1)

    process_arr(arr_dir, gpu_id)


if __name__ == "__main__":
    main()