import bpy
import os
import sys
import csv
import random
import math
import argparse
import numpy as np
import addon_utils

# ----------------------------
# ENABLE IMPORTERS
# ----------------------------
addon_utils.enable("io_scene_obj", default_set=True)
addon_utils.enable("io_scene_gltf2", default_set=True)
addon_utils.enable("io_scene_fbx", default_set=True)
addon_utils.enable("io_mesh_ply", default_set=True)
addon_utils.enable("io_mesh_stl", default_set=True)

# ----------------------------
# ARGPARSE
# ----------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_scenes", type=int, default=10)
parser.add_argument("--mean_objects", type=float, default=7.2)
parser.add_argument("--export_format", type=str, default="glb",
                    choices=["glb", "obj", "both"])
args = parser.parse_args(argv)

# ----------------------------
# CLEAN SCENE
# ----------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

# ----------------------------
# CSV LOADING
# ----------------------------
def load_csv(csv_path):
    entries = []
    skipped = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_path = os.path.abspath(row["filepath"])
            tex_path = row.get("texture_path", "")

            if not os.path.exists(obj_path):
                skipped += 1
                continue

            if tex_path:
                tex_path = os.path.abspath(tex_path)
                if not os.path.exists(tex_path):
                    tex_path = ""

            entries.append({"obj": obj_path, "tex": tex_path})

    print(f"[INFO] Loaded {len(entries)} valid entries, skipped {skipped}")
    return entries

def poisson_sample(mean):
    return max(1, np.random.poisson(mean))

# ----------------------------
# ROOM
# ----------------------------
def create_room(size=6):
    bpy.ops.mesh.primitive_plane_add(size=size)

    for i in range(4):
        bpy.ops.mesh.primitive_plane_add(size=size)
        wall = bpy.context.object
        wall.rotation_euler[0] = math.radians(90)

        if i == 0:
            wall.location = (0, size/2, size/2)
        elif i == 1:
            wall.location = (0, -size/2, size/2)
        elif i == 2:
            wall.location = (size/2, 0, size/2)
            wall.rotation_euler[2] = math.radians(90)
        elif i == 3:
            wall.location = (-size/2, 0, size/2)
            wall.rotation_euler[2] = math.radians(90)

# ----------------------------
# MATERIALS
# ----------------------------
def random_material():
    mat = bpy.data.materials.new(name="RandMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    bsdf.inputs["Base Color"].default_value = (
        random.random(),
        random.random(),
        random.random(),
        1
    )
    bsdf.inputs["Roughness"].default_value = random.uniform(0.2, 0.8)

    return mat

def apply_texture(obj, tex_path):
    if not tex_path:
        return False

    try:
        mat = bpy.data.materials.new(name="TexMat")
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(tex_path)

        bsdf = nodes["Principled BSDF"]
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

        obj.data.materials.clear()
        obj.data.materials.append(mat)
        return True
    except:
        return False

# ----------------------------
# UNIVERSAL IMPORT
# ----------------------------
def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()

    before = set(bpy.data.objects)

    try:
        if ext == ".obj":
            bpy.ops.wm.obj_import(filepath=path)

        elif ext in [".glb", ".gltf"]:
            bpy.ops.import_scene.gltf(filepath=path)

        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)

        elif ext == ".ply":
            bpy.ops.import_mesh.ply(filepath=path)

        elif ext == ".stl":
            bpy.ops.import_mesh.stl(filepath=path)

        else:
            print(f"[WARN] Unsupported: {path}")
            return []

    except Exception as e:
        print(f"[ERROR] Import failed: {path} | {e}")
        return []

    after = set(bpy.data.objects)
    return list(after - before)

# ----------------------------
# NORMALIZE
# ----------------------------
def normalize_object(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale = random.uniform(0.5, 1.2) / max_dim
        obj.scale = (scale, scale, scale)

# ----------------------------
# CAMERA + LIGHT
# ----------------------------
def setup_camera(size):
    bpy.ops.object.camera_add(location=(0, -size, size/2))
    cam = bpy.context.object
    cam.rotation_euler = (math.radians(75), 0, 0)
    bpy.context.scene.camera = cam

def setup_light():
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 4))
    bpy.context.object.data.energy = 1000

# ----------------------------
# FAST GPU RENDER
# ----------------------------
def setup_render(out_path):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'OPTIX'

    for d in cycles_prefs.devices:
        d.use = True

    scene.cycles.device = 'GPU'
    scene.cycles.samples = 32
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True

    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1

    scene.render.use_persistent_data = True

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = out_path

# ----------------------------
# EXPORT
# ----------------------------
def export_scene(base_path, export_format):
    if export_format in ["glb", "both"]:
        bpy.ops.export_scene.gltf(
            filepath=base_path + ".glb",
            export_format='GLB'
        )

    if export_format in ["obj", "both"]:
        bpy.ops.wm.obj_export(
            filepath=base_path + ".obj"
        )

# ----------------------------
# MAIN
# ----------------------------
entries = load_csv(args.csv)
os.makedirs(args.out, exist_ok=True)

for i in range(args.num_scenes):
    clear_scene()

    create_room()
    setup_camera(6)
    setup_light()

    k = poisson_sample(args.mean_objects)
    samples = random.choices(entries, k=k)

    placed = []

    for s in samples:
        objs = import_mesh(s["obj"])
        if not objs:
            continue

        for obj in objs:
            if obj.type != 'MESH':
                continue

            normalize_object(obj)

            obj.location = (
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                0
            )

            obj.rotation_euler[2] = random.uniform(0, 2*math.pi)

            if not apply_texture(obj, s["tex"]):
                obj.data.materials.clear()
                obj.data.materials.append(random_material())

            placed.append(obj)

    if len(placed) == 0:
        print(f"[WARN] Empty scene {i}, skipping")
        continue

    # Render
    img_path = os.path.join(args.out, f"scene_{i:05d}.png")
    setup_render(img_path)
    bpy.ops.render.render(write_still=True)

    # Export
    base_path = os.path.join(args.out, f"scene_{i:05d}")
    export_scene(base_path, args.export_format)

    print(f"[INFO] Scene {i} done")

print("Done.")