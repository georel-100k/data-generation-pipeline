import os
import bpy
import os
import sys

def get_input_folder():
    # Blender passes args after '--'
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Please provide folder path after '--'")

    folder = argv[argv.index("--") + 1]

    if not os.path.isdir(folder):
        raise RuntimeError(f"Invalid folder: {folder}")

    return os.path.abspath(folder)


def merge_all_meshes():
    bpy.ops.object.select_all(action='DESELECT')

    mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not mesh_objs:
        print("No mesh objects found.")
        return None

    for obj in mesh_objs:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()

    return bpy.context.active_object


def export_obj(obj, folder):
    folder_name = os.path.basename(folder.rstrip("/"))
    filepath = os.path.join(folder, f"{folder_name}.obj")

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.export_scene.obj(
        filepath=filepath,
        use_selection=True,
        use_materials=True
    )

    print(f"Exported to: {filepath}")


# --- RUN ---
input_folder = get_input_folder()

merged_obj = merge_all_meshes()

if merged_obj:
    export_obj(merged_obj, input_folder)
def find_obj_files(root_dir):
    results = []

    root_dir = os.path.abspath(root_dir)
    root_name = os.path.basename(root_dir.rstrip("/"))

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".obj"):
                obj_path = os.path.join(dirpath, file)

                # ID = parent folder of the .obj
                obj_id = os.path.basename(dirpath)

                results.append({
                    "root_folder": root_name,
                    "id": obj_id,
                    "obj_path": obj_path
                })

    return results


# --- RUN ---
root_folder = "/path/to/root"   # <-- change this
data = find_obj_files(root_folder)

# Print results
for item in data:
    print(item)