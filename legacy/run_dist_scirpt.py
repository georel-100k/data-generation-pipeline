import os
import subprocess
import queue
import threading
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  # Ensure consistent GPU ordering
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # Start with all GPUs visible

BLENDER = "./blender/blender-4.5.9-linux-x64/blender"
SCRIPT = "blender_script.py"

NUM_GPUS = 1
NUM_SCENES = 50000

DATASET_PATH = "/workspace/data/gscanned/"
OUTPUT_ROOT = "/workspace/data/output"

# Create task queue
task_queue = queue.Queue()

for i in range(NUM_SCENES):
    task_queue.put(i)

def run_scene(gpu_id, scene_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out_dir = os.path.join(OUTPUT_ROOT, f"scene_{scene_id}")
    os.makedirs(out_dir, exist_ok=True)

    scene_type = "indoor" if scene_id % 2 == 0 else "outdoor"

    cmd = [
        BLENDER,
        "--factory-startup",
        "-b",
        "-P", SCRIPT,
        "--",
        str(scene_id),
        out_dir,
        scene_type,
        DATASET_PATH
    ]

    print(f"[GPU {gpu_id}] Running scene {scene_id}")
    subprocess.run(cmd, env=env)


def worker(gpu_id):
    while not task_queue.empty():
        scene_id = task_queue.get()

        out_dir = os.path.join(OUTPUT_ROOT, f"scene_{scene_id}")

        # ✅ Skip already completed scenes (checkpointing)
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            print(f"[GPU {gpu_id}] Skipping scene {scene_id} (done)")
            continue

        try:
            run_scene(gpu_id, scene_id)
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed scene {scene_id}: {e}")

        task_queue.task_done()


threads = []

for gpu in range(NUM_GPUS):
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("✅ All scenes completed")