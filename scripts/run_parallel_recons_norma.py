"""
run_parallel_camspace_gpu.py
-----------------------------
Parallel camspace EXR re-renderer — one Blender process per NVIDIA GPU.
Each Blender instance is pinned to its own GPU via CUDA_VISIBLE_DEVICES.

Auto-detects:
  - Number of NVIDIA GPUs via nvidia-smi
  - Best render engine (EEVEE_NEXT > EEVEE > Cycles)

Usage:
    python run_parallel_recons_norma.py /path/to/final_output_scenes \
        [--blender /path/to/blender] \
        [--worker-script /path/to/fix_camspace_normals.py] \
        [--dry-run]
"""

import os
import sys
import glob
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime


# ── GPU detection ─────────────────────────────────────────────────────────────

def detect_gpus() -> list:
    """
    Returns list of dicts: [{id, name, memory_mb}, ...]
    Uses nvidia-smi. Falls back to [{'id': 0, 'name': 'GPU 0'}] if unavailable.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10
        )
        print(out)
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "id":        int(parts[0]),
                "name":      parts[1],
                "memory_mb": int(parts[2]) if parts[2].isdigit() else 0,
            })
        return gpus[:4]
    except Exception as e:
        print(f"[WARN] nvidia-smi failed ({e}), assuming 1 GPU (id=0)")
        return [{"id": 0, "name": "GPU 0", "memory_mb": 0}]


# ── Folder discovery ──────────────────────────────────────────────────────────

def discover_arr_dirs(root: str) -> list:
    pattern = os.path.join(root, "scene_*", "arr_*", "scene.blend")
    return sorted(os.path.dirname(p) for p in glob.glob(pattern))


def chunk_list(lst: list, n: int) -> list:
    """Split lst into n roughly equal chunks, no empty chunks."""
    k, rem = divmod(len(lst), n)
    chunks, i = [], 0
    for c in range(n):
        size = k + (1 if c < rem else 0)
        chunks.append(lst[i:i + size])
        i += size
    return [c for c in chunks if c]


# ── Worker (one per GPU) ──────────────────────────────────────────────────────

def _gpu_worker(args):
    """
    Runs in a child process via multiprocessing.Pool.
    Processes its chunk of arr_dirs sequentially, each as a Blender subprocess
    pinned to `gpu_id` via CUDA_VISIBLE_DEVICES.

    Returns (gpu_id, [(arr_dir, success, tail), ...])
    """
    gpu_id, arr_dirs, blender_exe, worker_script = args
    results = []

    # Pin this Blender process to one GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # HIP equivalent for ROCm (no-op on NVIDIA)
    env["HIP_VISIBLE_DEVICES"]  = str(gpu_id)

    for arr_dir in arr_dirs:
        cmd = [
            blender_exe,
            "--background",
            "--python", worker_script,
            "--",
            arr_dir,
            "--gpu-id", str(gpu_id),   # passed into worker for Cycles device setup
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                env=env,
            )
            success = proc.returncode == 0
            lines   = proc.stdout.strip().splitlines()
            tail    = "\n".join(lines[-25:]) if lines else "(no output)"
            results.append((arr_dir, success, tail))

            status = "✓" if success else "✗"
            rel    = os.path.relpath(arr_dir)
            # Print immediately (subprocess output is captured, this goes to parent stdout)
            print(f"  [{status}] GPU-{gpu_id}  {rel}", flush=True)
            if not success:
                print(f"       --- tail ---\n{tail}\n", flush=True)

        except subprocess.TimeoutExpired:
            results.append((arr_dir, False, "TIMEOUT after 600s"))
            print(f"  [✗] GPU-{gpu_id}  TIMEOUT  {arr_dir}", flush=True)
        except Exception as e:
            results.append((arr_dir, False, str(e)))
            print(f"  [✗] GPU-{gpu_id}  ERROR: {e}  {arr_dir}", flush=True)

    return gpu_id, results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root",            help="Root dataset directory")
    parser.add_argument("--blender",       default="blender")
    parser.add_argument("--worker-script", default=None)
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: root not found: {root}"); sys.exit(1)

    worker_script = args.worker_script or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fix_camspace_normals.py"
    )
    if not os.path.exists(worker_script):
        print(f"ERROR: worker script not found: {worker_script}"); sys.exit(1)

    gpus     = detect_gpus()
    arr_dirs = discover_arr_dirs(root)

    if not arr_dirs:
        print(f"No scene.blend files found under {root}"); sys.exit(1)

    n_gpus   = len(gpus)
    chunks   = chunk_list(arr_dirs, n_gpus)

    print(f"\n{'='*65}")
    print(f"  Root        : {root}")
    print(f"  Arr folders : {len(arr_dirs)}")
    print(f"  GPUs        : {n_gpus}")
    for g in gpus:
        mem = f"  {g['memory_mb']} MB" if g['memory_mb'] else ""
        print(f"    GPU {g['id']}: {g['name']}{mem}")
    print(f"  Blender     : {args.blender}")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    if args.dry_run:
        for i, (gpu, chunk) in enumerate(zip(gpus, chunks)):
            print(f"GPU {gpu['id']} ({gpu['name']}) — {len(chunk)} folders:")
            for d in chunk[:5]:
                print(f"    {os.path.relpath(d, root)}")
            if len(chunk) > 5:
                print(f"    ... and {len(chunk)-5} more")
        return

    worker_args = [
        (gpus[i]["id"], chunks[i], args.blender, worker_script)
        for i in range(len(chunks))
    ]

    t_start     = datetime.now()
    all_results = []

    with multiprocessing.Pool(processes=n_gpus) as pool:
        for gpu_id, results in pool.imap_unordered(_gpu_worker, worker_args):
            all_results.extend(results)

    t_end     = datetime.now()
    elapsed   = (t_end - t_start).total_seconds()
    succeeded = [r for r in all_results if r[1]]
    failed    = [r for r in all_results if not r[1]]

    print(f"\n{'='*65}")
    print(f"  Finished    : {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Elapsed     : {elapsed:.0f}s "
          f"({elapsed/max(len(arr_dirs),1):.1f}s avg per folder)")
    print(f"  Succeeded   : {len(succeeded)} / {len(all_results)}")
    print(f"  Failed      : {len(failed)}")

    if failed:
        log_path = os.path.join(root, "camspace_gpu_failures.log")
        with open(log_path, "w") as f:
            for arr_dir, _, tail in failed:
                f.write(f"\n{'='*60}\n{arr_dir}\n{tail}\n")
        print(f"  Failure log : {log_path}")

    print(f"{'='*65}\n")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
