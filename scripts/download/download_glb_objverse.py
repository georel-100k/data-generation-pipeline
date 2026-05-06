import objaverse
import pandas as pd
import os
import json
import shutil
from tqdm import tqdm

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('/workspace/data/objaverse/gemma_classified.csv')

with open('/workspace/data/objaverse/object-paths.json') as f:
    obj_path = json.load(f)

OUT_DIR = "/workspace/data/objaverse/glbs"
BATCH_SIZE = 100
PROCESSES = 10

os.makedirs(OUT_DIR, exist_ok=True)
try:
    df = pd.read_csv('/workspace/data/objaverse/gemma_sampled.csv')
    print("Loaded sampled CSV with gemma_location_type ✓")
except Exception as e:
    print(f"Error reading sampled CSV: {e}")
    df = pd.concat([
        df[df['gemma_location_type'] == 'both'].sample(10000),
        df[df['gemma_location_type'] == 'outdoor'].sample(10000),
        df[df['gemma_location_type'] == 'indoor'].sample(10000)
    ]).reset_index(drop=True)

uids = df["filename"].dropna().unique().tolist()
print(f"Queued {len(uids)} UIDs for download")
import random
# ── results storage ──────────────────────────────────────────────────────────
results = {"filename": [], "local_path": [], "status": []}
# check if uids are in output directory (in case of re-run)
existing_files = set(os.listdir(OUT_DIR))
new_uids = []
for uid in uids:
    expected_filename = f"{uid}.glb"
    if expected_filename in existing_files:
        continue
    new_uids.append(uid)
uids = new_uids
random.sample(uids, min(10, len(uids)))  
print(f"{len(uids)} UIDs to download after filtering existing files")

# ── batching loop ────────────────────────────────────────────────────────────
for i in tqdm(range(0, len(uids), BATCH_SIZE)):
    batch_uids = uids[i:i + BATCH_SIZE]
    print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({len(batch_uids)} items)")

    try:
        objects = objaverse.load_objects(
            uids=batch_uids,
            download_processes=PROCESSES,
        )
    except Exception as e:
        print(f"Batch failed: {e}")
        for uid in batch_uids:
            results["filename"].append(uid)
            results["local_path"].append(None)
            results["status"].append(f"batch_error: {e}")
        continue

    # ── move files ───────────────────────────────────────────────────────────
    for uid in batch_uids:
        src_path = objects.get(uid)
        dst_path = os.path.join(OUT_DIR, f"{uid}.glb")

        try:
            if src_path and os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                results["filename"].append(uid)
                results["local_path"].append(dst_path)
                results["status"].append("ok")
                print(f"✓ {uid}")
            else:
                results["filename"].append(uid)
                results["local_path"].append(None)
                results["status"].append("missing_src")
                print(f"✗ {uid} — missing")
        except Exception as e:
            results["filename"].append(uid)
            results["local_path"].append(None)
            results["status"].append(f"error: {e}")
            print(f"✗ {uid} — {e}")

# ── save manifest ────────────────────────────────────────────────────────────
manifest = pd.DataFrame(results)
manifest.to_csv(os.path.join(OUT_DIR, "download_manifest.csv"), index=False)

print(f"\nDone. {manifest['status'].eq('ok').sum()}/{len(uids)} files downloaded.")