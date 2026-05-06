# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="allenai/objaverse",
#     repo_type="dataset",
#     local_dir="./objaverse_metadata",
#     allow_patterns="metadata/*",
#     ignore_patterns="glbs/**"
# )
from huggingface_hub import hf_hub_download, list_repo_files
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

REPO_ID = "allenai/objaverse"
REPO_TYPE = "dataset"
TARGET_PREFIX = "metadata/"   # folder you want
OUT_DIR = "./data/objaverse/metadata"
MAX_WORKERS = 16   # tune this

os.makedirs(OUT_DIR, exist_ok=True)

# get file list
files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

# filter only target folder
files = [f for f in files if f.startswith(TARGET_PREFIX)]

def download_file(f):
    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f,
            local_dir=OUT_DIR,
            resume_download=True
        )
    except Exception as e:
        return f"ERROR: {f} -> {e}"

# parallel execution
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_file, f) for f in files]

    for future in as_completed(futures):
        print(future.result())