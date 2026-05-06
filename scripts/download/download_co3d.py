import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "co3d_links.tsv"
OUT_DIR = Path("data/co3d")
MAX_WORKERS = 8
TIMEOUT = 60
RETRIES = 3
MAX_SIZE_GB = 15  # skip if > 10GB
    
os.makedirs(OUT_DIR, exist_ok=True)

# ── load file ─────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, sep="\t", header=None, names=["file_name", "cdn_link"])
df = df.iloc[1:].reset_index(drop=True)

print(df.head())
print(f"Total files: {len(df)}")

# ── helper: get file size from header ─────────────────────────────────────
def get_file_size(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        size = r.headers.get("Content-Length")
        if size:
            return int(size)
    except:
        pass
    return None

# ── download function ─────────────────────────────────────────────────────
def download(row):
    filename = row["file_name"]
    url = row["cdn_link"]

    dst_path = os.path.join(OUT_DIR, filename)

    # skip existing
    if os.path.exists(dst_path):
        size_mb = os.path.getsize(dst_path) / (1024 * 1024)
        return (filename, "exists", f"{size_mb:.2f} MB")

    # ── check size BEFORE download ────────────────────────────────────────
    size_bytes = get_file_size(url)

    if size_bytes:
        size_gb = size_bytes / (1024**3)
        if size_gb > MAX_SIZE_GB:
            return (filename, "skipped_large", f"{size_gb:.2f} GB")

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    for attempt in range(RETRIES):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=TIMEOUT) as r:
                if r.status_code != 200:
                    return (filename, f"http_{r.status_code}", None)

                downloaded = 0

                with open(dst_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

            size_mb = downloaded / (1024 * 1024)
            return (filename, "ok", f"{size_mb:.2f} MB")

        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return (filename, f"error: {e}", None)

# ── parallel download ─────────────────────────────────────────────────────
results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download, row) for _, row in df.iterrows()]

    for future in as_completed(futures):
        filename, status, size = future.result()
        results.append((filename, status, size))
        print(f"{status}: {filename} | size: {size}")

# ── save manifest ─────────────────────────────────────────────────────────
manifest = pd.DataFrame(results, columns=["file_name", "status", "size"])
manifest_path = os.path.join(OUT_DIR, "download_manifest.csv")
manifest.to_csv(manifest_path, index=False)

# ── total size summary ─────────────────────────────────────────────────────
total_mb = sum(
    float(size.split()[0])
    for _, status, size in results
    if status == "ok" and size
)

print("\n────────────────────────────────────────")
print(f"Total downloaded: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
print(f"Manifest saved at: {manifest_path}")
print("Done!")
